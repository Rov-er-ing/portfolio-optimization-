"""
Comprehensive Test Suite — MLPO v1.0.

Tests verify:
    1. Tensor shape correctness at every module boundary.
    2. Gradient flow (∂L/∂θ is not None for layer 0 parameters).
    3. Financial constraint enforcement (PSD covariance, VIX override, weight bounds).
    4. Metric accuracy against known values.

Run: pytest mlpo/tests/ -v
"""

import pytest
import torch
import numpy as np

from mlpo.config import config, seed_everything


# ═══════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def set_seed():
    """Ensure deterministic tests."""
    seed_everything(42)


@pytest.fixture
def dummy_asset_features():
    """Dummy asset features: [B, Seq, P, F]."""
    return torch.randn(
        config.BATCH_SIZE, config.SEQUENCE_LENGTH,
        config.N_ASSETS, config.N_FEATURES,
    )


@pytest.fixture
def dummy_macro_features():
    """Dummy macro features: [B, Seq, N_macro]."""
    return torch.randn(
        config.BATCH_SIZE, config.SEQUENCE_LENGTH,
        config.N_MACRO,
    )


@pytest.fixture
def dummy_vix_normal():
    """VIX below threshold."""
    return torch.full((config.BATCH_SIZE,), 20.0)


@pytest.fixture
def dummy_vix_crisis():
    """VIX above threshold (trigger override)."""
    return torch.full((config.BATCH_SIZE,), 40.0)


# ═══════════════════════════════════════════════════
# Module A: LSTM Covariance Forecaster
# ═══════════════════════════════════════════════════

class TestLSTMCovariance:
    """Tests for the Cholesky-enforced covariance forecaster."""

    def test_output_shape(self, dummy_asset_features):
        from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
        model = LSTMCovarianceForecaster()
        sigma = model(dummy_asset_features)
        assert sigma.shape == (config.BATCH_SIZE, config.N_ASSETS, config.N_ASSETS), (
            f"Expected [{config.BATCH_SIZE}, {config.N_ASSETS}, {config.N_ASSETS}], "
            f"got {sigma.shape}"
        )

    def test_positive_semi_definite(self, dummy_asset_features):
        """Covariance matrix must have all eigenvalues >= 0."""
        from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
        model = LSTMCovarianceForecaster()
        sigma = model(dummy_asset_features)

        # Check PSD: all eigenvalues >= 0
        eigenvalues = torch.linalg.eigvalsh(sigma)
        min_eig = eigenvalues.min().item()
        assert min_eig > 0, (
            f"Covariance matrix is not PSD: min eigenvalue = {min_eig}"
        )

    def test_symmetry(self, dummy_asset_features):
        """Σ must equal Σᵀ."""
        from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
        model = LSTMCovarianceForecaster()
        sigma = model(dummy_asset_features)
        diff = (sigma - sigma.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, f"Covariance not symmetric: max diff = {diff}"

    def test_gradient_flow(self, dummy_asset_features):
        """Gradients must flow back to LSTM layer 0."""
        from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
        model = LSTMCovarianceForecaster()
        sigma = model(dummy_asset_features)
        loss = sigma.sum()
        loss.backward()

        # Check first LSTM layer has gradients
        first_param = list(model.lstm.parameters())[0]
        assert first_param.grad is not None, (
            "No gradient on LSTM layer 0 — gradient flow is broken"
        )
        assert first_param.grad.abs().sum() > 0, (
            "Gradient is all zeros — vanishing gradient detected"
        )


# ═══════════════════════════════════════════════════
# Module B: Regime Detector
# ═══════════════════════════════════════════════════

class TestRegimeDetector:
    """Tests for LSTM-GMM regime detection and VIX override."""

    def test_output_shapes(self, dummy_macro_features, dummy_vix_normal):
        from mlpo.models.regime_detector import RegimeDetector
        model = RegimeDetector()
        beta, regime_probs = model(dummy_macro_features, dummy_vix_normal)

        assert beta.shape == (config.BATCH_SIZE, config.N_ASSETS)
        assert regime_probs.shape == (config.BATCH_SIZE, config.N_REGIMES)

    def test_risk_budget_sums_to_one(self, dummy_macro_features, dummy_vix_normal):
        """Risk budgets must sum to 1 (valid probability distribution)."""
        from mlpo.models.regime_detector import RegimeDetector
        model = RegimeDetector()
        beta, _ = model(dummy_macro_features, dummy_vix_normal)
        sums = beta.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Risk budgets do not sum to 1: {sums}"
        )

    def test_vix_override_forces_defensive(
        self, dummy_macro_features, dummy_vix_crisis
    ):
        """When VIX >= 35, defensive allocation must be >= 40%."""
        from mlpo.models.regime_detector import RegimeDetector
        model = RegimeDetector()
        beta, _ = model(dummy_macro_features, dummy_vix_crisis)

        defensive_alloc = beta[:, config.DEFENSIVE_INDICES].sum(dim=-1)
        for i in range(config.BATCH_SIZE):
            assert defensive_alloc[i].item() >= config.DEFENSIVE_ALLOC - 0.01, (
                f"Sample {i}: defensive={defensive_alloc[i]:.4f} "
                f"< required {config.DEFENSIVE_ALLOC}"
            )


# ═══════════════════════════════════════════════════
# Module D: Sparse Attention
# ═══════════════════════════════════════════════════

class TestSparseAttention:
    """Tests for O(pk) sparse attention."""

    def test_output_shape(self):
        from mlpo.models.attention import SparseAttention
        model = SparseAttention(embed_dim=256, n_heads=4, k=15)
        x = torch.randn(config.BATCH_SIZE, config.N_ASSETS, 256)
        out = model(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_mask_sparsity(self):
        """Verify mask has at most k+1 True entries per row."""
        from mlpo.models.attention import SparseAttention
        model = SparseAttention(embed_dim=256, n_heads=4, k=15)

        # Set dummy neighborhoods
        neighborhoods = {
            asset: config.ASSET_LIST[:15]
            for asset in config.ASSET_LIST
        }
        model.set_neighborhoods(neighborhoods, config.ASSET_LIST)

        per_row = model.neighbor_mask.sum(dim=1)
        assert per_row.max().item() <= 16, (  # k=15 + self
            f"Mask too dense: max neighbors = {per_row.max().item()}"
        )


# ═══════════════════════════════════════════════════
# Financial Metrics
# ═══════════════════════════════════════════════════

class TestMetrics:
    """Tests for financial metric accuracy."""

    def test_sharpe_known_value(self):
        """Sharpe of constant excess return = return / vol * √252."""
        from mlpo.backtest.metrics import sharpe_ratio
        # Daily return of 0.1% with ~0 vol → high Sharpe
        returns = np.full(252, 0.001)
        s = sharpe_ratio(returns, rf=0.0)
        assert s > 10.0, f"Expected high Sharpe for constant returns, got {s}"

    def test_max_drawdown_known(self):
        """50% drawdown from known sequence."""
        from mlpo.backtest.metrics import max_drawdown
        # Price: 1.0 → 2.0 → 1.0 (50% drawdown)
        returns = np.array([1.0, -0.5])
        mdd = max_drawdown(returns)
        assert abs(mdd - 0.5) < 0.01, f"Expected MDD ≈ 0.50, got {mdd:.4f}"

    def test_cvar_more_extreme_than_var(self):
        """CVaR should be <= VaR (more negative)."""
        from mlpo.backtest.metrics import value_at_risk, conditional_var
        returns = np.random.normal(0, 0.02, 1000)
        var = value_at_risk(returns, 0.05)
        cvar = conditional_var(returns, 0.05)
        assert cvar <= var, f"CVaR ({cvar}) should be <= VaR ({var})"


# ═══════════════════════════════════════════════════
# Composite Loss
# ═══════════════════════════════════════════════════

class TestCompositeLoss:
    """Tests for the financial loss function."""

    def test_loss_is_differentiable(self):
        """Loss must support backward()."""
        from mlpo.training.loss import CompositeFinancialLoss
        criterion = CompositeFinancialLoss()
        weights = torch.randn(16, config.N_ASSETS, requires_grad=True)
        weights = torch.softmax(weights, dim=-1)
        returns = torch.randn(16, config.N_ASSETS)

        losses = criterion(weights, returns)
        losses["total"].backward()
        assert weights.grad is not None, "Loss is not differentiable"

    def test_turnover_penalty(self):
        """Turnover should be zero when weights don't change."""
        from mlpo.training.loss import CompositeFinancialLoss
        criterion = CompositeFinancialLoss()
        w = torch.softmax(torch.randn(16, config.N_ASSETS), dim=-1)
        returns = torch.randn(16, config.N_ASSETS)

        losses = criterion(w, returns, w_prev=w)
        assert losses["turnover"].item() < 1e-6, (
            f"Turnover should be 0 when weights unchanged, got {losses['turnover']}"
        )
