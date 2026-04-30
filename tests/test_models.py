import torch
from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
from mlpo.models.regime_detector import RegimeDetector
from mlpo.config import config, seed_everything

def test_lstm_covariance_shape_and_psd():
    """
    Test that LSTMCovarianceForecaster outputs the correct shape and a PSD matrix.
    Also verifies gradient flow through the Cholesky reconstruction.
    """
    seed_everything(42)
    
    batch_size = 16
    seq_len = config.SEQUENCE_LENGTH
    num_assets = config.N_ASSETS
    num_features = config.N_FEATURES
    
    model = LSTMCovarianceForecaster(
        n_assets=num_assets,
        n_features=num_features,
        hidden_size=256,
        n_layers=3,
        dropout=0.3
    )
    
    # Dummy input: [Batch, Seq, Assets, Features]
    dummy_input = torch.randn(batch_size, seq_len, num_assets, num_features, requires_grad=True)
    
    # Forward pass
    sigma = model(dummy_input)
    
    # 1. Output shape should be [Batch, Assets, Assets]
    assert sigma.shape == (batch_size, num_assets, num_assets), f"Expected shape {(batch_size, num_assets, num_assets)}, got {sigma.shape}"
    
    # 2. Covariance matrix must be symmetric
    # Use torch.allclose to check symmetry within precision limits
    assert torch.allclose(sigma, sigma.transpose(-1, -2), atol=1e-5), "Covariance matrix is not symmetric"
    
    # 3. Minimum eigenvalue of output Sigma is strictly positive
    # torch.linalg.eigvalsh computes eigenvalues of a real symmetric matrix
    # Allow for very small negative values due to floating point inaccuracies
    min_eigval = torch.linalg.eigvalsh(sigma).min()
    assert min_eigval > -1e-5, f"Covariance matrix is not positive definite, min eigenvalue: {min_eigval}"
    
    # 4. Gradient flow check
    # Sum the output and backpropagate to ensure gradients flow back to the LSTM weights
    loss = sigma.sum()
    loss.backward()
    
    # Assert gradient is not None on the first layer of the LSTM
    assert model.lstm.weight_ih_l0.grad is not None, "Gradients did not flow back to the LSTM weights"

def test_regime_detector_override():
    """
    Test that RegimeDetector correctly overrides risk budget when VIX >= 35.
    """
    seed_everything(42)
    
    batch_size = 4
    seq_len = 10
    n_macro = config.N_MACRO
    n_assets = config.N_ASSETS
    
    model = RegimeDetector(
        n_macro=n_macro,
        hidden_size=128,
        n_layers=2,
        n_regimes=3,
        n_assets=n_assets
    )
    
    macro_features = torch.randn(batch_size, seq_len, n_macro)
    
    # Create VIX values: first 2 are normal, last 2 are override
    vix_values = torch.tensor([20.0, 30.0, 35.0, 45.0])
    
    beta, regime_probs = model(macro_features, vix_values)
    
    # Check that sum is approximately 1.0 for all batch items
    assert torch.allclose(beta.sum(dim=1), torch.ones(batch_size)), "Beta vectors do not sum to 1"
    
    # For VIX >= 35, defensive allocation must be >= 0.40
    defensive_idx = config.DEFENSIVE_INDICES
    
    # Check normal cases (might or might not be >= 0.40 depending on random init)
    
    # Check override cases
    for i in [2, 3]:
        defensive_alloc = beta[i, defensive_idx].sum().item()
        assert defensive_alloc >= config.DEFENSIVE_ALLOC - 1e-5, f"Defensive alloc {defensive_alloc} < {config.DEFENSIVE_ALLOC} for VIX {vix_values[i]}"
    
    # Assert gradients flow (just through normal paths, since override path uses inplace/cloning which might detach, but let's check basic flow)
    loss = beta.sum()
    loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None, "Gradients did not flow back to the LSTM weights"


