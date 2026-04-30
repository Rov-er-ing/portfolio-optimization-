---
wave: 1
depends_on: []
files_modified:
  - mlpo/models/lstm_covariance.py
  - tests/test_models.py
autonomous: true
---

# Plan: LSTM Volatility Forecaster

<objective>
Implement the LSTM Volatility Forecaster (`mlpo/models/lstm_covariance.py`) using Cholesky factorization to guarantee Positive Semi-Definite (PSD) covariance matrix predictions. Include testing.
</objective>

<requirements>
- Task 2.1: Implement models/lstm_covariance.py (Cholesky factorization)
- Task 2.3: Unit tests for tensor shapes and PSD validity (LSTM part)
</requirements>

<tasks>

<task>
<description>Implement LSTM Covariance Forecaster Module</description>
<read_first>
- mlpo/config.py
- mlpo/models/lstm_covariance.py
</read_first>
<action>
Create `LSTMCovarianceForecaster` in `mlpo/models/lstm_covariance.py` inheriting from `torch.nn.Module`.
Architecture: 3-layer LSTM, 256 hidden units, Layer Normalization, Dropout (0.3).
Linear layer to project LSTM output to size `num_assets * (num_assets + 1) // 2` (the number of non-zero elements in a lower triangular matrix).
Forward pass: 
1. Pass input through LSTM.
2. Project to Cholesky factor dimensions.
3. Map flat output to a lower triangular matrix `L` using `torch.tril` and zero-padding or similar operations. Ensure diagonal is strictly positive (e.g., using `F.softplus`).
4. Compute final covariance matrix `Sigma = L @ L^T + epsilon * I`, where `epsilon = 1e-6`.
</action>
<acceptance_criteria>
- `mlpo/models/lstm_covariance.py` contains class `LSTMCovarianceForecaster`.
- Forward pass explicitly computes `L @ L.transpose(-1, -2)`.
</acceptance_criteria>
</task>

<task>
<description>Implement Unit Tests for PSD Validity and Tensor Shapes</description>
<read_first>
- mlpo/models/lstm_covariance.py
- tests/test_models.py
</read_first>
<action>
Create `tests/test_models.py` if it doesn't exist.
Add `test_lstm_covariance_shape_and_psd()` that initializes `LSTMCovarianceForecaster` with 45 assets and sequence length 60, runs a dummy input `[batch_size=16, seq_len=60, num_features]`, and asserts:
1. Output shape is `[batch_size, num_assets, num_assets]`.
2. Covariance matrix is symmetric.
3. Minimum eigenvalue of output `Sigma` is strictly positive using `torch.linalg.eigvalsh(Sigma).min() > 0`.
Add gradient flow check: call `.backward()` on a dummy sum loss and assert `model.lstm.weight_ih_l0.grad is not None`.
</action>
<acceptance_criteria>
- `tests/test_models.py` contains `test_lstm_covariance_shape_and_psd`.
- Test verifies output shape and `torch.linalg.eigvalsh`.
- Test verifies gradient `.grad is not None`.
</acceptance_criteria>
</task>

</tasks>

<verification>
Run `pytest tests/test_models.py -k test_lstm_covariance` and verify it passes.
</verification>

<must_haves>
- Covariance output must be guaranteed Positive Semi-Definite.
- Gradients must flow back through the Cholesky reconstruction to the LSTM weights.
</must_haves>
