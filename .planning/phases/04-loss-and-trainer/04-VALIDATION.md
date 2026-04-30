# Phase 4 Validation: Loss & Trainer

## Acceptance Criteria
1. **Composite Loss**:
   - Computes negative Sharpe, Drawdown, and Turnover correctly without returning `NaN` on static or zero-variance inputs.
   - Gradients flow back to the network weights seamlessly.
2. **Trainer Loop**:
   - Properly integrates `torch.cuda.amp.autocast()` and `GradScaler`.
   - Honors the 21-day embargo constraint defined in `config.EMBARGO_DAYS`.
3. **Hyperopt**:
   - `Optuna` correctly initializes a `MedianPruner`.
   - Gracefully terminates suboptimal trials after Epoch 20.

## Test Strategy (Pytest)

### 1. Loss Tests (`tests/test_loss.py`)
- **Differentiability**: Initialize dummy portfolio returns and weights, calculate the composite loss, and call `.backward()`. Assert that gradient matrices are populated.
- **Bounds Testing**: Verify the Turnover penalty is strictly positive or zero.

### 2. Trainer Tests (`tests/test_trainer.py`)
- **Mock Embargo Flow**: Ensure the split between training indices and validation indices leaves exactly a 21-day chronological gap.

## Manual-Only Verifications
| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| N/A | N/A | N/A | N/A |
*If none: "All phase behaviors have automated verification."*

All phase behaviors have automated verification.
