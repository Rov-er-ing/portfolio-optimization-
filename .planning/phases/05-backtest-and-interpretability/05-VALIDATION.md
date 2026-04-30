# Phase 5 Validation: Backtest & Interpretability

## Acceptance Criteria
1. **Backtest Engine**:
   - Accurately deducts exactly 10 bps for every percentage of weight turnover.
   - Computes realistic cumulative wealth trajectories, verifying that static (buy-and-hold) comparison avoids transaction penalties.
2. **Interpretability**:
   - Computes gradient attributions (using `captum` or backprop-derived heuristics) that show non-zero scores mapping back to the LSTM asset features and Macro inputs.
   - Logs activation of the `VIX >= 35.0` circuit breaker explicitly.

## Test Strategy (Pytest)

### 1. Backtest Engine Tests (`tests/test_backtest.py`)
- **Transaction Cost Logic**: Provide identical weights for $t=0$ and $t=1$. Ensure transaction costs evaluate to exactly `0.0`. Provide completely inverted weights and assert that the 10 bps scale is correctly applied to the turnover matrix.
- **Metric Verification**: Supply a known dummy return sequence and verify the generated cumulative return maps accurately mathematically.

### 2. Attribution Tests (`tests/test_interpret.py`)
- **Gradient Flow Mapping**: Verify that evaluating attribution does not throw dimension mismatch errors when tracking back to the `[Batch, Seq, Features]` input shape.

## Manual-Only Verifications
| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| N/A | N/A | N/A | N/A |
*If none: "All phase behaviors have automated verification."*

All phase behaviors have automated verification.
