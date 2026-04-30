# Phase 3 Validation: Differentiable Portfolio Optimization

## Acceptance Criteria
1. **Differentiable Risk Budgeting (`cvxpylayers`)**:
   - The optimization correctly outputs portfolio weights summing to 1.
   - Individual portfolio weights respect the bounds $0 \le w_i \le 0.15$.
   - Gradients flow seamlessly back from the output weights to the input covariance matrix ($\hat{\Sigma}$) and target risk budgets ($\beta_t$).
2. **Sparse Attention**:
   - The attention matrix respects the $K$-nearest neighbor masking, meaning elements outside the $K$-neighborhood are exactly zero.
   - Gradients continue to flow accurately through the sparse attention mechanism.

## Test Strategy (Pytest)
Testing involves mathematical correctness and gradient flow assertions.

### 1. Risk Budgeting Tests (`tests/test_optimization.py`)
- **Weight Constraints Check**: Pass dummy $\hat{\Sigma}$ and $\beta_t$ into the cvxpylayers solver and assert $w_i \ge 0$, $w_i \le 0.15$, and $\sum w_i \approx 1$.
- **Gradient Flow Check**: Call `loss = weights.sum()` and `.backward()`. Assert that both `Sigma.grad` and `beta.grad` are not `None` and do not contain NaNs.

### 2. Sparse Attention Tests (`tests/test_attention.py`)
- **Masking Check**: Forward pass with a dummy pre-computed neighborhood mask and assert that values outside the $k$-neighborhood result in exactly 0 attention weight after softmax.
- **Gradient Flow Check**: Ensure gradients flow back to query/key/value projection matrices inside the attention module.

## Manual-Only Verifications
| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| N/A | N/A | N/A | N/A |
*If none: "All phase behaviors have automated verification."*

All phase behaviors have automated verification.
