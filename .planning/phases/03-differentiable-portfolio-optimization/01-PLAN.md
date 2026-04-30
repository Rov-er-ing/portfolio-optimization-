---
phase: "03-differentiable-portfolio-optimization"
wave: 1
---

# Plan: Differentiable Risk Budgeting Layer

## Goal
Implement the core differentiable convex optimization layer (`models/risk_budgeting.py`) that calculates portfolio weights based on the covariance matrix and target risk budget, using `cvxpylayers`. Ensure unit tests exist to verify gradient flow.

## Dependencies
- Phase 2 outputs (covariance matrix and risk budget formats)
- `cvxpy` and `cvxpylayers` library behavior

## Must Haves
- Uses `cvxpy` to define the portfolio problem (sum to 1, max weight 0.15).
- Uses `CvxpyLayer` from `cvxpylayers.torch` to wrap the optimization problem into a PyTorch module.
- `tests/test_optimization.py` that verifies output bounds and gradient flow.

## Task 1: Create the Differentiable Risk Budgeting Module
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- File `mlpo/models/risk_budgeting.py` exists.
- Contains class `DifferentiableRiskBudgeting` inheriting from `torch.nn.Module`.
- `cvxpy.Problem` defined correctly subject to weight bounds.
</acceptance_criteria>
<action>
Create `mlpo/models/risk_budgeting.py`. Initialize `cvxpy.Variable(n_assets)`. Add parameters for Sigma and Beta. Define the squared error objective matching the risk contribution equation. Wrap the problem using `CvxpyLayer(problem, parameters, variables)`. Implement the `forward` method to pass `(Sigma, Beta)` into the layer and return the weights tensor.
</action>

## Task 2: Implement Constraints and Error Handling
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/models/risk_budgeting.py
</read_first>
<acceptance_criteria>
- Constraints `cp.sum(w) == 1` and `w >= 0`, `w <= 0.15` are present.
- Forward pass catches potential Solver errors.
</acceptance_criteria>
<action>
Update the problem definition in `mlpo/models/risk_budgeting.py` to enforce `0 <= w <= config.MAX_WEIGHT` and `sum(w) == 1`. Ensure solver arguments (`solver_args={"solve_method": "ECOS"}`) are passed to prevent solver instability.
</action>

## Task 3: Unit Tests for Risk Budgeting Layer
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/tests/test_optimization.py
</read_first>
<acceptance_criteria>
- File `tests/test_optimization.py` exists.
- Contains `test_differentiable_risk_budgeting()` executing a forward/backward pass.
</acceptance_criteria>
<action>
Create `tests/test_optimization.py`. Implement a test generating a dummy positive semi-definite matrix and a risk budget vector summing to 1. Pass them to `DifferentiableRiskBudgeting`. Assert that returned weights sum to 1 and are bounded. Compute a dummy loss and call `.backward()`. Assert `Sigma.grad` and `beta.grad` are strictly not `None`.
</action>
