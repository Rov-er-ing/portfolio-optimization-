import torch
from mlpo.models.risk_budgeting import DifferentiableRiskBudgeting
from mlpo.config import config, seed_everything

def test_differentiable_risk_budgeting():
    """
    Test the differentiable risk budgeting solver (unrolled gradient descent).
    Verifies constraints and gradient flow back to the input parameters.
    """
    seed_everything(42)
    
    batch_size = 4
    n_assets = config.N_ASSETS
    max_weight = config.MAX_WEIGHT
    
    # Instantiate the solver
    solver = DifferentiableRiskBudgeting(
        n_assets=n_assets,
        max_weight=max_weight,
        n_iter=20,
        step_size=0.05
    )
    
    # Dummy inputs
    sigma = torch.randn(batch_size, n_assets, n_assets)
    # Make it symmetric positive-definite
    sigma = torch.bmm(sigma, sigma.transpose(-1, -2)) + 0.1 * torch.eye(n_assets).unsqueeze(0)
    sigma.requires_grad = True
    
    # Target risk budget summing to 1
    beta = torch.rand(batch_size, n_assets)
    beta = beta / beta.sum(dim=-1, keepdim=True)
    beta.requires_grad = True
    
    # Run the forward pass
    weights = solver(sigma, beta)
    
    # 1. Shape check
    assert weights.shape == (batch_size, n_assets), f"Expected shape {(batch_size, n_assets)}, got {weights.shape}"
    
    # 2. Constraint check: Sum to 1
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-4), "Weights do not sum to 1"
    
    # 3. Constraint check: Max weight
    assert torch.all(weights <= max_weight + 1e-4), "Weights exceed max_weight"
    
    # 4. Constraint check: Min weight (Long only)
    assert torch.all(weights >= -1e-4), "Weights are negative"
    
    # 5. Gradient flow check
    # Create a dummy loss
    loss = weights.sum()
    loss.backward()
    
    # Verify gradients flow back to Sigma and Beta
    assert sigma.grad is not None, "Gradients did not flow back to Sigma"
    assert beta.grad is not None, "Gradients did not flow back to Beta"
