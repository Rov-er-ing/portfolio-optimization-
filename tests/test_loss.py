import torch
from mlpo.training.loss import CompositeFinancialLoss
from mlpo.config import seed_everything

def test_composite_loss_gradients():
    """
    Test the CompositeFinancialLoss module to ensure it computes scalar losses,
    returns the correct dictionary of metrics, and propagates gradients smoothly.
    """
    seed_everything(42)
    
    batch_size = 16
    n_assets = 45
    
    # Instantiate the loss
    loss_fn = CompositeFinancialLoss()
    
    # Dummy weights (must require grad for gradient testing)
    weights = torch.rand(batch_size, n_assets, requires_grad=True)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights.retain_grad()
    
    # Dummy asset returns
    returns = torch.randn(batch_size, n_assets) * 0.02
    
    # Dummy previous weights
    w_prev = torch.rand(batch_size, n_assets)
    w_prev = w_prev / w_prev.sum(dim=-1, keepdim=True)
    
    # Dummy model parameters for L2 calculation
    dummy_param = torch.randn(10, 10, requires_grad=True)
    
    # Forward pass
    metrics = loss_fn(weights, returns, w_prev, model_params=[dummy_param])
    
    # 1. Output Type Check
    assert isinstance(metrics, dict), "Output must be a dictionary"
    assert "total" in metrics, "Missing total loss"
    assert "sharpe" in metrics, "Missing sharpe metric"
    assert "mdd" in metrics, "Missing mdd metric"
    assert "turnover" in metrics, "Missing turnover metric"
    
    # 2. Output Format Check (Scalars)
    assert metrics["total"].dim() == 0, "Total loss must be a scalar"
    assert not torch.isnan(metrics["total"]), "Total loss is NaN"
    
    # 3. Gradient Flow Check
    total_loss = metrics["total"]
    total_loss.backward()
    
    assert weights.grad is not None, "Gradients did not flow back to weights"
    assert dummy_param.grad is not None, "Gradients did not flow back to L2 parameters"
    
    # 4. Turnover bounds
    assert metrics["turnover"] >= 0.0, "Turnover must be strictly non-negative"
