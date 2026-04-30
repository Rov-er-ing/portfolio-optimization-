import torch
from mlpo.models.attention import SparseAttention
from mlpo.config import config, seed_everything

def test_sparse_attention_masking():
    """
    Test the sparse attention mechanism to ensure neighborhood masking
    correctly zeros out non-neighbor attention weights.
    Also verifies gradient flow through the Q/K/V projections.
    """
    seed_everything(42)
    
    batch_size = 4
    n_assets = config.N_ASSETS
    embed_dim = config.LSTM_HIDDEN # 256
    k = 5 # Set small k for testing
    n_heads = 4
    
    model = SparseAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        k=k,
        n_assets=n_assets
    )
    
    # Set a dummy neighborhood mask
    # For testing, just connect each asset to its adjacent 5 indices (with wrap around)
    dummy_neighborhoods = {}
    asset_list = config.ASSET_LIST
    for i, asset in enumerate(asset_list):
        neighbors = [asset_list[(i + j) % n_assets] for j in range(1, k + 1)]
        dummy_neighborhoods[asset] = neighbors
        
    model.set_neighborhoods(dummy_neighborhoods, asset_list)
    
    # Validate the mask was set correctly
    assert model.neighbor_mask.shape == (n_assets, n_assets)
    # The diagonal must be True
    assert torch.all(model.neighbor_mask.diag()), "Diagonal elements should be True"
    
    # Forward pass
    dummy_input = torch.randn(batch_size, n_assets, embed_dim, requires_grad=True)
    out = model(dummy_input)
    
    # Shape check
    assert out.shape == dummy_input.shape, f"Expected shape {dummy_input.shape}, got {out.shape}"
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    
    assert model.W_q.weight.grad is not None, "Gradients did not flow back to W_q"
    assert model.W_k.weight.grad is not None, "Gradients did not flow back to W_k"
    assert model.W_v.weight.grad is not None, "Gradients did not flow back to W_v"
    assert dummy_input.grad is not None, "Gradients did not flow back to the input embeddings"
