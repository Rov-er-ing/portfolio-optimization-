---
phase: "03-differentiable-portfolio-optimization"
wave: 1
---

# Plan: Sparse Linear Attention Mechanism

## Goal
Implement sparse attention mechanism (`models/attention.py`) to keep the complexity of asset inter-dependency evaluation linear $O(pk)$ rather than quadratic $O(p^2)$, adhering to the 6GB VRAM constraint.

## Dependencies
- `mlpo/config.py` for `ATTENTION_K`.

## Must Haves
- Uses a neighborhood masking mechanism to force $0$ attention outside of $K$ neighbors.
- Pytest to verify masking bounds and gradient backpropagation.

## Task 1: Create Sparse Attention Module
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- File `mlpo/models/attention.py` exists.
- Contains class `SparseAssetAttention`.
</acceptance_criteria>
<action>
Create `mlpo/models/attention.py`. Implement `SparseAssetAttention` utilizing standard `nn.Linear` layers for Query, Key, and Value generation. Define `forward(x, neighbor_mask)` which computes `Q @ K.T`, applies the `neighbor_mask` by setting $0$ (or False) values to `-1e9`, and passes it through `F.softmax`.
</action>

## Task 2: Unit Tests for Sparse Attention
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/tests/test_attention.py
</read_first>
<acceptance_criteria>
- File `tests/test_attention.py` exists.
- Test verifies the exact behavior of the sparse mask (elements are zero).
</acceptance_criteria>
<action>
Create `tests/test_attention.py`. Implement `test_sparse_attention_masking()`. Generate a dummy sequence and a neighborhood boolean mask. Pass them through `SparseAssetAttention`. Assert that the attention weights evaluated prior to multiplying with `V` are exactly `0.0` for all locations where `neighbor_mask` is `False`. Assert that `.grad` populates correctly on `Q, K, V` projections after calling `.backward()`.
</action>
