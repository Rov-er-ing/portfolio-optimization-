---
wave: 1
depends_on: []
files_modified:
  - mlpo/models/regime_detector.py
  - tests/test_models.py
autonomous: true
---

# Plan: Real-Time Regime Detection

<objective>
Build the Real-Time Regime Detector (`mlpo/models/regime_detector.py`) using an LSTM encoder and Gaussian Mixture Model (GMM) with soft-gating and a hard VIX override logic. Include testing.
</objective>

<requirements>
- Task 2.2: Build models/regime_detector.py (GMM + VIX Override)
- Task 2.3: Unit tests for tensor shapes and PSD validity (Regime part)
</requirements>

<tasks>

<task>
<description>Implement Regime Detector Module</description>
<read_first>
- mlpo/config.py
- mlpo/models/regime_detector.py
</read_first>
<action>
Create `RegimeDetector` in `mlpo/models/regime_detector.py` inheriting from `torch.nn.Module`.
Architecture: 2-layer LSTM encoder (128 units).
Followed by a GMM component that outputs probabilities for $k$ regimes (e.g., $k=3$: expansion, neutral, contraction). Use a linear layer followed by Softmax for regime probabilities.
Compute dynamic risk budget $\beta_t = \sum_k (\gamma_k \cdot b_k)$ where $b_k$ are learned regime base allocations.
Implement hard override: if the input tensor indicates VIX >= 35, use `torch.where` to force the final output $\beta_t$ to a defensive allocation constraint (e.g., max equity allocation <= 60%, or cash/defensive >= 40%). Must use `torch.where` to preserve gradient graph.
</action>
<acceptance_criteria>
- `mlpo/models/regime_detector.py` contains `RegimeDetector`.
- Implementation uses `torch.where` or similar tensor-level masking for the VIX >= 35 condition.
</acceptance_criteria>
</task>

<task>
<description>Implement Unit Tests for Regime Detector</description>
<read_first>
- mlpo/models/regime_detector.py
- tests/test_models.py
</read_first>
<action>
Append to `tests/test_models.py`.
Add `test_regime_detector_override()` that provides mock inputs where VIX < 35 and VIX >= 35.
Assert that when VIX >= 35, the output precisely matches the hard-coded defensive allocation.
Assert gradient flow through the network.
</action>
<acceptance_criteria>
- `tests/test_models.py` contains `test_regime_detector_override`.
- Test verifies both normal and override behavior based on VIX value.
</acceptance_criteria>
</task>

</tasks>

<verification>
Run `pytest tests/test_models.py -k test_regime_detector` and verify it passes.
</verification>

<must_haves>
- VIX override must trigger deterministically at >= 35.
- Override mechanism must not break PyTorch computational graph.
</must_haves>
