---
phase: "05-backtest-and-interpretability"
wave: 2
---

# Plan: Interpretability Engine

## Goal
Implement `interpret/shap_attribution.py` using `captum` Integrated Gradients to trace feature attributions mapping neural outputs backwards to their macroscopic inputs, solving the 'Black Box' compliance problem.

## Dependencies
- `mlpo/models/portfolio.py`
- `captum` (Python package requirement)

## Must Haves
- Calculate an attribution score matrix mapping output weights -> LSTM features and Macro inputs.
- Explicitly log instances when the deterministic VIX threshold is triggered.

## Task 1: Build SHAP/Captum Explainer
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/models/portfolio.py
</read_first>
<acceptance_criteria>
- `mlpo/interpret/shap_attribution.py` contains `ModelExplainer`.
- Successfully wraps the end-to-end `PortfolioOptimizer` into an Integrated Gradients extraction pipeline.
</acceptance_criteria>
<action>
Create `mlpo/interpret/shap_attribution.py`. Implement the `ModelExplainer` class. Instantiate `captum.attr.IntegratedGradients` using the full `PortfolioOptimizer` model. Write an `explain_weights()` method that takes a single sequence instance (Asset Features, Macro Features) and extracts attribution tensors scoring the significance of each input on a specific target output asset weight. Also, add logic to log a warning whenever `vix_value >= config.VIX_THRESHOLD`.
</action>

## Task 2: Build Interpretability Tests
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/interpret/shap_attribution.py
</read_first>
<acceptance_criteria>
- `tests/test_interpret.py` passes the attribution tensor output format test without breaking Autograd rules.
</acceptance_criteria>
<action>
Create `tests/test_interpret.py`. Write a test utilizing dummy tensor inputs, instantiating `ModelExplainer`, and checking that the resulting attribution scores exactly match the dimensionality of the inputs `[Batch, Seq, Features]`.
</action>
