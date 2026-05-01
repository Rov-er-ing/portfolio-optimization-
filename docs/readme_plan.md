# README Implementation Plan: ML Portfolio Optimization (MLPO)

## 1. Executive Header & Hero Section
*   **Why:** Answers "What is this and why should I care in 5 seconds?" High-impact for Portfolio Managers and Quants.
*   **Format:** Visual Block. Badges (Python, PyTorch, License, DOI) + Large font Hero Metrics (Sharpe, Max DD, Latency).
*   **Mistake to Avoid:** Cluttering with too much text. Keep it strictly data-driven and visual.
*   **Estimated Length:** 15 lines.

## 2. The Quantitative Problem Statement
*   **Why:** Establishes the gap between "Predict-then-Optimize" and "End-to-End Differentiable" methods. Answers "What problem does this solve?"
*   **Format:** 4-6 lines of technical prose.
*   **Mistake to Avoid:** Using marketing buzzwords like "revolutionary" or "novel." Stick to technical limitations (estimation error, non-convex constraints).
*   **Estimated Length:** 10 lines.

## 3. High-Level Architecture (C4 Context)
*   **Why:** Answers "How does this fit into my existing stack?" for Infrastructure Engineers.
*   **Format:** Mermaid C4Context diagram + 3-4 descriptive sentences.
*   **Mistake to Avoid:** Over-complicating the external systems. Focus on Data In (Yahoo/FRED) and Weights Out (OMS).
*   **Estimated Length:** 25 lines.

## 4. Deep-Dive: 5 Core ML Components
*   **Why:** Validates the "Research Basis" for Quants and ML Researchers.
*   **Format:** Mermaid C4Component diagram + Summary Table (Component | Tech | Innovation).
*   **Mistake to Avoid:** Glossing over the "Differentiable" aspect. Explicitly explain M3 (cvxpylayers).
*   **Estimated Length:** 40 lines.

## 5. Performance & Stress Testing (COVID-19 Case Study)
*   **Why:** Peer-reviewed proof for PMs. Answers "Does it actually work in a crash?"
*   **Format:** Results Table (Model vs. Benchmarks) + Callout Box for March 2020.
*   **Mistake to Avoid:** Using backtest results without stating "Out-of-Sample." Be honest about the 21-day embargo.
*   **Estimated Length:** 30 lines.

## 6. Engineering Quickstart
*   **Why:** Lowers the barrier to entry. Answers "How do I see it working?"
*   **Format:** 5-line Bash code block.
*   **Mistake to Avoid:** Missing dependencies or data download steps. Must be a "copy-paste-run" experience.
*   **Estimated Length:** 12 lines.

## 7. Data Infrastructure & Sources
*   **Why:** Transparency for Researchers. Answers "What is the model trained on?"
*   **Format:** Markdown Table (Asset Class | Instruments | Source).
*   **Mistake to Avoid:** Forgetting the FRED macro state variables. They are the "secret sauce" for regime detection.
*   **Estimated Length:** 20 lines.

## 8. Model Technical Specification
*   **Why:** For the skeptics. Answers "Is the math sound?"
*   **Format:** LaTeX math blocks for Loss Function + Hyperparameter Table.
*   **Mistake to Avoid:** Hiding the regularisation settings (Dropout/LayerNorm). Quants want to see the "hardness" of the model.
*   **Estimated Length:** 25 lines.

## 9. Interpretability, Audit & Compliance
*   **Why:** Critical for Compliance Officers. Answers "Can we explain a 20% shift in a meeting?"
*   **Format:** Prose + Audit JSON snippet + Feature Importance table.
*   **Mistake to Avoid:** Making it sound like "Magic." Use SHAP terminology strictly.
*   **Estimated Length:** 20 lines.

## 10. Design Decisions & ADRs
*   **Why:** Architectural transparency. Answers "Why choose Cholesky over direct prediction?"
*   **Format:** Numbered list with one-line summaries + link to DECISIONS.md.
*   **Mistake to Avoid:** Summaries that are too long. Each ADR should be one punchy sentence.
*   **Estimated Length:** 15 lines.

## 11. API Reference & Latency SLAs
*   **Why:** For ML Infra engineers. Answers "Can I put this in production?"
*   **Format:** Concise Table (Method | Path | Latency) + cURL examples.
*   **Mistake to Avoid:** Ignoring the 24.5ms SLA. This is a major selling point.
*   **Estimated Length:** 20 lines.

## 12. Development, Testing & Benchmarks
*   **Why:** For contributors. Answers "How do I verify a new change?"
*   **Format:** Pytest commands + Latency benchmarking script reference.
*   **Mistake to Avoid:** Forgetting the gradient flow test. It is the only way to verify the differentiable solver works.
*   **Estimated Length:** 15 lines.

## 13. Academic Citation
*   **Why:** Authority signal.
*   **Format:** BibTeX block.
*   **Estimated Length:** 12 lines.

## 14. Limitations & Roadmap
*   **Why:** Builds trust through honesty. Answers "What can't it do (yet)?"
*   **Format:** Bulleted list (Limitations) + Table (Roadmap).
*   **Mistake to Avoid:** Overselling v1.0. Be clear that it is "Long-only."
*   **Estimated Length:** 15 lines.
