# PLAN: Phase 1 - Environment & Foundation

## Phase Goal
Establish the project structure, centralized configuration, and a memory-efficient data pipeline (Ingestion -> Features -> PyTorch Dataset).

## Wave 1: Structure & Configuration
*Prerequisite: None*

### Task 1.1.1: Initialize Directory Structure
- **Type:** execute
- **Action:** Create the `mlpo/` package and subdirectories: `data/`, `models/`, `training/`, `backtest/`, `interpret/`, `api/`, `tests/`. Add `__init__.py` to each.
- **Read First:** `prd.txt` (Section 6)
- **Acceptance Criteria:**
  - `dir mlpo` shows all 7 subdirectories.
  - `mlpo/__init__.py` exists.

### Task 1.1.2: Implement `config.py`
- **Type:** execute
- **Action:** Create `mlpo/config.py` as a singleton configuration class. Include:
  - Asset lists (Equities, Fixed Income, Commodities).
  - Sequence length (60), batch size (16/32).
  - Epsilon for numerical stability ($10^{-6}$).
  - Device routing (CUDA/CPU).
- **Read First:** `prd.txt` (Section 1.1, 1.3, 6)
- **Acceptance Criteria:**
  - `mlpo/config.py` contains `ASSET_LIST`.
  - `mlpo/config.py` contains `DEVICE` detection logic.

## Wave 2: Data Ingestion
*Prerequisite: Wave 1*

### Task 1.2.1: Implement `data/ingestion.py`
- **Type:** execute
- **Action:** Create `mlpo/data/ingestion.py` to fetch multi-asset historical data using `yfinance`. Implement adjustment logic for splits and dividends.
- **Read First:** `mlpo/config.py`
- **Acceptance Criteria:**
  - `mlpo/data/ingestion.py` contains a function to fetch data for all assets in `config.ASSET_LIST`.
  - Fetch logic handles `auto_adjust=True` or manual adjustment.

## Wave 3: Features & Dataset
*Prerequisite: Wave 2*

### Task 1.3.1: Implement `data/features.py`
- **Type:** execute
- **Action:** Create `mlpo/data/features.py` for rolling feature engineering.
  - Enforce `.shift(1)` universally.
  - Implement rolling Z-scores and volatility estimates.
  - Cache asset neighborhood correlations ($k \in [10, 30]$).
- **Read First:** `prd.txt` (Section 2, 3.4), `mlpo/data/ingestion.py`
- **Acceptance Criteria:**
  - `mlpo/data/features.py` contains no lookahead bias (grep for `.shift(1)`).
  - Feature matrix has expected dimensions for all instruments.

### Task 1.4.1: Implement `data/dataset.py`
- **Type:** execute
- **Action:** Create `mlpo/data/dataset.py` implementing a PyTorch `Dataset`.
  - Use memory-mapped arrays for efficient loading of rolling windows.
  - Implement walk-forward logic with 21-day embargo.
- **Read First:** `prd.txt` (Section 2, 4), `mlpo/data/features.py`
- **Acceptance Criteria:**
  - `mlpo/data/dataset.py` contains `MLPODataset` class.
  - Embargo logic is verifiable via unit test or code review.

## Verification
### Must-Haves
- [ ] Directory structure exactly matches PRD.
- [ ] `config.py` provides central source of truth for all modules.
- [ ] Data pipeline successfully loads 60-day windows into PyTorch tensors without memory overflow.
- [ ] Zero lookahead bias in feature engineering.

### Automated Checks
- `pytest mlpo/tests/test_data_pipeline.py` (To be created during execution)
- `python -c "from mlpo import config; print(config.DEVICE)"`
