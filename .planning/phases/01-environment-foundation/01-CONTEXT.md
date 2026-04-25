# Phase 1: Environment & Foundation - Context

**Gathered:** 2026-04-25
**Status:** Ready for planning
**Source:** PRD Express Path (prd.txt)

<domain>
## Phase Boundary
This phase establishes the physical and data foundations of the MLPO system. It delivers the directory structure, the centralized configuration singleton, and the entire data pipeline from raw ingestion to PyTorch-ready memory-efficient datasets.

**Deliverables:**
- Directory structure initialized.
- `config.py` containing asset universe (45 instruments), hyperparams, and hardware constraints.
- `data/ingestion.py` for fetching multi-asset historical data.
- `data/features.py` for state variables and lookahead-free rolling features.
- `data/dataset.py` for lazy-loading PyTorch datasets.
</domain>

<decisions>
## Implementation Decisions

### Directory Structure
- Follow the exact structure defined in Section 6 of `prd.txt`.
- Root: `mlpo/` (or current directory if preferred, but `mlpo/` sub-folder is cleaner). The user workspace is `portfolio`, so `mlpo/` will be the package name.

### Configuration (`config.py`)
- Must be a Singleton or SimpleNamespace.
- Include asset list: 30 Equities, 10 Fixed Income, 5 Commodities.
- Macro State Variables: 6 variables.
- GPU Constraints: Max sequence length 60 days, Batch size 16/32.

### Data Ingestion
- Use `yfinance` for free access, fallback to stubs for Bloomberg.
- Mandatory: Corporate action adjustments (splits, dividends).

### Feature Engineering
- Universal `.shift(1)` enforcement to prevent lookahead bias.
- Z-score normalization using strictly trailing windows.

### Dataset Loading
- Utilize memory-mapped arrays or chunked HDF5 for rolling windows.
- Support `torch.cuda.amp` (FP16) via data type selection.

### the agent's Discretion
- Specific choice of HDF5 vs .npy for memory mapping.
- Specific rolling window sizes for volatility features (beyond the 60-day LSTM limit).
- Error handling strategies for missing ticker data.

</decisions>

<canonical_refs>
## Canonical References

### Project Docs
- `prd.txt` — Section 1.1, 1.4, 2, and 6.
- `.planning/ROADMAP.md` — Phase 1 definition.
- `.planning/REQUIREMENTS.md` — REQ-1.1.* series.

</canonical_refs>

<specifics>
## Specific Ideas
- Neighborhood correlations ($k \in [10, 30]$) should be pre-computed and cached in this phase.
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` logic in a central utility.

</specifics>

<deferred>
## Deferred Ideas
- Real-time stream ingestion (Phase 1 focus is historical/batch).
- Kubernetes deployment configs.

</deferred>

---

*Phase: 01-environment-foundation*
*Context gathered: 2026-04-25 via PRD Express Path*
