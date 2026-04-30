---
phase: "04-loss-and-trainer"
wave: 2
---

# Plan: Trainer & Walk-Forward Protocol

## Goal
Implement the main PyTorch training loop (`training/trainer.py`) integrating Automatic Mixed Precision (AMP) and the chronological Walk-Forward splitter with a 21-day embargo constraint.

## Dependencies
- `mlpo/training/loss.py`
- `mlpo/data/dataset.py`

## Must Haves
- Uses `torch.cuda.amp.autocast()` and `GradScaler` for memory optimization.
- Enforces strict chronological index separation (`config.EMBARGO_DAYS`) to prevent lookahead bias.

## Task 1: Build Trainer Engine
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `mlpo/training/trainer.py` contains `PortfolioTrainer`.
- Contains `fit()` method implementing epochs, batch iteration, zero_grad, AMP forward/backward, and scaler step.
- Implements `evaluate()` method.
</acceptance_criteria>
<action>
Create `mlpo/training/trainer.py`. Implement the `PortfolioTrainer` class encapsulating model parameters, dataloaders, and the optimizer. Instantiate `torch.cuda.amp.GradScaler()`. In the training loop, construct the AMP context, feed inputs, evaluate loss, calculate gradients scaled appropriately, and update weights.
</action>

## Task 2: Build Walk-Forward Splitter
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/data/dataset.py
</read_first>
<acceptance_criteria>
- `mlpo/training/splitter.py` exists with `WalkForwardSplitter`.
- Implements a 21-day margin (embargo) between train and validation chunks.
</acceptance_criteria>
<action>
Create `mlpo/training/splitter.py`. Write logic that accepts total sequence lengths, creates training folds moving sequentially forward in time, and injects exactly `config.EMBARGO_DAYS` as unassigned indices between train and val limits.
</action>

## Task 3: Test Trainer & Splitter
<acceptance_criteria>
- `tests/test_trainer.py` verifies the splitter limits and AMP compilation.
</acceptance_criteria>
<action>
Create `tests/test_trainer.py`. Write unit tests verifying that `train_end + config.EMBARGO_DAYS <= val_start` strictly holds true across multiple folds. Write a fast mock run of the `PortfolioTrainer` using dummy tensors to ensure no AMP `RuntimeError` instances.
</action>
