# Phase 4 Research: Loss & Trainer

## 1. Composite Financial Loss (`training/loss.py`)
Traditional Machine Learning optimization typically targets Mean Squared Error (MSE), which poorly represents real-world financial consequences like severe drawdowns. The MLPO v1.0 loss function relies on a composite equation.

**Equation:**
$$ \mathcal{L}(\theta) = -\frac{\mathbb{E}[R_p]}{\sqrt{\text{Var}(R_p)}} + \gamma_1 \text{MDD}(R_p) + \gamma_2 \text{Turnover} + \gamma_3 \|\theta\|_2^2 $$

- **Sharpe Ratio ($\frac{\mathbb{E}[R_p]}{\sqrt{\text{Var}(R_p)}}$)**: Encourages superior risk-adjusted returns. Because the objective minimizes loss, we evaluate the *negative* Sharpe Ratio.
- **Maximum Drawdown (MDD)**: Differentiable approximation of the maximum historical portfolio peak-to-trough decline.
- **Turnover**: Penalizes excessive trading: $\gamma_2 \sum |w_t - w_{t-1}|$.
- **L2 Regularization**: Controls extreme weight saturation inside neural networks.

## 2. Trainer Engine (`training/trainer.py`)
Because the portfolio optimizer runs `cvxpylayers`/unrolled algorithms dynamically, calculating the Hessian during the backward pass expands memory requirements significantly.
- **Automatic Mixed Precision (AMP)**: Requires `torch.cuda.amp.GradScaler()` and `autocast()`. We compute the forward pass in FP16 to halve VRAM pressure, and backpropagate gradients using FP32 to stop gradient vanishing.
- **Walk-Forward Loop**: Standard train-test splits induce lookahead bias in time series. The Walk-Forward loop creates an advancing chronological pipeline. A strict `config.EMBARGO_DAYS = 21` (one trading month) gap acts as a firewall between the end of the training data and the start of validation/testing data to avert correlation leakage.

## 3. Optuna Integration (`training/hyperopt.py`)
Since searching hyperparameter combinations (like LSTM hidden layer dimensions, dropout rates, and learning rates) strains the single consumer GPU, the `Optuna.pruners.MedianPruner` must be deeply embedded.
- **Rule**: If a trial falls below the median validation Sharpe Ratio by Epoch 20, Optuna forcefully aborts the trial, saving critical compute resources.
