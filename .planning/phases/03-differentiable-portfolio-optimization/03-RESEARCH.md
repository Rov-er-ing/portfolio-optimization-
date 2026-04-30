# Phase 3 Research: Differentiable Portfolio Optimization

## 1. Differentiable Risk Budgeting (`cvxpylayers`)
The primary goal of the Risk Budgeting module is to output a set of portfolio weights $w$ that satisfy the risk budget $\beta_t$ dynamically calculated by the Regime Detector, based on the covariance matrix $\hat{\Sigma}$ from the Volatility Forecaster.

### Mathematical Formulation
**Objective:**
Minimize the squared error between the actual risk contribution of the portfolio and the target risk budget $\beta_t$, while applying $L1$ and $L2$ regularizations to reduce turnover and control sparsity.
$$ \min_{w} \left\| \frac{w \odot (\hat{\Sigma}w)}{w^T \hat{\Sigma} w} - \beta_t \right\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w - w_{t-1}\|^2 $$

**Constraints:**
- Fully invested portfolio: $\sum w_i = 1$
- Long-only constraint with upper bound (max weight): $0 \le w_i \le 0.15$

**Differentiability via cvxpylayers:**
`cvxpylayers` requires defining the convex optimization problem using parameterized inputs (Covariance matrix, Risk budgets, L1/L2 penalties, and previous weights) using `cp.Parameter`. This establishes the computational graph enabling forward optimization and backward derivative propagation.

### Hardware Adaptations (RTX 3050 - 6GB VRAM constraint)
Since `cvxpylayers` requires calculating Hessians for backpropagation, it is highly VRAM intensive.
- Limit batch sizes tightly (16 maximum, defined in `config.BATCH_SIZE`).
- Forward pass will optionally utilize Automatic Mixed Precision (AMP) in FP16, but gradients must strictly remain in FP32 to avoid vanishing gradients within the linear solver backward logic.

## 2. Sparse Attention Mechanism
Standard attention mechanism calculates dependencies across all asset pairs, leading to an $\mathcal{O}(P^2)$ complexity, where $P$ is the number of assets. Given the memory budget, this is prohibitively expensive.

**Logic & Strategy:**
We will implement sparse linear attention targeting the $k$ most correlated neighbors per asset.
- Complexity drops to $\mathcal{O}(P \times k)$ where $k \in [10, 30]$ (defined as `config.ATTENTION_K = 15`).
- Masks must be employed prior to the Softmax layer by applying large negative values (`-1e9`) to non-neighboring assets to guarantee non-neighboring assets achieve a $0$ attention score.
- The adjacency/neighborhood matrices will be cached so dynamic routing happens strictly within pre-defined masks without calculating distance arrays dynamically inside the model graph.

## 3. Integration Challenges
We must ensure that the output tensors from `lstm_covariance` ($\hat{\Sigma}$) and `regime_detector` ($\beta_t$) properly match the required input shapes and data types for the `cvxpylayers` PyTorch module without losing tracking information (`requires_grad=True`).
