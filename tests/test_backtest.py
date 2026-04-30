import numpy as np
from mlpo.backtest.engine import BacktestEngine
from mlpo.config import config

def test_backtest_transaction_costs():
    """
    Test the BacktestEngine transaction cost application and drift mechanics.
    """
    engine = BacktestEngine(transaction_cost_bps=10.0)
    tc_multiplier = 10.0 / 10000.0
    
    T, P = 3, 2
    # Start with equal weight, so w_prev = [0.5, 0.5]
    
    # Step 0: Target [0.5, 0.5] -> No shift from equal weight w_prev.
    # Returns [0.0, 0.0] -> Gross 0 -> Net 0.
    
    weights = np.array([
        [0.5, 0.5],
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    
    returns = np.array([
        [0.0, 0.0], # period 0 returns
        [0.1, -0.1], # period 1 returns
        [0.05, 0.05] # period 2 returns
    ])
    
    results = engine.run(weights, returns)
    
    # Check shape
    assert results["portfolio_returns"].shape == (T,)
    
    # Period 0: w_target=[0.5,0.5], w_prev=[0.5,0.5], shift=0, tc=0. Gross=0, Net=0.
    assert np.isclose(results["turnover"][0], 0.0)
    assert np.isclose(results["costs"][0], 0.0)
    assert np.isclose(results["portfolio_returns"][0], 0.0)
    
    # Period 1: w_target=[1.0, 0.0], drifted w_prev from period 0 is [0.5, 0.5]
    # turnover = |1-0.5| + |0-0.5| = 1.0
    assert np.isclose(results["turnover"][1], 1.0)
    expected_cost_1 = 1.0 * tc_multiplier
    assert np.isclose(results["costs"][1], expected_cost_1)
    
    gross_1 = 1.0 * 0.1 + 0.0 * (-0.1) # 0.1
    expected_net_1 = gross_1 - expected_cost_1
    assert np.isclose(results["portfolio_returns"][1], expected_net_1)
    
    # Period 2: w_target=[1.0, 0.0]. Drifted w_prev from period 1:
    # w_target was [1.0, 0.0], returns were [0.1, -0.1].
    # w_drifted = [1.0*1.1, 0.0*0.9] = [1.1, 0.0]
    # w_prev = [1.1/1.1, 0.0] = [1.0, 0.0]
    # turnover = |1-1| + |0-0| = 0.0
    assert np.isclose(results["turnover"][2], 0.0)
    assert np.isclose(results["costs"][2], 0.0)
    
    gross_2 = 1.0 * 0.05 + 0.0 * 0.05 # 0.05
    assert np.isclose(results["portfolio_returns"][2], gross_2)
    
    # Check cumulative wealth matches expected sequence:
    # cum[0] = 1.0
    # cum[1] = 1.0 * (1 + 0.1 - 0.001) = 1.099
    # cum[2] = 1.099 * (1 + 0.05) = 1.15395
    assert np.isclose(results["cumulative_wealth"][0], 1.0)
    assert np.isclose(results["cumulative_wealth"][1], 1.099)
    assert np.isclose(results["cumulative_wealth"][2], 1.15395)
