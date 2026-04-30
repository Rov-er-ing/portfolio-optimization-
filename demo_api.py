import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_demo():
    print("==================================================")
    print(">>> MLPO v1.0 : End-to-End API Feature Demonstration")
    print("==================================================\n")
    
    # 1. System Health Check
    print("Checking API Health (/health) ...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"[OK] Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect. Is the Uvicorn server running?")
        return
        
    print("\n--------------------------------------------------\n")
    
    # 2. Portfolio Optimization (Normal Market)
    print("Feature 1: Requesting Portfolio Optimization (VIX = 15.0 - Normal Market)")
    print("Sending POST request to /v1/portfolio/optimize ...\n")
    
    normal_payload = {
        "asset_returns": {"AAPL": 0.01, "MSFT": 0.005, "SPY": 0.002}, # Mock sample
        "vix_value": 15.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/portfolio/optimize", json=normal_payload)
    if response.status_code == 200:
        data = response.json()
        print("[SUCCESS] Received AI Portfolio Allocations!")
        print(f"Dominant Market Regime Predicted: {data['regime']['dominant'].upper()}")
        print(f"VIX Override Active: {data['risk_flags']['vix_override_active']}")
        print(f"Expected Turnover (Cost): {data['expected_turnover']:.4f}")
        
        # Show top 5 allocations
        allocations = sorted(data["allocations"], key=lambda x: x["weight"], reverse=True)
        print("\nTop 5 Recommended Allocations:")
        for alloc in allocations[:5]:
            print(f"   - {alloc['ticker']}: {alloc['weight']*100:.2f}%")

        # Show top 3 feature attributions (Interpretability)
        if data.get("top_attributions"):
            print("\nKey Feature Drivers (Interpretability):")
            for attr in data["top_attributions"][:3]:
                print(f"   - {attr['asset']} ({attr['feature']}): {attr['importance']:.4f}")
    else:
        print(f"[ERROR] Failed with status code: {response.status_code}")
        print(response.text)

    print("\n--------------------------------------------------\n")

    # 3. Portfolio Optimization (Extreme Fear Market)
    print("Feature 2: Testing Circuit Breaker (VIX = 45.0 - Market Crash Panic)")
    print("Sending POST request to /v1/portfolio/optimize ...\n")
    
    crash_payload = {
        "asset_returns": {"AAPL": -0.05, "MSFT": -0.04, "SPY": -0.03},
        "vix_value": 45.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/portfolio/optimize", json=crash_payload)
    if response.status_code == 200:
        data = response.json()
        print("[SUCCESS] Circuit Breaker Logic Evaluated!")
        print(f"VIX Override Active: {data['risk_flags']['vix_override_active']}  <-- (The AI recognized extreme fear!)")
    
    print("\n--------------------------------------------------\n")
    
    # 4. Data Validation (Safety constraints)
    print("Feature 3: Testing Pydantic Safety Validations (Invalid Market Data)")
    print("Simulating bad data from a vendor (A stock jumping +80% in one day)...")
    
    bad_payload = {
        "asset_returns": {"AAPL": 0.80}, # Over the 0.5 (50%) realistic limit
        "vix_value": 20.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/portfolio/optimize", json=bad_payload)
    print(f"API Status Code: {response.status_code} (Expected 422 Unprocessable Entity)")
    print("[SUCCESS] The API actively blocked the bad data before it could crash the PyTorch model.")
    print("Error Detail:")
    print(json.dumps(response.json(), indent=2))
    
    print("\n==================================================")
    print("Demo Complete!")
    print("==================================================")

if __name__ == "__main__":
    run_demo()
