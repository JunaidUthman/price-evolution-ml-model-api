import requests
import time
import subprocess
import os
import signal

def test_api():
    # Start the API in the background
    print("Starting API...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    process = subprocess.Popen(
        ["./venv/bin/python", "main.py"],
        cwd=".",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    try:
        # Wait for API to start
        print("Waiting for API to start (10s)...")
        time.sleep(10)

        # Health check
        print("Testing health check...")
        response = requests.get("http://localhost:8000/api/v1/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Prediction test
        print("Testing prediction...")
        payload = {
            "prix_usd": 150.0,
            "prix_lag_1": 145.0,
            "prix_lag_3": 140.0,
            "prix_lag_7": 135.0,
            "volatilite_7j": 5.0
        }
        response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
        print(f"Prediction status: {response.status_code}")
        print(f"Response: {response.json()}")

    finally:
        # Stop the API
        print("Stopping API...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

if __name__ == "__main__":
    test_api()
