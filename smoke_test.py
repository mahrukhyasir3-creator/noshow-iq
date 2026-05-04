import sys
import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:7860"

TEST_RECORD = {
    "age": 45,
    "days_in_advance": 7,
    "appointment_weekday": 2,
    "scholarship": 0,
    "hypertension": 1,
    "diabetes": 0,
    "alcoholism": 0,
    "handicap": 0,
    "sms_received": 1,
}

def check(name, resp, expected_status=200):
    if resp.status_code == expected_status:
        print(f"PASS  {name}")
        return True
    else:
        print(f"FAIL  {name} — HTTP {resp.status_code}")
        return False

results = []
r = requests.get(f"{BASE_URL}/health", timeout=10)
results.append(check("/health", r))

r = requests.post(f"{BASE_URL}/predict", json=TEST_RECORD, timeout=10)
results.append(check("/predict", r))

r = requests.get(f"{BASE_URL}/stats", timeout=10)
results.append(check("/stats", r))

print()
if all(results):
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)