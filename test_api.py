"""
Run this in a SECOND terminal while app.py is running:
    python test_api.py
"""
import json
import urllib.error
import urllib.request

URL = "http://127.0.0.1:5000/api/predict"


def call(symptoms):
    req = urllib.request.Request(
        URL, data=json.dumps({"symptoms": symptoms}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"HTTP error": e.code, "body": json.loads(e.read())}


tests = {
    "Malaria-like": ["fever", "chills", "cyclic_fever_chills", "sweating",
                     "mosquito_bites"],
    "Heart-attack-like": ["chest_pain", "pain_radiating_arm_jaw_back",
                          "sweating"],
    "Invalid symptom": ["fever", "banana"],
    "Empty list": [],
}
for name, syms in tests.items():
    print(f"\n=== {name} ===")
    print(json.dumps(call(syms), indent=2))
