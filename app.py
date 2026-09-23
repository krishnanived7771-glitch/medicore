"""
Phase 3 - Flask backend.

Run:   python app.py
Then open in your browser:  http://127.0.0.1:5000
"""
import json
import os
from flask import Flask, jsonify, request, send_from_directory
from predict import predict_top3

app = Flask(__name__)

with open("symptoms.json") as f:
    META = json.load(f)

CONTEXT = {"mosquito_bites", "travel_endemic_area",
           "contact_sick_person", "sudden_onset"}

CONTEXT_TEXT = {
    "mosquito_bites": "Recently bitten by mosquitoes",
    "travel_endemic_area": "Recently travelled to a malaria/dengue area",
    "contact_sick_person": "Recent contact with a sick person",
    "sudden_onset": "Symptoms started suddenly",
}

DISCLAIMER = ("This tool is for educational purposes only and is not medical "
              "advice. See a doctor for any health concern.")
EMERGENCY_MESSAGE = ("Some of your symptoms can be serious. Please seek "
                     "medical help immediately or call your local "
                     "emergency number.")


def pretty(symptom_id):
    return CONTEXT_TEXT.get(symptom_id, symptom_id.replace("_", " ").capitalize())


@app.get("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")


@app.get("/api/symptoms")
def symptoms():
    items = [{"id": s,
              "label": pretty(s),
              "type": "context" if s in CONTEXT else "symptom",
              "emergency": s in META["emergency"]}
             for s in META["symptoms"]]
    return jsonify(items)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    selected = data.get("symptoms")

    if not isinstance(selected, list) or not selected:
        return jsonify({"error": "Send a non-empty list called 'symptoms'."}), 400
    try:
        result = predict_top3(selected)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result["emergency"] = bool(result["emergency_flags"])
    result["emergency_message"] = EMERGENCY_MESSAGE if result["emergency"] else ""
    result["disclaimer"] = DISCLAIMER
    return jsonify(result)


if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)