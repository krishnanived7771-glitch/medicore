"""
Phase 2 - Prediction helper. The website backend (Phase 3) will import this.

    from predict import predict_top3
    result = predict_top3(["fever", "chills", "sweating", "mosquito_bites"])
"""
import json
import joblib
import pandas as pd

_bundle = joblib.load("model.joblib")
_model = _bundle["model"]
_features = _bundle["features"]

with open("symptoms.json") as f:
    _EMERGENCY = set(json.load(f)["emergency"])


def predict_top3(selected_symptoms):
    """selected_symptoms: list of symptom names, e.g. ["fever", "cough"]."""
    unknown = [s for s in selected_symptoms if s not in _features]
    if unknown:
        raise ValueError(f"Unknown symptoms: {unknown}")

    # Build one row in the exact column order used for training
    row = pd.DataFrame([[1 if f in selected_symptoms else 0
                         for f in _features]], columns=_features)
    probs = _model.predict_proba(row)[0]

    ranked = sorted(zip(_model.classes_, probs),
                    key=lambda x: x[1], reverse=True)[:3]

    def label(rank, p):
        # Naive Bayes is overconfident, so the website shows labels, not %
        if rank == 0:
            return "Most likely" if p >= 0.40 else "Possible"
        return "Possible" if p >= 0.10 else "Less likely"

    return {
        "predictions": [{"disease": d,
                         "label": label(i, float(p)),
                         "score": round(float(p) * 100, 1)}  # for debugging only
                        for i, (d, p) in enumerate(ranked)],
        "emergency_flags": sorted(_EMERGENCY & set(selected_symptoms)),
        "few_symptoms_warning": len(selected_symptoms) < 3,
    }


if __name__ == "__main__":
    tests = {
        "Malaria-like": ["fever", "chills", "cyclic_fever_chills", "sweating",
                         "headache", "mosquito_bites"],
        "Cold-like": ["runny_nose", "sneezing", "sore_throat",
                      "contact_sick_person"],
        "Heart-attack-like": ["chest_pain", "pain_radiating_arm_jaw_back",
                              "sweating", "shortness_of_breath"],
        "Too few symptoms": ["fever"],
    }
    for name, syms in tests.items():
        print(f"\n{name}: {syms}")
        print(json.dumps(predict_top3(syms), indent=2))
