"""
Phase 1 - Synthetic dataset generator for the symptom checker (student project).

HOW IT WORKS
- One MASTER list of symptoms (+ a few context questions).
- For each disease, PROFILES gives the probability that a patient with that
  disease reports each symptom. Anything not listed gets a small noise chance.
- Each generated row simulates one patient. A random "severity" factor makes
  some patients mild (fewer symptoms) and some typical.

IMPORTANT (write this in your report):
The probabilities below are approximate, knowledge-based estimates from common
clinical descriptions. Verify them against WHO / CDC / NHS pages and adjust.
Because the data is synthetic, model accuracy will look higher than it would
on real patient data.

Run:  python generate_dataset.py
Out:  disease_dataset.csv, symptoms.json
"""
import json
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
ROWS_PER_DISEASE = 300
NOISE = 0.02  # chance of an unrelated symptom appearing

# ---------------------------------------------------------------- master list
SYMPTOMS = [
    # general
    "fever", "high_fever", "chills", "cyclic_fever_chills", "sweating",
    "cold_clammy_skin", "headache", "severe_frontal_headache",
    "pain_behind_eyes", "fatigue", "weakness", "muscle_aches",
    "bone_joint_pain", "loss_of_appetite",
    # stomach
    "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "severe_abdominal_pain", "thirst_dry_mouth",
    # nervous system / sugar / pressure
    "dizziness", "lightheaded_on_standing", "fainting", "shakiness", "hunger",
    "irritability_anxiety", "confusion", "tingling_around_mouth",
    "slurred_speech", "blurred_vision", "clumsiness", "seizure",
    # heart / circulation / breathing
    "fast_heartbeat", "weak_pulse", "pale_skin", "rapid_shallow_breathing",
    "shortness_of_breath", "chest_pain", "pain_radiating_arm_jaw_back",
    "indigestion_heartburn", "sense_of_doom",
    # respiratory / cold
    "cough", "sore_throat", "runny_nose", "nasal_congestion", "sneezing",
    "loss_of_smell", "altered_taste",
    # skin
    "rash", "itchy_rash", "fluid_filled_blisters", "scabs_crusts",
    "rash_different_stages", "swollen_lymph_nodes",
    # bleeding / jaundice
    "gum_bleeding", "nosebleed", "easy_bruising", "vomiting_blood",
    "blood_in_stool", "blood_in_urine", "jaundice",
    # context questions
    "mosquito_bites", "travel_endemic_area", "contact_sick_person",
    "sudden_onset",
]

# Symptoms that should trigger an urgent "get medical help now" message in the
# app, separate from the model's prediction.
EMERGENCY_SYMPTOMS = [
    "chest_pain", "pain_radiating_arm_jaw_back", "fainting", "seizure",
    "vomiting_blood", "blood_in_stool", "gum_bleeding", "nosebleed",
    "severe_abdominal_pain", "shortness_of_breath", "slurred_speech",
    "confusion",
]

# --------------------------------------------------------------- disease data
PROFILES = {
    "Low Blood Sugar": {
        "sweating": .8, "headache": .6, "dizziness": .7, "fainting": .3,
        "shakiness": .85, "hunger": .85, "fatigue": .6, "weakness": .5,
        "slurred_speech": .2, "blurred_vision": .35, "clumsiness": .3,
        "seizure": .08, "fast_heartbeat": .6, "irritability_anxiety": .6,
        "pale_skin": .4, "confusion": .35, "tingling_around_mouth": .3,
        "nausea": .25, "cold_clammy_skin": .3, "sudden_onset": .6,
    },
    "Low Blood Pressure": {
        "dizziness": .85, "lightheaded_on_standing": .8, "fainting": .45,
        "fatigue": .6, "weakness": .5, "blurred_vision": .55,
        "pale_skin": .45, "cold_clammy_skin": .4, "rapid_shallow_breathing": .2,
        "weak_pulse": .3, "nausea": .4, "confusion": .2,
        "thirst_dry_mouth": .3, "fast_heartbeat": .35, "headache": .15,
        "sudden_onset": .3,
    },
    "Malaria": {
        "fever": .95, "high_fever": .5, "chills": .9, "cyclic_fever_chills": .6,
        "sweating": .8, "headache": .8, "fatigue": .85, "muscle_aches": .7,
        "bone_joint_pain": .5, "nausea": .65, "vomiting": .5, "diarrhea": .3,
        "abdominal_pain": .35, "loss_of_appetite": .65, "cough": .15,
        "jaundice": .15, "pale_skin": .2, "fast_heartbeat": .2,
        "rapid_shallow_breathing": .1, "mosquito_bites": .7,
        "travel_endemic_area": .55,
    },
    "Dengue": {
        "fever": .95, "high_fever": .9, "chills": .35, "headache": .85,
        "severe_frontal_headache": .75, "pain_behind_eyes": .75,
        "muscle_aches": .85, "bone_joint_pain": .8, "nausea": .6,
        "vomiting": .5, "loss_of_appetite": .6, "altered_taste": .35,
        "rash": .5, "swollen_lymph_nodes": .3, "fatigue": .8,
        "severe_abdominal_pain": .15, "gum_bleeding": .12, "nosebleed": .12,
        "easy_bruising": .15, "vomiting_blood": .05, "blood_in_stool": .06,
        "blood_in_urine": .03, "rapid_shallow_breathing": .08,
        "thirst_dry_mouth": .25, "irritability_anxiety": .15,
        "mosquito_bites": .8, "travel_endemic_area": .5, "sudden_onset": .8,
    },
    "Common Cold": {
        "runny_nose": .9, "nasal_congestion": .85, "sneezing": .85,
        "sore_throat": .65, "cough": .6, "headache": .3, "fever": .12,
        "chills": .1, "muscle_aches": .2, "fatigue": .4, "loss_of_smell": .2,
        "altered_taste": .2, "swollen_lymph_nodes": .1,
        "loss_of_appetite": .2, "contact_sick_person": .7,
        "sudden_onset": .2,
    },
    "Chickenpox": {
        "fever": .7, "fatigue": .6, "headache": .4, "loss_of_appetite": .5,
        "muscle_aches": .3, "rash": .95, "itchy_rash": .9,
        "fluid_filled_blisters": .9, "scabs_crusts": .7,
        "rash_different_stages": .75, "swollen_lymph_nodes": .2,
        "abdominal_pain": .15, "sore_throat": .1, "cough": .1, "chills": .15,
        "contact_sick_person": .7,
    },
    "COVID-19": {
        "fever": .65, "chills": .4, "cough": .7, "shortness_of_breath": .3,
        "fatigue": .7, "sore_throat": .55, "runny_nose": .45,
        "nasal_congestion": .5, "sneezing": .2, "loss_of_smell": .3,
        "altered_taste": .3, "headache": .55, "muscle_aches": .5,
        "bone_joint_pain": .25, "nausea": .2, "vomiting": .1, "diarrhea": .2,
        "loss_of_appetite": .4, "dizziness": .2, "rash": .05,
        "contact_sick_person": .65,
    },
    "Heart Attack": {
        "chest_pain": .85, "pain_radiating_arm_jaw_back": .6,
        "shortness_of_breath": .6, "sweating": .65, "cold_clammy_skin": .4,
        "nausea": .45, "vomiting": .2, "dizziness": .4, "fainting": .15,
        "fatigue": .4, "weakness": .3, "indigestion_heartburn": .3,
        "fast_heartbeat": .35, "sense_of_doom": .3, "irritability_anxiety": .3,
        "sudden_onset": .7,
    },
}


def make_row(profile):
    """Simulate one patient. Returns a 0/1 list in SYMPTOMS order."""
    severity = RNG.uniform(0.6, 1.1)  # <1 = milder case
    while True:
        row = []
        for s in SYMPTOMS:
            p = profile.get(s)
            p = NOISE if p is None else min(p * severity, 0.98)
            row.append(int(RNG.random() < p))
        if sum(row) >= 2:  # a patient reports at least 2 things
            return row


def main():
    records = []
    for disease, profile in PROFILES.items():
        unknown = set(profile) - set(SYMPTOMS)
        assert not unknown, f"{disease}: symptom not in master list: {unknown}"
        for _ in range(ROWS_PER_DISEASE):
            records.append(make_row(profile) + [disease])

    df = pd.DataFrame(records, columns=SYMPTOMS + ["disease"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("disease_dataset.csv", index=False)

    with open("symptoms.json", "w") as f:
        json.dump({"symptoms": SYMPTOMS, "emergency": EMERGENCY_SYMPTOMS,
                   "diseases": list(PROFILES)}, f, indent=2)

    print("Rows:", len(df), "| Symptom columns:", len(SYMPTOMS))
    print(df["disease"].value_counts())
    print("Avg symptoms per patient:", round(df[SYMPTOMS].sum(axis=1).mean(), 1))


if __name__ == "__main__":
    main()
