"""
Phase 2 - Train, compare and save the disease prediction model.

Run:  python train_model.py
Needs: disease_dataset.csv (from generate_dataset.py)
Out:   model.joblib, confusion_matrix.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# 1. Load data ---------------------------------------------------------------
df = pd.read_csv("disease_dataset.csv")
X = df.drop(columns="disease")
y = df["disease"]
print(f"Data: {X.shape[0]} rows, {X.shape[1]} symptom features, "
      f"{y.nunique()} diseases\n")

# 2. Train/test split (test set is untouched until the final check) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# 3. Candidate models --------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "Naive Bayes": BernoulliNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300,
                                            random_state=SEED),
}

# 4. Compare with 5-fold cross-validation on the training set ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = {}
print("Cross-validation accuracy (5-fold):")
for name, model in models.items():
    s = cross_val_score(model, X_train, y_train, cv=cv)
    scores[name] = s.mean()
    print(f"  {name:14s} {s.mean():.3f} (+/- {s.std():.3f})")

best_name = max(scores, key=scores.get)
print(f"\nBest model: {best_name}")

# 5. Final evaluation on the held-out test set -------------------------------
best = models[best_name].fit(X_train, y_train)
pred = best.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred))

# 6. Confusion matrix picture (good for your report) -------------------------
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# 7. Save model + feature order (the website needs both) ---------------------
joblib.dump({"model": best,
             "features": list(X.columns),
             "classes": list(best.classes_)}, "model.joblib")
print("Saved model.joblib and confusion_matrix.png")
