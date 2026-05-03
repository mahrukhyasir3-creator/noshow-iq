import joblib
import os
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

MODEL_PATH = os.getenv("MODEL_PATH", "noshow_model.joblib")


def train(X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_res, y_res)
    joblib.dump(clf, MODEL_PATH)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    p0, r0, f0, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0]
    )
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1]
    )

    metrics = {
        "training_size": len(X_train),
        "imbalance_technique": "SMOTE + class_weight=balanced",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_0_show": {
            "precision": round(float(p0[0]), 4),
            "recall": round(float(r0[0]), 4),
            "f1": round(float(f0[0]), 4),
        },
        "class_1_noshow": {
            "precision": round(float(p1[0]), 4),
            "recall": round(float(r1[0]), 4),
            "f1": round(float(f1[0]), 4),
        },
        "accuracy": round(report["accuracy"], 4),
    }
    print(classification_report(
        y_test, y_pred, target_names=["Show", "No-Show"]
    ))
    return metrics


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train() first."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    clf = load_model()
    feature_order = [
        "age", "days_in_advance", "appointment_weekday",
        "scholarship", "hypertension", "diabetes",
        "alcoholism", "handicap", "sms_received",
    ]
    row = [[features.get(f, 0) for f in feature_order]]
    prob = clf.predict_proba(row)[0][1]

    if prob >= 0.6:
        risk = "high"
        recommendation = (
            "Send reminder SMS + call patient. "
            "Consider double-booking this slot."
        )
    elif prob >= 0.35:
        risk = "medium"
        recommendation = (
            "Send reminder SMS 24 hours before appointment."
        )
    else:
        risk = "low"
        recommendation = "Standard reminder. Patient likely to attend."

    return {
        "risk_level": risk,
        "no_show_probability": round(float(prob), 4),
        "recommendation": recommendation,
    }