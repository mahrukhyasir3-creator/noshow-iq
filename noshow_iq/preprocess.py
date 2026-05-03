import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace("ï»¿", "")
    )

    rename_map = {
        "no-show": "no_show",
        "noshow": "no_show",
        "hipertension": "hypertension",
        "handcap": "handicap",
        "scheduledday": "scheduled_day",
        "appointmentday": "appointment_day",
        "patientid": "patient_id",
        "appointmentid": "appointment_id",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns},
        inplace=True
    )

    df = df[df["age"].between(0, 120)].copy()
    df.dropna(subset=["no_show"], inplace=True)

    df["no_show"] = (
        df["no_show"].str.strip().str.upper() == "YES"
    ).astype(int)

    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)

    df["days_in_advance"] = (
        df["appointment_day"] - df["scheduled_day"]
    ).dt.days.clip(lower=0)

    df["appointment_weekday"] = df["appointment_day"].dt.dayofweek

    return df


def get_features(df: pd.DataFrame):
    feature_cols = [
        "age",
        "days_in_advance",
        "appointment_weekday",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["no_show"]
    return X, y


def clean_single_record(record: dict) -> dict:
    age = int(record.get("age", 30))
    age = max(0, min(120, age))

    days_in_advance = int(record.get("days_in_advance", 0))
    try:
        sched = pd.to_datetime(record.get("scheduled_day"))
        appt = pd.to_datetime(record.get("appointment_day"))
        if sched and appt:
            days_in_advance = max(0, (appt - sched).days)
    except Exception:
        pass

    return {
        "age": age,
        "days_in_advance": days_in_advance,
        "appointment_weekday": int(record.get("appointment_weekday", 0)),
        "scholarship": int(bool(record.get("scholarship", 0))),
        "hypertension": int(bool(record.get("hypertension", 0))),
        "diabetes": int(bool(record.get("diabetes", 0))),
        "alcoholism": int(bool(record.get("alcoholism", 0))),
        "handicap": int(bool(record.get("handicap", 0))),
        "sms_received": int(bool(record.get("sms_received", 0))),
    }