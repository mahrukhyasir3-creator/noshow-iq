from noshow_iq.preprocess import clean_single_record


def test_clean_record_basic():
    record = {"age": 30, "days_in_advance": 5, "sms_received": 1}
    cleaned = clean_single_record(record)
    assert cleaned["age"] == 30


def test_clean_record_invalid_age_low():
    record = {"age": -5}
    cleaned = clean_single_record(record)
    assert cleaned["age"] == 0


def test_clean_record_invalid_age_high():
    record = {"age": 999}
    cleaned = clean_single_record(record)
    assert cleaned["age"] == 120


def test_clean_record_defaults():
    cleaned = clean_single_record({})
    assert "hypertension" in cleaned
    assert cleaned["hypertension"] == 0


def test_clean_record_boolean():
    record = {"scholarship": True, "diabetes": False}
    cleaned = clean_single_record(record)
    assert cleaned["scholarship"] == 1
    assert cleaned["diabetes"] == 0


def test_clean_record_sms():
    record = {"sms_received": 1}
    cleaned = clean_single_record(record)
    assert cleaned["sms_received"] == 1