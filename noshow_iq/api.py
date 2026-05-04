import os
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from dotenv import load_dotenv
from noshow_iq.preprocess import clean_single_record
from noshow_iq.model import predict

load_dotenv()
import joblib

# Try multiple paths for model
for _path in [
    "/app/noshow_model.joblib",
    "noshow_model.joblib",
    "/home/user/app/noshow_model.joblib",
]:
    if os.path.exists(_path):
        os.environ["MODEL_PATH"] = _path
        print(f"Model found at: {_path}")
        break
else:
    print("WARNING: Model not found!")

# Train model if not exists
MODEL_PATH = "/app/noshow_model.joblib"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "noshow_model.joblib"

os.environ["MODEL_PATH"] = MODEL_PATH

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/noshow")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client.get_database("noshow")
    predictions_col = db["predictions"]
    training_runs_col = db["training_runs"]
except Exception:
    predictions_col = None
    training_runs_col = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.1.0"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    try:
        cleaned = clean_single_record(data)
        result = predict(cleaned)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_input": data,
        "cleaned_features": cleaned,
        "risk_level": result["risk_level"],
        "no_show_probability": result["no_show_probability"],
        "recommendation": result["recommendation"],
    }
    try:
        if predictions_col is not None:
            predictions_col.insert_one(doc)
    except Exception:
        pass

    return jsonify(result), 200


@app.route("/history", methods=["GET"])
def history():
    try:
        docs = list(
            predictions_col.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(20)
        )
    except Exception:
        docs = []
    return jsonify(docs), 200


@app.route("/stats", methods=["GET"])
def stats():
    try:
        pipeline = [
            {
                "$facet": {
                    "counts": [
                        {
                            "$group": {
                                "_id": "$risk_level",
                                "count": {"$sum": 1},
                            }
                        }
                    ],
                    "avg_prob": [
                        {
                            "$group": {
                                "_id": None,
                                "avg": {"$avg": "$no_show_probability"},
                                "total": {"$sum": 1},
                            }
                        }
                    ],
                }
            }
        ]
        result = list(predictions_col.aggregate(pipeline))
        counts = {
            item["_id"]: item["count"]
            for item in result[0]["counts"]
        }
        avg_data = result[0]["avg_prob"]
        avg_prob = avg_data[0]["avg"] if avg_data else 0.0
        total = avg_data[0]["total"] if avg_data else 0
        last_run = training_runs_col.find_one(
            {}, {"_id": 0, "timestamp": 1},
            sort=[("timestamp", -1)]
        )
    except Exception:
        counts = {}
        avg_prob = 0.0
        total = 0
        last_run = None

    return jsonify({
        "total_predictions": total,
        "high_risk_count": counts.get("high", 0),
        "medium_risk_count": counts.get("medium", 0),
        "low_risk_count": counts.get("low", 0),
        "average_probability": round(avg_prob or 0.0, 4),
        "last_trained": last_run["timestamp"] if last_run else None,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)