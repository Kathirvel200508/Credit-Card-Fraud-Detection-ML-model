from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Scoring API")

model = joblib.load("random_forest_model.pkl")
fraud_rate_map = joblib.load("fraud_rate_map.pkl")
freq_map = joblib.load("freq_map.pkl")
global_mean = joblib.load("global_mean.pkl")


class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int


@app.post("/score")
def score_transaction(tx: Transaction):

    merchant_category_fraud_rate = fraud_rate_map.get(
        tx.merchant_category,
        global_mean
    )

    merchant_category_frequency = freq_map.get(
        tx.merchant_category,
        0
    )

    X = pd.DataFrame([{
        "amount": tx.amount,
        "transaction_hour": tx.transaction_hour,
        "foreign_transaction": tx.foreign_transaction,
        "location_mismatch": tx.location_mismatch,
        "device_trust_score": tx.device_trust_score,
        "velocity_last_24h": tx.velocity_last_24h,
        "cardholder_age": tx.cardholder_age,
        "merchant_category_fraud_rate": merchant_category_fraud_rate,
        "merchant_category_frequency": merchant_category_frequency
    }])

    X = X[model.feature_names_in_]

    risk_score = model.predict_proba(X)[0][1]

    if risk_score >= 0.9:
        decision = "DECLINE"
    elif risk_score >= 0.7:
        decision = "ESCALATE"
    else:
        decision = "APPROVE"

    return {
        "risk_score": round(float(risk_score), 4),
        "decision": decision
    }
