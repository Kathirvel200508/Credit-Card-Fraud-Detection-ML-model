from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Fraud Scoring API")

model = joblib.load("random_forest_model.pkl")
freq_map = joblib.load("freq_map.pkl")

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int

    class Config:
        extra = "forbid"


@app.post("/score")
def score_transaction(tx: Transaction):
    USD_INR_RATE = 90.0  

    amount_usd = tx.amount / USD_INR_RATE

    hour_sin = np.sin(2 * np.pi * tx.transaction_hour / 24)
    hour_cos = np.cos(2 * np.pi * tx.transaction_hour / 24)

    merchant_category_frequency = freq_map.get(tx.merchant_category, 0)

    X = pd.DataFrame([{
        "amount": amount_usd,
        "foreign_transaction": tx.foreign_transaction,
        "location_mismatch": tx.location_mismatch,
        "device_trust_score": tx.device_trust_score,
        "velocity_last_24h": tx.velocity_last_24h,
        "cardholder_age": tx.cardholder_age,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "merchant_category_frequency": merchant_category_frequency
    }])

    for col in model.feature_names_in_:
        if col.startswith("mcc_"):
            X[col] = 1 if col == f"mcc_{tx.merchant_category}" else 0

    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    risk_score = model.predict_proba(X)[0][1]

    LOW = 0.3
    HIGH = 0.6

    if risk_score < LOW:
        decision = "APPROVE"
    elif risk_score > HIGH:
        decision = "BLOCK"
    else:
        decision = "AI_REVIEW"

    return {
        "risk_score": round(float(risk_score), 4),
        "decision": decision
    }
