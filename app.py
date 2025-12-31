from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
app = FastAPI(title="Fraud Scoring API")
model = joblib.load("random_forest_model.pkl")

class Transaction(BaseModel):
    amount: float
    device_trust_score: float
    velocity_last_24h: int
    foreign_transaction: int
    location_mismatch: int
    transaction_hour: int

@app.post("/score")
def score_transaction(tx: Transaction):
    X = np.array([[
        tx.amount,
        tx.device_trust_score,
        tx.velocity_last_24h,
        tx.foreign_transaction,
        tx.location_mismatch,
        tx.transaction_hour
    ]])
    risk_score = model.predict_proba(X)[0][1]

    if risk_score < 0.3:
        decision = "Approve"
    elif risk_score > 0.7:
        decision = "Decline"
    else:
        decision = "AI_REVIEW"
    return {
        "risk_score": round(float(risk_score), 4),
        "decision": decision
    }

