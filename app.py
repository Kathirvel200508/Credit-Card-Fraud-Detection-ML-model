from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os 
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv
app = FastAPI(title="Fraud Scoring API")

load_dotenv()
API_KEY = os.getenv("API_KEY")
LOW = 0.3
HIGH = 0.6
USD_INR_RATE = 90.0
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
async def score_transaction(tx: Transaction):
      

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

    if risk_score < LOW:
        return {
            "risk_score": round(float(risk_score), 4),
            "decision": "APPROVE"
        }
    elif risk_score > HIGH:
        return {
            "risk_score": round(float(risk_score), 4),
            "decision": "BLOCK"
        }
    ai_result = await ai_review(tx.dict())
    final_decision = ("BLOCK" if ai_result["verdict"] == "FRAUD" else "APPROVE")
    return {
    "risk_score": round(float(risk_score), 4),
    "decision": final_decision,
    "ai_verdict": ai_result["verdict"],
    "reason": ai_result["reason"]
}

    
async def ai_review(transaction_data):
    prompt = f"""
You are a fraud analyst.
Higher the Device trust score more trustable the user is and lower means low trust.
Consider all kinds of possible outcomes and always choose the most practical one and dont forget that false negatives will make a greater impact on the bank's reputation.
Don't always decide transaction is FRAUD because there can be some legit transactions too that looks like fraud at first.
Respond ONLY in raw JSON.
NO markdown.
NO backticks.

Format:
{{
  "verdict": "FRAUD" or "LEGIT",
  "reason": "YOUR REASON UNDER 50 WORDS"
}}

Transaction:
{transaction_data}
"""

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )

    if response.status_code != 200:
        raise HTTPException(500, detail=response.text)

    ai_text = response.json()["choices"][0]["message"]["content"].strip()

    if ai_text.startswith("```"):
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

    try:
        ai_json = json.loads(ai_text)

        verdict = ai_json.get("verdict") or ai_json.get("status")
        reason = ai_json.get("reason") or ai_json.get("explanation") or "No reason provided"

        return {
            "verdict": verdict,
            "reason": reason
        }
    except Exception:
        raise HTTPException(500, detail="AI response format invalid")   