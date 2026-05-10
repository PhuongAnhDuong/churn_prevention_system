from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


# =========================
# Load precomputed recommendation results
# =========================

recommendations = pd.read_csv("ranked_top_k_all_users.csv")


# =========================
# Create FastAPI app
# =========================

app = FastAPI(
    title="Churn-Aware Recommender API",
    description="Input customer ID and return churn-aware Top-K product recommendations",
    version="1.0"
)


# =========================
# Request schema
# =========================

class RecommendationRequest(BaseModel):
    customerid: float
    top_k: int = 10


# =========================
# Retention decision rule
# =========================

def retention_strategy(row):

    # High churn risk customers
    if row["churn_prob"] > 0.7:

        # High purchase potential
        if row["pred_purchases"] > 1:
            return "Targeted discount + product recommendation"

        else:
            return "Retention email campaign"

    # Medium churn customers
    elif row["churn_prob"] > 0.4:
        return "Personalized recommendation"

    # Low churn customers
    else:
        return "Standard recommendation"


# =========================
# Health check endpoint
# =========================

@app.get("/")
def home():
    return {
        "message": "Churn-Aware Recommender API is running"
    }


# =========================
# Inference endpoint
# =========================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    customer_id = request.customerid
    top_k = request.top_k

    # Retrieve ranked recommendations for this user
    user_recs = recommendations[
        recommendations["customerid"] == customer_id
    ].sort_values("rank").head(top_k)

    if user_recs.empty:
        return {
            "customerid": customer_id,
            "message": "Customer not found or no recommendations available"
        }

    churn_prob = float(user_recs["churn_prob"].iloc[0])
    retention_action = retention_strategy(user_recs.iloc[0])

    products = user_recs[
        [
            "rank",
            "stockcode",
            "ranking_score",
            "pred_purchases",
            "p_alive"
        ]
    ].to_dict(orient="records")

    return {
        "customerid": customer_id,
        "churn_probability": churn_prob,
        "retention_action": retention_action,
        "top_k": top_k,
        "recommended_products": products
    }