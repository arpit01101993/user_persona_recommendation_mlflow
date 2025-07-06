import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="User Persona Recommendation API",
    description="Serve real-time personalized recommendations.",
    version="1.0.0"
)

# Load the latest Production model from the MLflow Registry
# Or use local path like: mlflow.sklearn.load_model("models:/UserPersonaRecommendations/Production")
model = mlflow.sklearn.load_model("models:/UserPersonaRecommendations/Production")

# Define request schema
class RecommendationRequest(BaseModel):
    user_id: str
    features: list  # Example: [0.4, 1.2, 0.9, ...]

# Define response schema
class RecommendationResponse(BaseModel):
    user_id: str
    recommended_products: list

# POST endpoint for prediction
@app.post("/predict", response_model=RecommendationResponse)
def predict_rec(request: RecommendationRequest):
    # Convert input features to array
    input_data = np.array(request.features).reshape(1, -1)

    # Predict product ID(s) or category
    prediction = model.predict(input_data)

    # Wrap output as JSON
    return RecommendationResponse(
        user_id=request.user_id,
        recommended_products=[str(pred) for pred in prediction]
    )

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
