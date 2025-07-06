import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

model = mlflow.sklearn.load_model("models:/UserPersonaRecommendations/Production")

class InferenceRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: InferenceRequest):
    input_data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"recommendation": int(prediction[0])}
