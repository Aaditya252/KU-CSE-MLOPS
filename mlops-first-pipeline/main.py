from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/iris_random_forest_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Random Forest API")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health_check():
    return {"status": "healthy"}