from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# MODEL_NAME = 'GermanCredit_LGBM'
# MODEL_VERSION = '1'
# model_uri = f'models:/{MODEL_NAME}/{MODEL_VERSION}'
# m-ec511d7c0ea846459cabdb0f227b77a6
# model_uri = "/app/mlruns/1/models/m-ec511d7c0ea846459cabdb0f227b77a6/artifacts"
model_uri = "mlruns/1/models/m-ec511d7c0ea846459cabdb0f227b77a6/artifacts"

model = mlflow.pyfunc.load_model(model_uri)

class PredictionRequest(BaseModel):
    features: dict

@app.get('/')
def read_root():
    return {'message': 'ML API is running (With MLflow Model Registry)'}

@app.post('/predict')
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.features])
    prediction = model.predict(df)
    return {'prediction': int(prediction[0])}