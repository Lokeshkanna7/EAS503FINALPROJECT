from typing import Union

from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd
import numpy as np

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import joblib
app = FastAPI()

with open('random_forest_model_v1.pkl', 'rb') as f:
    reloaded_model = joblib.load(f)


class payload(BaseModel):  # Ensure this matches the reference in the function
    CreditScore : float
    Age: float
    Balance: float
    NumOfProducts: int
    EstimatedSalary: float
    Gender: str
    Age_Balance_Ratio: float
    Credit_Age_Ratio: float
    Balance_Products_Ratio: float


app = FastAPI()
@app.get("/")
def read_root():
    return{"Name"  : "Lokesh Kanna Rajaram", 
           "Model": "Random_forest_Model"}



@app.post("/predict")
def predict(payload: payload):
    df = pd.DataFrame([payload.model_dump().values()],columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0]}