from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model_trained.pkl')  #model load

class Patient(BaseModel):
    Age:int
    Number:int
    Start:int

@app.post("/predict")
def predict(data:Patient):
    features=np.array([[data.Age,data.Number,data.Start]])
    pred=model.predict(features)[0]
    proba=model.predict_proba(features)[0][1]
    return{
        "prediction":"present" if pred==1 else "absent", 
        "risk":f'{proba*100:.2f}%'
    } 


