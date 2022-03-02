#Next create an app with FastAPI to predict test cases
from fastapi import FastAPI
import pickle
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel

class Notes (BaseModel):
    Variance: float
    Skewness: float
    Kurtosis: float
    Entropy: float
    class Config:
        schema_extra = {
            "example": {
                "Variance": 0.838816,
                "Skewness": 5.42950,
                "Kurtosis": -6.69215,
                "Entropy": -4.60000,
               }
        }

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = pickle.load(open("model_lr.pkl", "rb"))

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/predict')
def get_notes_category(data: Notes):
    received = data.dict()
    variance = received['Variance']
    skewness = received['Skewness']
    kurtosis = received['Kurtosis']
    entropy = received['Entropy']
    pred_name = model.predict([[variance, skewness, kurtosis, entropy]]).tolist()[0]
    return {'prediction': pred_name}
