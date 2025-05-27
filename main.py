from typing import List
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import secrets
from dotenv import load_dotenv
import os

app = FastAPI()
security = HTTPBasic()

load_dotenv()

USERNAME = os.getenv("BASIC_AUTH_USERNAME")
PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

class TrainItem(BaseModel):
    data: List[float]
    label: str


class ClassifyItem(BaseModel):
    data: List[float]


class Input(BaseModel):
    train: List[TrainItem]
    classify: List[ClassifyItem]


def label_to_int(label: str) -> int:
    # Adjust this logic to match your label format
    return 1 if label.lower() in ["true", "yes", "1", "relevant"] else 0

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.post("/predict")
def train_and_predict(input_data: Input, user: str = Depends(authenticate)):
    # Extract embeddings and labels
    X = [np.array(item.data) for item in input_data.train]
    y = [label_to_int(item.label) for item in input_data.train]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier
    clf = BalancedRandomForestClassifier(
        max_depth=9, 
        max_features=0.8, 
        max_samples=0.3, 
        n_estimators=185
    )
    clf.fit(X_train, y_train)
    
    # Classify new items
    X_classify = [np.array(item.data) for item in input_data.classify]
    relevance_prediction = clf.predict(X_classify)
    relevance_prediction_probas = clf.predict_proba(X_classify)
    
    # Return results as JSON
    return {
        "predictions": relevance_prediction.tolist(),
        "probabilities": relevance_prediction_probas.tolist()
    }
