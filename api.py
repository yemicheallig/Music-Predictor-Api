from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("genre_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.get("/")
def home():
    return {"message": "Music Genre API is running"}

@app.post("/predict")
def predict(data: dict):
    # Convert user input to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Predict
    pred = model.predict(df)[0]
    genre = label_encoder.inverse_transform([pred])[0]

    return {"predicted_genre": genre}