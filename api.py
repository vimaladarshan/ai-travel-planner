from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Sample dataset
data = {
    "booking_date": [
        "2024-01-01", "2024-01-10", "2024-01-20", "2024-02-01", "2024-02-15", 
        "2024-03-01", "2024-03-15", "2024-04-01", "2024-05-01", "2024-06-01"
    ],
    "travel_date": [
        "2024-03-01", "2024-03-15", "2024-04-01", "2024-04-15", "2024-05-01",
        "2024-06-01", "2024-06-15", "2024-07-01", "2024-08-01", "2024-09-01"
    ],
    "price": [250, 300, 320, 400, 450, 500, 550, 600, 700, 800]  
}

df = pd.DataFrame(data)
df["booking_date"] = pd.to_datetime(df["booking_date"])
df["travel_date"] = pd.to_datetime(df["travel_date"])
df["days_before_departure"] = (df["travel_date"] - df["booking_date"]).dt.days

# Train ML Model
X = df[["days_before_departure"]]
y = df["price"]
model = LinearRegression()
model.fit(X, y)

@app.get("/")
def read_root():
    return {"message": "AI-driven Travel Price Prediction API is Running 🚀"}

@app.get("/predict/")
def predict_price(days_before_departure: int):
    prediction = model.predict(np.array([[days_before_departure]]))[0]
    return {"days_before_departure": days_before_departure, "predicted_price": round(prediction, 2)}
