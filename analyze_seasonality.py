import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib  
matplotlib.use("TkAgg")

# Sample flight booking data
data = {
    "booking_date": [
        "2024-01-01", "2024-01-10", "2024-01-20", "2024-02-01", "2024-02-15", 
        "2024-03-01", "2024-03-15", "2024-04-01", "2024-05-01", "2024-06-01"
    ],
    "travel_date": [
        "2024-03-01", "2024-03-15", "2024-04-01", "2024-04-15", "2024-05-01",
        "2024-06-01", "2024-06-15", "2024-07-01", "2024-08-01", "2024-09-01"
    ],
    "price": [250, 300, 320, 400, 450, 500, 550, 600, 700, 800]  # More variation
}


df = pd.DataFrame(data)

# Convert to datetime
df["booking_date"] = pd.to_datetime(df["booking_date"])
df["travel_date"] = pd.to_datetime(df["travel_date"])

# Calculate days before departure
df["days_to_departure"] = (df["travel_date"] - df["booking_date"]).dt.days

# Prepare data for AI model
X = df[["days_to_departure"]]
y = df["price"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on new data
future_days = np.array([[30], [20], [10], [5], [1]])  # Days before departure
predicted_prices = model.predict(future_days)

# Show Predictions
for days, price in zip(future_days, predicted_prices):
    print(f"Days before departure: {days[0]}, Predicted Price: ${price:.2f}")

# Visualize results
plt.figure(figsize=(8, 4))
plt.scatter(df["days_to_departure"], df["price"], color="blue", label="Actual Prices")
plt.plot([d[0] for d in future_days], predicted_prices, color="red", linestyle="dashed", label="Predicted Prices")
plt.xlabel("Days Before Departure")
plt.ylabel("Flight Price")
plt.title("Flight Price Prediction Based on Booking Time")
plt.legend()
plt.grid()
plt.show()
