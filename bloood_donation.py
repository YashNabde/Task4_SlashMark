


# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 2️⃣ Load Dataset
df = pd.read_csv("blood.csv")   # adjust path if needed

print("Dataset Loaded Successfully!\n")


# 3️⃣ Data Exploration
print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())


# Define Features and Target

target_column = df.columns[-1]  

X = df.drop(target_column, axis=1)
y = df[target_column]


# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Done!\n")


# 6️⃣ Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("Model Trained Successfully!\n")


# 7️⃣ Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# 8️⃣ Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)


# 9️⃣ Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Donations")
plt.show()


# 🔟 Save Model
joblib.dump(model, "blood_donation_model.pkl")
print("\nModel Saved as blood_donation_model.pkl")


# 1️⃣1️⃣ Predict New Donation (Example)
print("\n--- Predict New Donation ---")

sample_input = X.iloc[[0]]   # first row sample
future_prediction = model.predict(sample_input)

print("Predicted Donation Value:", future_prediction[0])