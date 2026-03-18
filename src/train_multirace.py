from data_loader import load_multiple_races
from model import train_model
from feature_engineering import feature_columns
from feature_engineering import preprocess_data
from sklearn.metrics import mean_absolute_error
import joblib
import os 
from datetime import datetime

sessions = [
    9157,  # Bahrain 2023
    9158,  # Saudi Arabia 2023
    9159,   # Australia 2023
    9160,  # Azerbaijan 2023
    9161,  # Miami 2023
]

df = load_multiple_races(sessions)
df = preprocess_data(df)

print(df["LapTime"].describe())

test_session = 9159

train_df = df[df["session_key"] != test_session]
test_df = df[df["session_key"] == test_session]

model, train_mae = train_model(train_df, feature_columns)

test_X = test_df[feature_columns]
test_y = test_df["LapTime"]

preds = model.predict(test_X)

test_mae = mean_absolute_error(test_y, preds)

print("Train MAE:", train_mae)
print("Test MAE (unseen race):", test_mae)

os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"results/multirace_model_{timestamp}.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")