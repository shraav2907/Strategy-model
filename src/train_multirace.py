import pandas as pd

from data_loader import load_multiple_races
from model import train_model
from feature_engineering import feature_columns
from feature_engineering import preprocess_data
from sklearn.metrics import mean_absolute_error
import joblib
import os 
from datetime import datetime
import numpy as np


sessions = [
    7953,  # Bahrain 2023 — dry
    7779,  # Saudi Arabia 2023 — dry
    7787,  # Australia 2023 — dry
    9094,  # Monaco 2023 — wet/mixed
    9110,  # Canada 2023 — wet/mixed
    9126,  # United Kingdom 2023 — wet/mixed
    9165,  # Singapore 2023 — wet/mixed
    9173,  # Japan 2023 — wet
    9205,  # Brazil 2023 — wet
]

df = load_multiple_races(sessions)
print(df["compound_code"].value_counts())
print(df["rain_intensity"].describe())
df = preprocess_data(df)

def augment_wet_data(df, multiplier=10):
    wet_penalty = {
        0: 15,   # soft — wrong tyre in rain
        1: 10,   # medium — struggles in rain
        2: 8,    # hard — slightly better but still wrong
        3: 0,    # intermediate — baseline in wet
        4: -3,   # full wet — faster in very heavy rain
    }

    wet_laps = df[df["compound_code"] >= 3].copy()
    dry_laps = df[df["compound_code"] <= 2].copy()
    augmented = []

    for _ in range(multiplier):
        noise = wet_laps.copy()
        noise["LapTime"] = wet_laps["LapTime"] * np.random.uniform(0.98, 1.02, len(wet_laps))
        noise["rain_intensity"] = np.clip(
            wet_laps["rain_intensity"] + np.random.normal(0, 0.05, len(wet_laps)), 0.3, 1.0
        )
        augmented.append(noise)

    for compound in [0, 1, 2]:
        compound_laps = dry_laps[dry_laps["compound_code"] == compound].copy()
        if len(compound_laps) == 0:
            continue
        for _ in range(3):
            synthetic = compound_laps.copy()
            base_wet_time = 110 + wet_penalty[compound]
            synthetic["LapTime"] = base_wet_time + np.random.normal(0, 2, len(synthetic))
            synthetic["rain_intensity"] = np.random.uniform(0.5, 1.0, len(synthetic))
            synthetic["wet_track_memory"] = synthetic["rain_intensity"]
            augmented.append(synthetic)

    return pd.concat([df] + augmented, ignore_index=True)

df = augment_wet_data(df, multiplier=10)
print(f"After augmentation: {len(df)} rows")
print(df["compound_code"].value_counts())

wet_laps=df[df["rain_intensity"]>0]
print(f"Laps with rain: {len(wet_laps)} out of {len(df)}")
print(f"Wet Compound laps: {len(df[df['compound_code'] >=3])}")
print(df["LapTime"].describe())

test_session = 7787 #Australia

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