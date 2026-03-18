from dataset import load_multiple_seasons
from src.model import train_model
from src.feature_engineering import FEATURE_COLUMNS
import joblib

years = [2019, 2020, 2021, 2022]

df = load_multiple_seasons(years)

wet_df = df[df["regime"] == "wet"]
dry_df = df[df["regime"] == "dry"]

wet_model, wet_mae = train_model(wet_df, FEATURE_COLUMNS)
dry_model, dry_mae = train_model(dry_df, FEATURE_COLUMNS)

joblib.dump(wet_model, "results/wet_model.pkl")
joblib.dump(dry_model, "results/dry_model.pkl")

print("Wet MAE:", wet_mae)
print("Dry MAE:", dry_mae)