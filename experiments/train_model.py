from src.dataset import load_multiple_races
from src.feature_engineering import preprocess_data, feature_columns
from src.model import train_model

years = [2022, 2023]

df_raw = load_multiple_races(years)
df = preprocess_data(df_raw)

model, mae = train_model(
    df,
    feature_columns,
    "results/race_model.pkl"
)

print("Model MAE:", mae)