import pandas as pd

compound_map = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4
}

wet_suitability = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}

dry_penalty = {0: 0, 1: 1, 2: 2, 3: 8, 4: 15}

feature_columns = [
    "rain_intensity",
    "rain_squared",
    "wet_track_memory",
    "track_temp",
    "tire_age",
    "air_temp",
    "compound_code",
    "rain_compound_interaction",
    "wet_suitability_score",
    "dry_penalty"
]

def preprocess_data(data):
    df = data[
        [
            "session_key",
            "LapTime",
            "driver_number",
            "LapNumber",
            "rain_intensity",
            "tire_age",
            "track_temp",
            "air_temp",
            "compound_code",
        ]
    ].copy()

    df = df.dropna()

    df = df[(df["LapTime"] > 80) & (df["LapTime"] < 180)]
    df = df.sort_values(["driver_number", "LapNumber"])

    df["rain_intensity"] = df["rain_intensity"].astype(float)
    df["track_temp"] = df["track_temp"].astype(float)
    df["air_temp"] = df["air_temp"].astype(float)
    df["tire_age"] = df["tire_age"].astype(int)
    df["LapNumber"] = df["LapNumber"].astype(int)
    df["wet_track_memory"] = (
        df.groupby("driver_number")["rain_intensity"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df = add_wet_features(df)
    df = df.dropna()

    return df


def add_wet_features(df):

    df["rain_squared"] = df["rain_intensity"] ** 2
    df["rain_compound_interaction"] = (
        df["rain_intensity"] * df["compound_code"]
    )
    df["wet_suitability_score"] = df["compound_code"].map(wet_suitability)
    df["dry_penalty"] = df["compound_code"].map(dry_penalty) * (1 - df["rain_intensity"])

    return df