import requests
import time
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"

def fetch_laps(session_key):

    url = f"{BASE_URL}/laps?session_key={session_key}"
    r = requests.get(url)
    r.raise_for_status()

    return pd.DataFrame(r.json())


def fetch_weather(session_key):

    url = f"{BASE_URL}/weather?session_key={session_key}"
    r = requests.get(url)
    r.raise_for_status()

    return pd.DataFrame(r.json())


def fetch_stints(session_key):

    url = f"{BASE_URL}/stints?session_key={session_key}"
    r = requests.get(url)
    r.raise_for_status()

    return pd.DataFrame(r.json())


def load_race(session_key):

    laps = fetch_laps(session_key)
    time.sleep(1)
    weather = fetch_weather(session_key)
    time.sleep(1)
    stints = fetch_stints(session_key)
    time.sleep(1)

    laps = laps[laps["driver_number"] == 1]
    laps = laps[laps["lap_duration"].notna()]
    laps = laps[laps["lap_duration"] > 40]
    laps = laps[laps["lap_duration"] < 200]
    
    laps = laps.rename(columns={
        "lap_duration": "LapTime",
        "lap_number": "LapNumber"
    })

    weather = weather.rename(columns={
        "rainfall": "rain_intensity",
        "track_temperature": "track_temp",
        "air_temperature": "air_temp"
    })

    laps["Compound"] = None

    for _, stint in stints.iterrows():

        mask = (
            (laps["driver_number"] == stint["driver_number"]) &
            (laps["LapNumber"] >= stint["lap_start"]) &
            (laps["LapNumber"] <= stint["lap_end"])
        )

        laps.loc[mask, "Compound"] = stint["compound"]

    compound_map = {
        "SOFT": 0,
        "MEDIUM": 1,
        "HARD": 2,
        "INTERMEDIATE": 3,
        "WET": 4
    }
    
    laps = laps.dropna(subset=["date_start"])
    weather = weather.dropna(subset=["date"])
    
    laps["compound_code"] = laps["Compound"].map(compound_map)
    laps["date_start"] = pd.to_datetime(laps["date_start"])
    weather["date"] = pd.to_datetime(weather["date"])
    
    data = pd.merge_asof(
        laps.sort_values("date_start"),
        weather.sort_values("date"),
        left_on="date_start",
        right_on="date",
        direction="nearest"
    )

    data["tire_age"] = (
        data.groupby(
            (data["compound_code"] != data["compound_code"].shift()).cumsum()
        ).cumcount()
    )

    data["session_key"] = session_key

    data = data.dropna(subset=[
        "LapTime",
        "rain_intensity",
        "track_temp",
        "air_temp",
        "compound_code"
    ])

    return data

def load_multiple_races(session_keys):

    all_data = []

    for key in session_keys:

        try:
            df = load_race(key)
            all_data.append(df)

        except Exception as e:
            print(f"Failed to load session {key}: {e}")

    if len(all_data) == 0:
        raise ValueError("No race data could be loaded")

    return pd.concat(all_data, ignore_index=True)