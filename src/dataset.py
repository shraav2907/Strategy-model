import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"

def fetch_sessions(year):
    url = f"{BASE_URL}/sessions?year={year}&session_type=Race"
    response = requests.get(url)
    response.raise_for_status()
    
    return response.json()


def fetch_laps(session_key):
    url = f"{BASE_URL}/laps?session_key={session_key}"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    return df


def convert_openf1_format(df):
    df = df.rename(columns={
        "lap_duration": "LapTime",
        "compound": "Compound",
        "tyre_age": "TyreLife",
        "track_temperature": "TrackTemp",
        "air_temperature": "AirTemp",
        "rainfall": "Rainfall"
    })

    df["LapTime"] = pd.to_timedelta(df["LapTime"], unit="s")

    return df


def load_multiple_races(years):
    dfs = []

    for year in years:

        sessions = fetch_sessions(year)

        for s in sessions:
            session_key = s["session_key"]

            print(f"Downloading race {session_key}")

            laps = fetch_laps(session_key)

            laps = convert_openf1_format(laps)

            dfs.append(laps)

    df = pd.concat(dfs, ignore_index=True)

    return df