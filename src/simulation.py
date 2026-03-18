import pandas as pd
from datetime import datetime

PIT_LOSS = 22

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
def simulate_strategy(
    model,
    initial_state,
    total_laps,
    pit_laps,
    compounds,
    feature_columns,
    rain_trajectory=None,
    log_every_n_laps=10,
):

    total_time = 0
    tire_age = 0
    wet_track_memory = initial_state["rain_intensity"]

    current_compound = compounds[0]
    pit_pointer = 0

    for lap in range(1, total_laps+1):
        if pit_pointer < len(pit_laps) and lap == pit_laps[pit_pointer]:
            total_time += initial_state["pit_loss"]
            tire_age = 0
            current_compound = compounds[pit_pointer + 1]
            pit_pointer += 1
            log(f"Lap {lap}: Pit stop - switching to compound {current_compound}")

        rain = (
            rain_trajectory[lap-1]
            if rain_trajectory is not None
            else initial_state["rain_intensity"]
        )

        wet_track_memory = 0.8 * wet_track_memory + 0.2 * rain
        
        state = {
            "rain_intensity": rain,
            "rain_squared": rain ** 2,
            "wet_track_memory": wet_track_memory,
            "track_temp": initial_state["track_temp"],
            "tire_age": tire_age,
            "air_temp": initial_state["air_temp"],
            "compound_code": current_compound,
            "rain_compound_interaction": rain * current_compound,
        }

        features = pd.DataFrame([state])
        features = features.reindex(columns=feature_columns, fill_value=0)
        features = features[model.feature_names_in_]
        lap_time = model.predict(features)[0]
        total_time += lap_time
        tire_age += 1
        if lap % log_every_n_laps == 0 or lap in pit_laps:
            log(f"Lap {lap}: Lap time = {lap_time:.2f} seconds, Total time = {total_time:.2f} seconds")

    return total_time