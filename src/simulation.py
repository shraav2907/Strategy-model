import numpy as np
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
    rain_trajectory=None,
    log_every_n_laps=10,
    silent=False
):

    total_time = 0
    tire_age = 0
    wet_track_memory = initial_state["rain_intensity"]
    wet_suitability = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    dry_penalty = {0: 0, 1: 1, 2: 2, 3: 8, 4: 15}

    current_compound = compounds[0]
    pit_pointer = 0

    for lap in range(1, total_laps+1):
        if pit_pointer < len(pit_laps) and lap == pit_laps[pit_pointer]:
            total_time += initial_state["pit_loss"]
            tire_age = 0
            current_compound = compounds[pit_pointer + 1]
            pit_pointer += 1
            if not silent:
                log(f"Lap {lap}: Pit stop - switching to compound {current_compound}")
        rain = (
            rain_trajectory[lap-1]
            if rain_trajectory is not None
            else initial_state["rain_intensity"]
        )

        wet_track_memory = 0.8 * wet_track_memory + 0.2 * rain
        
        features = np.array([[
                    rain,                           # rain_intensity
                    rain ** 2,                      # rain_squared
                    wet_track_memory,               # wet_track_memory
                    initial_state["track_temp"],    # track_temp
                    tire_age,                       # tire_age
                    initial_state["air_temp"],      # air_temp
                    current_compound,               # compound_code
                    rain * current_compound,        # rain_compound_interaction
                    wet_suitability[current_compound],  # wet_suitability_score
                    dry_penalty[current_compound] * (1 - rain)  # dry_penalty
                ]], dtype=np.float64)

        lap_time = model.predict(features)[0]
        total_time += lap_time
        tire_age += 1
        if not silent and (lap % log_every_n_laps == 0 or lap in pit_laps):
            log(f"Lap {lap}: Lap time = {lap_time:.2f}...")

    return total_time