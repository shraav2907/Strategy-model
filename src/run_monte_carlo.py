import joblib
import numpy as np
from monte_carlo import monte_carlo_optimisation

np.random.seed(30)

model = joblib.load("results/model.pkl")

base_state = {
    "rain_intensity": 0.2,
    "wet_track_memory": 0.2,
    "track_temp": 30,
    "tire_age": 0,
    "air_temp": 20,
    "compound_code": 0,
    "pit_loss": 22
}

mean_time, var_time = monte_carlo_optimisation(
    model,
    base_state,
    total_laps=58,
    mean=0.2,
    std=0.1,
    runs=1000
)

print(f"Expected race time: {mean_time:.2f} seconds")
print(f"Variance: {var_time:.2f}")