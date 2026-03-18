import joblib
from strategy import find_best_strategy_monte_carlo

model = joblib.load("results\multirace_model_20260315_151919.pkl")
base_state = {
    "rain_intensity": 0.1,
    "track_temp": 30,
    "air_temp": 25,
    "tire_age": 0,
    "compound_code": 0,
    "pit_loss": 22,
}

total_laps = 58

compounds = [0, 1, 2]

best_strategy, time = find_best_strategy_monte_carlo(
    model,
    base_state,
    total_laps,
    compounds,
)

print("Best strategy:", best_strategy)
print("Pit Laps:", best_strategy["pit_laps"])
print("Compounds:", best_strategy["compounds"])
print("Objective:", time)