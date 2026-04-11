import time
from strategy import find_best_strategy_monte_carlo

if __name__ == "__main__":
    model_path = "results/multirace_model_20260410_155024.pkl"

    base_state = {
        "rain_intensity": 0.8,
        "track_temp": 18,
        "air_temp": 15,
        "tire_age": 0,
        "compound_code": 3,
        "pit_loss": 22,
    }

    start = time.time()
    best_strategy, obj = find_best_strategy_monte_carlo(
        model_path, base_state, total_laps=58, compounds=[0, 1, 2, 3, 4], n_simulations=5
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Best strategy: {best_strategy}")
    print(f"Objective: {obj:.2f}")