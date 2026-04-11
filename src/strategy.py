import time
import itertools
import numpy as np
import joblib
from concurrent.futures import ProcessPoolExecutor

from simulation import simulate_strategy
from monte_carlo import sample_weather_trajectory

def generate_two_stop_strategies(total_laps, compounds, min_gap=5):
    strategies = []
    lap_range = range(10, total_laps - 10, 5)

    for p1, p2 in itertools.combinations(lap_range, 2):
        if p2 - p1 < min_gap:
            continue
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    if len({c1, c2, c3}) < 2:
                        continue
                    strategies.append({
                        "pit_laps": [p1, p2],
                        "compounds": [c1, c2, c3],
                    })

    return strategies


def evaluate_strategy_monte_carlo(model, base_state, total_laps, strategy, n_simulations=50):
    times = []
    for _ in range(n_simulations):
        rain = sample_weather_trajectory(
            total_laps,
            base_state["rain_intensity"],
            sigma=0.05
        )
        t = simulate_strategy(
            model,
            base_state,
            total_laps,
            strategy["pit_laps"],
            strategy["compounds"],
            rain_trajectory=rain,
            silent=True
        )
        times.append(t)

    times = np.array(times)
    return {"mean_time": np.mean(times), "std_time": np.std(times)}


def evaluate_one_strategy(model, base_state, total_laps, strategy, lambda_risk, n_simulations):
    stats = evaluate_strategy_monte_carlo(model, base_state, total_laps, strategy, n_simulations)
    return stats["mean_time"] + lambda_risk * stats["std_time"]

_worker_model = None

def _init_worker(model_path):
    global _worker_model
    _worker_model = joblib.load(model_path)
    _worker_model.n_jobs = 1
    
def _evaluate_worker(args):
    model_path, base_state, total_laps, strategy, lambda_risk, n_simulations = args
    model = joblib.load(model_path)
    model.n_jobs = 1
    return evaluate_one_strategy(model, base_state, total_laps, strategy, lambda_risk, n_simulations)

def find_best_strategy_monte_carlo(
    model_path,
    base_state,
    total_laps,
    compounds,
    lambda_risk=0.1,
    n_simulations=5,
    n_jobs=11
):
    strategies = generate_two_stop_strategies(total_laps, compounds)
    print(f"Evaluating {len(strategies)} strategies...")
    start = time.time()

    args = [(model_path, base_state, total_laps, s, lambda_risk, n_simulations) for s in strategies]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        objectives = list(executor.map(_evaluate_worker, args, chunksize=10))

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} seconds")

    best_idx = int(np.argmin(objectives))
    return strategies[best_idx], objectives[best_idx]