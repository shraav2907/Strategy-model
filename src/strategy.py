import itertools
import numpy as np
from joblib import Parallel, delayed
from joblib import parallel_backend

from simulation import simulate_strategy
from monte_carlo import sample_weather_trajectory
from feature_engineering import feature_columns


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

                    strategies.append(
                        {
                            "pit_laps": [p1, p2],
                            "compounds": [c1, c2, c3],
                        }
                    )

    return strategies

def evaluate_one_strategy(
    model,
    base_state,
    total_laps,
    strategy,
    lambda_risk,
    n_simulations
):
    stats = evaluate_strategy_monte_carlo(
        model,
        base_state,
        total_laps,
        strategy,
        n_simulations,
    )
    
    objective = stats["mean_time"] + lambda_risk * stats["std_time"]
    return objective


def evaluate_strategy_monte_carlo(
    model,
    base_state,
    total_laps,
    strategy,
    n_simulations=5,
):

    times = []

    for _ in range(n_simulations):

        rain = sample_weather_trajectory(
            total_laps,
            base_state["rain_intensity"],
            sigma=0.05
        )

        time = simulate_strategy(
            model,
            base_state,
            total_laps,
            strategy["pit_laps"],
            strategy["compounds"],
            feature_columns,
            rain_trajectory=rain,
        )

        times.append(time)

    times = np.array(times)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
    }


def find_best_strategy_monte_carlo(
    model,
    base_state,
    total_laps,
    compounds,
    lambda_risk=0.1,
    n_simulations=5,
    n_jobs=-1
):

    strategies = generate_two_stop_strategies(total_laps, compounds)
    
    print(f"Evaluating {len(strategies)} strategies across {n_jobs} cores...")

    with parallel_backend('threading', n_jobs=n_jobs):
        objectives = Parallel(verbose=1)(
            delayed(evaluate_strategy_monte_carlo)(
                model, base_state, total_laps, strategy, lambda_risk, n_simulations
            )
            for strategy in strategies
        )
    
    best_idx = int(np.argmin(objectives))
    
    return strategies[best_idx], objectives[best_idx]