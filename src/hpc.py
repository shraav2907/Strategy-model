from mpi4py import MPI
import joblib
import numpy as np
from strategy import evaluate_strategy_monte_carlo, find_best_strategy_monte_carlo, generate_two_stop_strategies

def run_mpi_strategy_search(
    model_path,
    base_state,
    total_laps,
    compounds,
    lambda_risk=0.1,
    n_simulations=5,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        model = joblib.load(model_path)
        all_strategies = generate_two_stop_strategies(total_laps, compounds)
        
        print(f"[Rank 0] {len(all_strategies)} strategies generated. Distributing across {size} cores...")
        
        chunks = [[] for _ in range(size)]
        for i, strategy in enumerate(all_strategies):
            chunks[i % size].append(strategy)
    else:
        model = None
        chunks = None
        
    my_strategies = comm.scatter(chunks, root=0)
    model = comm.bcast(model, root=0)
    print(f"[Rank {rank}] Evaluating {len(my_strategies)} strategies ...")
    my_results = []
    for strategy in my_strategies:
        stats = evaluate_strategy_monte_carlo(
            model,
            base_state,
            total_laps,
            strategy,
            n_simulations
        )
        objective = stats["mean_time"] + lambda_risk * stats["std_time"]
        my_results.append((objective, strategy))
         
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]
        best_objective, best_strategy = min(flat_results, key=lambda x: x[0])
        
        print(f"\n[Rank 0] Best strategy found:")
        print(f" Pit laps: {best_strategy['pit_laps']}")
        print(f" Compounds: {best_strategy['compounds']}")
        print(f" Objective value: {best_objective:.2f}")
        
        return best_strategy, best_objective
    
    return None, None
    
if __name__ == "__main__":
    base_state = {
        "rain_intensity": 0.2,
        "track_temperature": 30,
        "air_temp": 25,
        "tire_age": 0,
        "compound_code": 1,
        "pit_loss": 22,
    }
    
    run_mpi_strategy_search(
        model_path="results/multirace_model.pkl",
        base_state=base_state,
        total_laps=58,
        compounds=[0, 1, 2],
        lambda_risk=0.1,
        n_simulations=50
    )