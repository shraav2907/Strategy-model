# F1 Race Strategy Optimisation Under Intense Weather Conditions

A machine learning pipeline for predicting Formula One lap times and optimising pit stop strategies under uncertain and dynamic weather scenarios, with HPC acceleration support.

## Project Structure
```
src/
│
├── data_loader.py            # OpenF1 API ingestion, lap-weather temporal merge
├── dataset.py                # Multi-year session discovery and download
├── feature_engineering.py   # Feature construction: wet memory
├── model.py                  # Random Forest training, MAE evaluation
├── train_multirace.py        # Cross-race training orchestration entry point
│
├── simulation.py             # Deterministic lap-by-lap race simulator
├── monte_carlo.py            # Stochastic rainfall trajectory sampling
├── strategy.py               # Strategy generation, Monte Carlo evaluation, optimisation
│
├── run_strategy_simulation.py  # Entry point: find best pit stop strategy
├── run_monte_carlo.py          # Entry point: evaluate a fixed strategy under uncertainty
│
├── hpc.py                    # HPC acceleration utilities (parallel extensions)
└── results/                  # Saved model artefacts (.pkl files)
```

## Pipeline 

```
OpenF1 API  ──>  data_loader.py  ──>  feature_engineering.py  ──>  model.py
                                                                        |
                                                                   model.pkl
                                                                        |
                                              run_strategy_simulation.py
                                                        |
                                 strategy.py  (generate all 2-stop strategies)
                                        |
                          for each strategy:
                            monte_carlo.py  (N rainfall trajectories)
                                    |
                            simulation.py  (lap-by-lap race time)
                                    |
                     argmin  J(s) = μ_s + λ * σ_s
                                    |
                              best_strategy
```
## Setup
### Requirements
- Python 3.10
- Conda or venv
### Create Environment
```bash
conda create -n f1_strategy python=3.10
conda activate f1_strategy
```
 
### Install Dependencies
 
```bash
pip install pandas numpy scikit-learn joblib requests
```

## Running the code
### Step 1 - Train the model
Edit the session keys in `train_multirace.py` to select which races to train on, then run:

```bash
python train_multirace.py
```
The trained model gets saved to `results/multirace_model_<timestamp>.pkl`.

### Step 2 - Find the best strategy

```bash
python run_strategy_simulation.py
```
This loads the saved model, generates all two-stop strategy combinations, evaluates each under Monte Carlo weather simulation, and returns the strategy with the lowest risk-adjsuted objective score.
