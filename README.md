# SCDAA Local Version

Coursework repository for stochastic control and HJB/LQR experiments.

## Repository structure

- `config.py` — shared problem setup (matrices, horizon, device/dtype, plot dir).
- `lqr_solver.py` — analytical Riccati solver and evaluators for value/control.
- `monte_carlo.py` — Monte Carlo estimators for optimal and constant controls.
- `networks.py` — neural network models used across exercises.
- `exercise_1_1.py` — Riccati solution and analytical value/control diagnostics.
- `exercise_1_2.py` — Monte Carlo convergence studies.
- `exercise_2_1.py` — supervised approximation of value function.
- `exercise_2_2.py` — supervised approximation of optimal control.
- `exercise_3.py` — DGM PDE solve for fixed constant control.
- `exercise_4.py` — policy iteration (evaluation + improvement).
- `run_all.py` — run all exercises or a selected subset.
- `plots/` — generated output figures.

## Installation

```bash
pip install -r requirements.txt
```

If you have a CUDA-capable GPU and a GPU-enabled PyTorch build installed,
the scripts will use it automatically. Otherwise they fall back to CPU.

## Usage

Run a single exercise:

```bash
python exercise_3.py
```

Run all exercises:

```bash
python run_all.py
```

Run selected exercises:

```bash
python run_all.py 1.1 3 4
```

All scripts save figures under `plots/`.
