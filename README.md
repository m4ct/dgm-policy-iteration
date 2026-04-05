# SCDAA DGM Policy Iteration Coursework

This repository contains a Python implementation for the 2026 SCDAA coursework on stochastic control. It combines analytical LQR methods, Monte Carlo validation, supervised learning, Deep Galerkin Method (DGM) experiments, and policy iteration for the HJB equation.

## Project structure

- `config.py` - shared problem parameters, device selection, and plot settings.
- `lqr_solver.py` - analytical Riccati-based LQR solver, value function, and optimal control.
- `monte_carlo.py` - Monte Carlo estimators for the optimal-control and constant-control settings.
- `networks.py` - neural network architectures used across the exercises.
- `exercise_1_1.py` - Riccati solve and analytical value-function plots.
- `exercise_1_2.py` - Monte Carlo convergence checks against the analytical solution.
- `exercise_2_1.py` - supervised learning of the value function with a DGM network.
- `exercise_2_2.py` - supervised learning of the optimal control with a feed-forward network.
- `exercise_3.py` - DGM solution of the HJB PDE for a fixed constant control.
- `exercise_4.py` - policy iteration for the HJB equation using learned value and policy networks.
- `check_cuda.py` - quick PyTorch/CUDA environment check.
- `run_all.py` - runs all exercises, or a selected subset, from the command line.
- `plots/` - generated PNG figures written by the exercise scripts.
- `SCDAA-CW-2026.pdf` - coursework brief.
- `requirements.txt` - Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```

The scripts select CUDA automatically when available and otherwise run on CPU.

## Running the code

Run a single exercise:

```bash
python exercise_1_1.py
```

Run all exercises:

```bash
python run_all.py
```

Run a subset of exercises:

```bash
python run_all.py 1.1 1.2 3 4
```

Valid exercise labels are `1.1`, `1.2`, `2.1`, `2.2`, `3`, and `4`.

Check CUDA availability:

```bash
python check_cuda.py
```

## Plot outputs

All plots are saved directly in `plots/`. The current repository includes outputs for:

- `exercise_1_1_riccati.png`
- `exercise_1_1_value_function.png`
- `exercise_1_2_timestep_convergence.png`
- `exercise_1_2_mc_convergence.png`
- `exercise_2_1_loss.png`
- `exercise_2_1_surfaces.png`
- `exercise_2_2_loss.png`
- `exercise_2_2_vector_fields.png`
- `exercise_2_2_component_error.png`
- `exercise_3_loss.png`
- `exercise_3_error_vs_mc.png`
- `exercise_4_policy_iteration_loss.png`
- `exercise_4_value_comparison.png`

## Notes

Exercises 2.1 to 4 involve neural network training and can take substantially longer to run than Exercises 1.1 and 1.2.
