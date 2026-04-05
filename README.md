# SCDAA_local_version

Refactored SCDAA coursework code with clear separation per exercise and shared reusable core components.

## Repository structure

- `scdaa_core/`
  - `lqr.py`: shared analytical LQR solver and sample generators.
  - `networks.py`: reusable `NetDGM` model.
  - `plotting.py`: helper for exercise-specific plot output folders.
- `exercises/`
  - `ex1_1.py`: Riccati and value-function visualisation.
  - `ex1_2.py`: Monte Carlo and timestep convergence checks.
  - `ex2_1.py`: value-function supervised learning.
  - `ex2_2.py`: optimal-control supervised learning.
- `run_all_exercises.py`: runs all exercises in sequence.
- `optimiser_comparison_ex2_1.py`: optimiser comparison for Exercise 2.1.

## Plot output layout

All generated plots are now saved under:

- `plots/exercise_1_1/`
- `plots/exercise_1_2/`
- `plots/exercise_2_1/`
- `plots/exercise_2_2/`

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single exercise:

```bash
python -m exercises.ex2_1
```

Run all exercises:

```bash
python run_all_exercises.py
```

Run optimiser comparison:

```bash
python optimiser_comparison_ex2_1.py
```

## Cleanup

Legacy notebook files and root-level generated PNG artifacts were removed.
Use the exercise scripts to regenerate outputs into `plots/<exercise_name>/`.

## CUDA verification

To confirm GPU usage end-to-end:

```bash
python check_cuda.py
```

For training scripts (`exercises/ex2_1.py`, `exercises/ex2_2.py`, and `optimiser_comparison_ex2_1.py`), startup logs now print selected device and CUDA properties.
They also run one-time assertions that model parameters, batches, and predictions are all on the selected device.
