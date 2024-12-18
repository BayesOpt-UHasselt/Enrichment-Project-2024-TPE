from __future__ import annotations

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0, 2 * np.pi)
    y = trial.suggest_float("y", 0, 2 * np.pi)
    f = np.cos(2 * x) * np.cos(y) + np.sin(x)
    c = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)
    trial.set_user_attr("c", float(c))
    trial.set_user_attr("feasible", c <= 0.5)
    return f


def constraints_func(trial: optuna.FrozenTrial) -> tuple[float]:
    return (trial.user_attrs["c"] - 0.5, )


sampler = optunahub.load_module("samplers/ctpe").cTPESampler(constraints_func=constraints_func, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
is_feasible = np.array([t.user_attrs["feasible"] for t in study.trials])
values = np.array([t.value for t in study.trials])
print(f"Found optimal: {np.min(values[is_feasible])}")
