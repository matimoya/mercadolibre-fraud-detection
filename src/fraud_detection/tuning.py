"""Búsqueda de hiperparámetros con Optuna, optimizando ganancia en pesos."""

import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.constants import N_SPLITS, N_TRIALS, RANDOM_STATE
from fraud_detection.evaluation import optimal_probability_threshold
from fraud_detection.paths import BEST_PARAMS_JSON
from fraud_detection.tracking import registrar
from fraud_detection.training import XGBOOST_BASE, Bloques, iter_fold_results


def suggest_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda de XGBoost.

    El orden de las sugerencias es parte del estado del sampler: cambiarlo
    cambia la secuencia de pruebas aunque la semilla sea la misma.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        **XGBOOST_BASE,
    }


def gain_over_folds(
    development: pd.DataFrame,
    bloques: Bloques,
    parametros: dict,
    trial: optuna.Trial | None = None,
) -> float:
    """Ganancia sobre aprobar todo, acumulada bloque a bloque.

    Con un `trial`, corta la prueba apenas la acumulada queda por debajo de la
    mediana: una configuración que arranca mal no gasta los bloques restantes.
    """
    total = 0.0
    modelo = XGBClassifier(**parametros)
    for numero, resultado in enumerate(iter_fold_results(development, bloques, modelo)):
        total += resultado["ganancia"]
        if trial is not None:
            trial.report(total, numero)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return total


def _anotar_prueba(_estudio: optuna.Study, prueba: optuna.trial.FrozenTrial) -> None:
    """Guardar cada prueba completada como corrida anidada."""
    if prueba.value is not None:
        registrar(
            f"prueba {prueba.number}",
            parametros=prueba.params,
            metricas={"ganancia": prueba.value},
            anidada=True,
        )


def run_search(
    development: pd.DataFrame,
    bloques: Bloques,
    n_trials: int = N_TRIALS,
    registrar_en_mlflow: bool = True,
) -> optuna.Study:
    """Buscar hiperparámetros maximizando ganancia, no AUC."""
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    estudio.optimize(
        lambda trial: gain_over_folds(development, bloques, suggest_params(trial), trial),
        n_trials=n_trials,
        callbacks=[_anotar_prueba] if registrar_en_mlflow else None,
    )
    return estudio


def pruned_trials(estudio: optuna.Study) -> int:
    """Cuántas pruebas se cortaron temprano."""
    return sum(prueba.state == optuna.trial.TrialState.PRUNED for prueba in estudio.trials)


def best_params(estudio: optuna.Study) -> dict:
    """La configuración elegida, lista para instanciar el modelo."""
    return {**estudio.best_params, **XGBOOST_BASE}


def save_best(estudio: optuna.Study, destino: Path = BEST_PARAMS_JSON, **extra: Any) -> Path:
    """Escribir la configuración que después levanta la evaluación final."""
    configuracion = {
        "parametros": best_params(estudio),
        "ganancia_validacion": estudio.best_value,
        **extra,
        "umbral": optimal_probability_threshold(),
        "pruebas": len(estudio.trials),
        "folds": N_SPLITS,
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False))
    return destino


def load_best(origen: Path = BEST_PARAMS_JSON) -> dict:
    """Leer la configuración elegida por la búsqueda."""
    return json.loads(origen.read_text())
