"""Búsqueda de hiperparámetros con Optuna, optimizando ganancia en pesos."""

import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.constants import N_SPLITS, N_TRIALS, RANDOM_STATE
from fraud_detection.evaluation import optimal_probability_threshold
from fraud_detection.paths import BEST_PARAMS_JSON, TRIALS_CSV, desde_raiz
from fraud_detection.tracking import mlflow, registrar
from fraud_detection.training import XGBOOST_BASE, XGBOOST_DEFAULT, Bloques, iter_fold_results

# Optuna anuncia cada prueba con todos sus parámetros; el progreso lo informa
# `_informar_prueba`, en una línea y con el formato del resto del pipeline.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


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
        logger.debug(
            "bloque %d: auc %.4f | ap %.4f | ganancia %+.0f (acumulada %+.0f)",
            numero,
            resultado["auc"],
            resultado["ap"],
            resultado["ganancia"],
            total,
        )
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


def _informar_prueba(n_trials: int):
    """Una línea de log por prueba terminada, completa o podada."""

    def informar(estudio: optuna.Study, prueba: optuna.trial.FrozenTrial) -> None:
        if prueba.value is None:
            logger.info(
                "prueba %d/%d: podada en el bloque %s",
                prueba.number + 1,
                n_trials,
                prueba.last_step,
            )
        else:
            logger.info(
                "prueba %d/%d: ganancia %+.0f (mejor %+.0f)",
                prueba.number + 1,
                n_trials,
                prueba.value,
                estudio.best_value,
            )

    return informar


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
    avisos = [_anotar_prueba] if registrar_en_mlflow else []
    logger.info("búsqueda: %d pruebas sobre %d bloques", n_trials, len(bloques))
    estudio.optimize(
        lambda trial: gain_over_folds(development, bloques, suggest_params(trial), trial),
        n_trials=n_trials,
        callbacks=[*avisos, _informar_prueba(n_trials)],
    )
    return estudio


# Sin test: la búsqueda ya está cubierta y el resto es registrar en MLflow, como en `tracking`.
def search_and_record(  # pragma: no cover
    development: pd.DataFrame,
    bloques: Bloques,
    n_trials: int = N_TRIALS,
    destino: Path = TRIALS_CSV,
) -> tuple[optuna.Study, float]:
    """Buscar, medir la configuración por defecto y registrar todo en una corrida de MLflow.

    Es la etapa completa que comparten la CLI y `04_tuning.ipynb`. Quien la
    llama elige antes el experimento: `configurar(EXPERIMENTO_BUSQUEDA)`.

    Returns
    -------
    estudio : optuna.Study
        La búsqueda, con todas sus pruebas.
    ganancia_defecto : float
        Ganancia de `XGBOOST_DEFAULT` sobre los mismos bloques: la vara del ajuste.
    """
    with mlflow.start_run(run_name="busqueda con TPE"):
        estudio = run_search(development, bloques, n_trials=n_trials)
        logger.info("midiendo la configuración por defecto sobre los mismos bloques")
        ganancia_defecto = gain_over_folds(development, bloques, XGBOOST_DEFAULT)
        destino.parent.mkdir(parents=True, exist_ok=True)
        estudio.trials_dataframe().to_csv(destino, index=False)
        mlflow.log_params(estudio.best_params)
        mlflow.log_metrics(
            {
                "ganancia_ajustado": estudio.best_value,
                "ganancia_defecto": ganancia_defecto,
                "pruebas_podadas": pruned_trials(estudio),
            }
        )
        mlflow.log_artifact(str(destino))
    logger.info("ganancia con parámetros por defecto: %+.0f", ganancia_defecto)
    logger.info("pruebas guardadas en %s", desde_raiz(destino))
    return estudio, ganancia_defecto


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
    logger.info("configuración guardada en %s", desde_raiz(destino))
    return destino


def load_best(origen: Path = BEST_PARAMS_JSON) -> dict:
    """Leer la configuración elegida por la búsqueda."""
    return json.loads(origen.read_text())
