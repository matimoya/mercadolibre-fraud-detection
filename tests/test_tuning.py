"""Búsqueda de hiperparámetros."""

import json
from pathlib import Path

import optuna
import pandas as pd
import pytest
from xgboost import XGBClassifier

from fraud_detection.constants import DATE
from fraud_detection.evaluation import optimal_probability_threshold
from fraud_detection.modeling import temporal_folds
from fraud_detection.training import XGBOOST_BASE, build_model
from fraud_detection.tuning import run_search, save_best

PRUEBAS = 2


def test_handoff_entre_busqueda_y_evaluacion(development: pd.DataFrame, tmp_path: Path):
    """La búsqueda deja una configuración que la evaluación puede levantar y usar tal cual.

    Es el único acoplamiento entre las dos etapas: si el JSON no alcanza para
    reconstruir el modelo, la evaluación final no reproduce lo que se eligió.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bloques = list(temporal_folds(development[DATE]))

    estudio = run_search(development, bloques, n_trials=PRUEBAS, registrar_en_mlflow=False)
    guardado = json.loads(save_best(estudio, destino=tmp_path / "parametros.json").read_text())

    assert guardado["pruebas"] == PRUEBAS
    assert guardado["umbral"] == pytest.approx(optimal_probability_threshold())
    assert XGBOOST_BASE.items() <= guardado["parametros"].items()

    modelo = build_model(XGBClassifier(**guardado["parametros"]))
    assert (
        modelo.named_steps["modelo"].get_params()["max_depth"] == estudio.best_params["max_depth"]
    )
