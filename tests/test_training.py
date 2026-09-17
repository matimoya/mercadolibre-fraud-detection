"""Entrenamiento y persistencia del modelo."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from fraud_detection.constants import DATE, RANDOM_STATE, TARGET
from fraud_detection.evaluation import optimal_probability_threshold
from fraud_detection.modeling import temporal_folds
from fraud_detection.training import build_model, evaluate_on_folds, load_model, save_model, train


@pytest.fixture(name="estimador")
def fixture_estimador() -> XGBClassifier:
    """Un XGBoost diminuto: acá importa el cableado, no la calidad del ajuste."""
    return XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1, random_state=RANDOM_STATE)


def test_medicion_por_bloques(development: pd.DataFrame, estimador: XGBClassifier):
    """Una fila por bloque, y el umbral por defecto es el de la regla de negocio."""
    bloques = list(temporal_folds(development[DATE]))

    resultados = evaluate_on_folds(development, bloques, estimador)
    explicito = evaluate_on_folds(
        development, bloques, estimador, umbral=optimal_probability_threshold()
    )

    assert list(build_model(estimador).named_steps) == ["preprocesamiento", "modelo"]
    assert len(resultados) == len(bloques)
    assert resultados.auc.between(0, 1).all()
    pd.testing.assert_frame_equal(resultados, explicito)


def test_persistencia(development: pd.DataFrame, estimador: XGBClassifier, tmp_path: Path):
    """El artefacto decide igual que el modelo en memoria, y viaja con su umbral."""
    modelo = train(development, estimador)
    predictores = development.drop(columns=[TARGET])

    destino = save_model(modelo, development, destino=tmp_path / "modelo.joblib")
    recuperado = load_model(destino)

    np.testing.assert_allclose(
        recuperado["pipeline"].predict_proba(predictores)[:, 1],
        modelo.predict_proba(predictores)[:, 1],
    )
    assert recuperado["umbral"] == pytest.approx(optimal_probability_threshold())
    assert recuperado["entrenado_hasta"] == development[DATE].max()
