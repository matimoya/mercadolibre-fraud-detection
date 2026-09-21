"""Armado, entrenamiento y persistencia del modelo."""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fraud_detection.constants import DATE, RANDOM_STATE, TARGET
from fraud_detection.evaluation import approve_all, expected_gain, optimal_probability_threshold
from fraud_detection.features import build_preprocessor
from fraud_detection.paths import MODEL_JOBLIB, desde_raiz

logger = logging.getLogger(__name__)

Bloques = Sequence[tuple[np.ndarray, np.ndarray]]

XGBOOST_BASE = {
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
XGBOOST_DEFAULT = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    **XGBOOST_BASE,
}


def build_model(estimador: BaseEstimator, scale: bool = False, use_score: bool = True) -> Pipeline:
    """Preprocesamiento y modelo en un solo objeto, que es lo que se serializa.

    Parameters
    ----------
    scale : bool, default=False
        Estandarizar las numéricas. Solo lo necesita la regresión logística.
    use_score : bool, default=True
        Incluir `score`, que todavía no está confirmado como disponible al decidir.
    """
    return Pipeline(
        [
            ("preprocesamiento", build_preprocessor(scale=scale, use_score=use_score)),
            ("modelo", clone(estimador)),
        ]
    )


def iter_fold_results(
    development: pd.DataFrame,
    bloques: Bloques,
    estimador: BaseEstimator,
    umbral: float | None = None,
    **opciones: Any,
) -> Iterator[dict]:
    """Entrenar y medir bloque por bloque, cediendo cada resultado apenas está.

    Es un generador para que la búsqueda de hiperparámetros pueda podar una
    configuración que arranca mal sin gastar los bloques que faltan.

    Yields
    ------
    dict
        AUC, average precision y ganancia sobre aprobar todo de ese bloque.
    """
    corte = optimal_probability_threshold() if umbral is None else umbral
    predictores, etiqueta = development.drop(columns=[TARGET]), development[TARGET]

    for numero, (entrenamiento, validacion) in enumerate(bloques):
        modelo = build_model(estimador, **opciones)
        modelo.fit(predictores.iloc[entrenamiento], etiqueta.iloc[entrenamiento])

        valid = development.iloc[validacion]
        probabilidad = pd.Series(
            modelo.predict_proba(predictores.iloc[validacion])[:, 1], index=valid.index
        )
        piso = expected_gain(valid, approve_all(valid))

        yield {
            "fold": numero,
            "auc": roc_auc_score(valid[TARGET], probabilidad),
            "ap": average_precision_score(valid[TARGET], probabilidad),
            "ganancia": expected_gain(valid, probabilidad.lt(corte)) - piso,
        }


def evaluate_on_folds(
    development: pd.DataFrame,
    bloques: Bloques,
    estimador: BaseEstimator,
    umbral: float | None = None,
    **opciones: Any,
) -> pd.DataFrame:
    """Una fila por bloque temporal, con la ganancia medida en pesos."""
    resultados = iter_fold_results(development, bloques, estimador, umbral, **opciones)
    return pd.DataFrame(resultados).set_index("fold")


def train(development: pd.DataFrame, estimador: BaseEstimator, **opciones: Any) -> Pipeline:
    """Entrenar con todo el desarrollo, que es lo que se aplica al período reservado."""
    logger.info(
        "entrenando con %d transacciones hasta %s",
        len(development),
        f"{development[DATE].max():%Y-%m-%d}",
    )
    return build_model(estimador, **opciones).fit(
        development.drop(columns=[TARGET]), development[TARGET]
    )


def save_model(
    modelo: Pipeline,
    development: pd.DataFrame,
    umbral: float | None = None,
    destino: Path = MODEL_JOBLIB,
) -> Path:
    """Guardar el pipeline con su umbral y hasta cuándo entrenó.

    Un modelo sin su umbral no decide nada, y sin la fecha no se sabe con qué
    datos se lo entrenó.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": modelo,
            "umbral": optimal_probability_threshold() if umbral is None else umbral,
            "entrenado_hasta": development[DATE].max(),
        },
        destino,
    )
    logger.info("modelo guardado en %s", desde_raiz(destino))
    return destino


def load_model(origen: Path = MODEL_JOBLIB) -> dict:
    """Recuperar el pipeline entrenado con su umbral."""
    return joblib.load(origen)
