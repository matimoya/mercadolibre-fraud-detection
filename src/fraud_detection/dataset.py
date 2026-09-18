"""Carga del dataset y reserva del período de evaluación.

Centralizado para que todos los notebooks partan del mismo frame: misma
normalización de texto y mismo índice. No guarda nada en disco.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.constants import (
    AMOUNT,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_DOMAINS,
    DATE,
    NUMERIC_COLUMNS,
    SCORE_RANGE,
    TARGET,
    TEST_START,
)
from fraud_detection.diagnostics import check_labels
from fraud_detection.paths import DATASET_CSV

logger = logging.getLogger(__name__)


def load_transactions(csv: Path | str = DATASET_CSV) -> pd.DataFrame:
    """Leer el CSV del enunciado con la fecha ya parseada, sin filtrar nada."""
    frame = pd.read_csv(csv, parse_dates=[DATE])
    logger.info("%d transacciones leídas de %s", len(frame), Path(csv).name)
    return frame


def normalize_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Sacar espacios de los códigos de texto y tratar el vacío como nulo.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos recién leídos.

    Returns
    -------
    pandas.DataFrame
        Copia con las columnas de texto normalizadas.

    Notes
    -----
    En este dataset no cambia ninguna fila, pero deja la regla aplicada para
    que el resultado no dependa de que la fuente venga siempre limpia.
    """
    clean = frame.copy()
    for col in clean.select_dtypes(include=["object", "string", "str"]).columns:
        clean[col] = clean[col].str.strip().replace("", pd.NA)
    return clean


def _check_domains(frame: pd.DataFrame) -> None:
    """Rechazar códigos y escalas que el modelo aceptaría sin error pero leería mal."""
    for col in BINARY_COLUMNS:
        if not frame[col].dropna().isin([0, 1]).all():
            raise ValueError(f"Revisar dominio de {col}.")
    # Como float, 'a' pasaría el dominio (1.0 == 1) pero el one-hot la leería como "1.0".
    if not pd.api.types.is_integer_dtype(frame["a"]):
        raise ValueError("Revisar tipo de a: tiene que ser entero.")
    for col, dominio in CATEGORY_DOMAINS.items():
        if not frame[col].dropna().isin(dominio).all():
            raise ValueError(f"Revisar dominio de {col}.")
    if not frame["score"].dropna().between(*SCORE_RANGE).all():
        raise ValueError("Revisar escala de score.")


def validate_transactions(frame: pd.DataFrame) -> None:
    """Detener el pipeline ante errores esenciales, en lugar de corregirlos en silencio.

    Una etiqueta inválida o un monto no positivo son un problema de origen, no
    algo a imputar.

    Raises
    ------
    ValueError
        Si alguna regla del contrato de datos no se cumple.
    """
    esperadas = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS + [DATE, TARGET]
    faltantes = sorted(set(esperadas) - set(frame.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas del contrato: {faltantes}.")
    if not pd.api.types.is_datetime64_any_dtype(frame[DATE]):
        raise ValueError("Revisar formato de fecha.")
    if frame[DATE].isna().any():
        raise ValueError("Revisar fechas ausentes.")
    check_labels(frame, TARGET)
    if frame[AMOUNT].isna().any() or not np.isfinite(frame[AMOUNT]).all():
        raise ValueError("Revisar montos ausentes o infinitos.")
    if not frame[AMOUNT].gt(0).all():
        raise ValueError("Revisar montos no positivos y posibles devoluciones.")
    _check_domains(frame)

    numericas = [
        col
        for col in frame.select_dtypes(include="number").columns
        if col not in CATEGORICAL_COLUMNS + BINARY_COLUMNS + [TARGET]
    ]
    if numericas != NUMERIC_COLUMNS:
        raise ValueError(f"El contrato de columnas numéricas cambió: {numericas}.")
    if np.isinf(frame[numericas]).any().any():
        raise ValueError("Revisar valores infinitos.")
    logger.info("contrato de datos verificado")


def split_development(
    frame: pd.DataFrame, test_start: pd.Timestamp = TEST_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separar desarrollo del período reservado, ordenados por fecha.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con la columna de fecha válida.
    test_start : pandas.Timestamp, default=TEST_START
        Primera fecha del período reservado.

    Returns
    -------
    development : pandas.DataFrame
        Todo lo anterior al corte. Acá se toman todas las decisiones.
    test : pandas.DataFrame
        El período reservado para la evaluación final.

    Raises
    ------
    ValueError
        Si alguno de los dos queda vacío o si se solapan en el tiempo.
    """
    ordenado = frame.sort_values(DATE)
    development = ordenado.loc[ordenado[DATE] < test_start].reset_index(drop=True)
    test = ordenado.loc[ordenado[DATE] >= test_start].reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError(f"El corte {test_start:%Y-%m-%d} deja un conjunto vacío.")
    if development[DATE].max() >= test[DATE].min():
        raise ValueError("Los períodos se solapan.")
    logger.info(
        "desarrollo: %d filas hasta %s | reservado: %d filas desde %s",
        len(development),
        f"{development[DATE].max():%Y-%m-%d}",
        len(test),
        f"{test[DATE].min():%Y-%m-%d}",
    )
    return development, test


def load_periods(csv: Path | str = DATASET_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Desarrollo y período reservado, con el texto normalizado y el contrato validado."""
    frame = normalize_text(load_transactions(csv))
    validate_transactions(frame)
    return split_development(frame)
