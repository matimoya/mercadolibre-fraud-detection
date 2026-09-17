"""Carga del dataset y reserva del período de evaluación.

Centralizado para que todos los notebooks partan del mismo frame: misma
normalización de texto y mismo índice. No guarda nada en disco.
"""

from pathlib import Path

import pandas as pd

from fraud_detection.constants import DATE, TEST_START
from fraud_detection.paths import DATASET_CSV


def load_transactions(csv: Path | str = DATASET_CSV) -> pd.DataFrame:
    """Leer el CSV del enunciado con la fecha ya parseada, sin filtrar nada."""
    return pd.read_csv(csv, parse_dates=[DATE])


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
    return development, test
