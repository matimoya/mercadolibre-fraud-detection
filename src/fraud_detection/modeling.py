"""Utilidades de validación temporal para futuros experimentos de modelado."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def temporal_folds(dates: pd.Series, n_splits: int = 3, gap_days: int = 0):
    """Generar folds expansivos con TimeSeriesSplit sobre días calendario.

    Parameters
    ----------
    dates : pandas.Series
        Fechas válidas, aunque las filas no estén ordenadas.
    n_splits : int, default=3
        Número de bloques de validación.
    gap_days : int, default=0
        Días excluidos entre entrenamiento y validación.

    Yields
    ------
    train_positions : numpy.ndarray
        Posiciones para seleccionar entrenamiento mediante iloc.
    valid_positions : numpy.ndarray
        Posiciones para seleccionar validación mediante iloc.

    Raises
    ------
    ValueError
        Si hay fechas inválidas o algún bloque no tiene transacciones.

    Notes
    -----
    Se usa una grilla diaria regular, incluidos días sin transacciones.
    Cada fold evalúa igual duración, pero puede contener distinto volumen.
    gap_days debe justificarse con el retraso de etiquetas y las ventanas
    de construcción de variables; cero es una hipótesis provisional.
    """
    parsed = pd.to_datetime(dates, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.empty or parsed.isna().any():
        raise ValueError("Se requieren fechas válidas para todas las filas.")
    day = parsed.dt.normalize()
    calendar = pd.date_range(day.min(), day.max(), freq="D")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    for train_days, valid_days in splitter.split(calendar):
        train_idx = np.flatnonzero(day.isin(calendar[train_days]))
        valid_idx = np.flatnonzero(day.isin(calendar[valid_days]))
        if train_idx.size == 0 or valid_idx.size == 0:
            raise ValueError("Un fold no tiene transacciones: revisar cortes.")
        yield train_idx, valid_idx
