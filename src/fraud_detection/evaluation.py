"""Métrica de negocio: ganancia esperada de una política de aprobación.

Simula qué habría rendido cada política sobre las transacciones históricas,
usando sus etiquetas ya conocidas.
"""

import pandas as pd

from fraud_detection.constants import AMOUNT, GAIN_RATE, TARGET
from fraud_detection.diagnostics import check_labels


def optimal_probability_threshold(gain_rate: float = GAIN_RATE) -> float:
    """Umbral de probabilidad que maximiza la ganancia esperada.

    Aprobar conviene cuando gain_rate * m * (1 - p) > m * p. Como el monto m
    es positivo se cancela, y el umbral no depende de él.

    Parameters
    ----------
    gain_rate : float, default=GAIN_RATE
        Ganancia por unidad de monto de una legítima aprobada.

    Returns
    -------
    float
        gain_rate / (1 + gain_rate): probabilidad por debajo de la cual
        conviene aprobar.

    Notes
    -----
    Vale solo sin costo fijo por rechazo o revisión; un costo fijo no escala
    con el monto y haría que el umbral dependiera de él.
    """
    return gain_rate / (1 + gain_rate)


def expected_gain(
    frame: pd.DataFrame,
    approve: pd.Series,
    gain_rate: float = GAIN_RATE,
    target: str = TARGET,
    amount: str = AMOUNT,
) -> float:
    """Ganancia de aprobar el subconjunto indicado por la máscara.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con etiqueta binaria y monto en unidades originales.
    approve : pandas.Series
        Máscara booleana alineada con frame: True aprueba.
    gain_rate : float, default=GAIN_RATE
        Ganancia por unidad de monto de una legítima aprobada.
    target, amount : str
        Columnas de etiqueta y monto.

    Returns
    -------
    float
        gain_rate * monto de legítimas aprobadas - monto de fraudes aprobados.

    Raises
    ------
    ValueError
        Si la máscara no está alineada con frame o contiene nulos.
    """
    check_labels(frame, target)
    if not approve.index.equals(frame.index) or approve.isna().any():
        raise ValueError("La máscara debe estar alineada y no contener nulos.")
    approved = frame.loc[approve.astype(bool)]
    legit = approved.loc[approved[target].eq(0), amount].sum()
    fraud = approved.loc[approved[target].eq(1), amount].sum()
    return gain_rate * legit - fraud


def max_gain(
    frame: pd.DataFrame, gain_rate: float = GAIN_RATE, target: str = TARGET, amount: str = AMOUNT
) -> float:
    """Ganancia máxima alcanzable: aprobar todas las legítimas y ningún fraude.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con etiqueta binaria y monto en unidades originales.
    gain_rate : float, default=GAIN_RATE
        Ganancia por unidad de monto de una legítima aprobada.
    target, amount : str
        Columnas de etiqueta y monto.

    Returns
    -------
    float
        Techo teórico de una predicción perfecta; ningún modelo lo supera.
    """
    return expected_gain(frame, frame[target].eq(0), gain_rate, target, amount)


def sweep_gain(
    frame: pd.DataFrame,
    values: pd.Series,
    thresholds,
    gain_rate: float = GAIN_RATE,
    target: str = TARGET,
    amount: str = AMOUNT,
) -> pd.Series:
    """Probar muchos cortes: ganancia de aprobar cuando values < t.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con etiqueta binaria y monto en unidades originales.
    values : pandas.Series
        Puntaje de riesgo alineado con frame: un score entero o una
        probabilidad. Valores más altos significan más riesgo.
    thresholds : iterable
        Los cortes a probar, uno por uno.
    gain_rate : float, default=GAIN_RATE
        Ganancia por unidad de monto de una legítima aprobada.
    target, amount : str
        Columnas de etiqueta y monto.

    Returns
    -------
    pandas.Series
        Una fila por corte probado: el corte y la ganancia que habría dado.

    Raises
    ------
    ValueError
        Si values no está alineada con frame.

    Notes
    -----
    Quedarse con el mejor corte y reportarlo sobre los mismos datos con los
    que se eligió da una ganancia optimista. Para una vara honesta, elegir el
    corte con el entrenamiento de cada fold y medirlo en su validación.
    """
    if not values.index.equals(frame.index):
        raise ValueError("values debe estar alineada con frame.")
    return pd.Series(
        {
            corte: expected_gain(frame, values.lt(corte).fillna(False), gain_rate, target, amount)
            for corte in thresholds
        },
        dtype=float,
    )


def gain_by_policy(
    frame: pd.DataFrame,
    policies: dict[str, pd.Series],
    gain_rate: float = GAIN_RATE,
    target: str = TARGET,
    amount: str = AMOUNT,
) -> pd.DataFrame:
    """Comparar políticas de aprobación entre sí y contra la ganancia máxima.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con etiqueta binaria y monto en unidades originales.
    policies : dict of str to pandas.Series
        Nombre de la política y su máscara de aprobación.
    gain_rate : float, default=GAIN_RATE
        Ganancia por unidad de monto de una legítima aprobada.
    target, amount : str
        Columnas de etiqueta y monto.

    Returns
    -------
    pandas.DataFrame
        Por política: ganancia, pct_del_maximo (porcentaje del techo que
        devuelve max_gain), aprobadas, aprobadas_pct y fraudes_aprobados.
    """
    techo = max_gain(frame, gain_rate, target, amount)
    rows = {}
    for name, approve in policies.items():
        mask = approve.astype(bool)
        gain = expected_gain(frame, mask, gain_rate, target, amount)
        rows[name] = {
            "ganancia": gain,
            "pct_del_maximo": 100 * gain / techo if techo else float("nan"),
            "aprobadas": int(mask.sum()),
            "aprobadas_pct": 100 * mask.sum() / len(frame),
            "fraudes_aprobados": int(frame.loc[mask, target].sum()),
        }
    return pd.DataFrame(rows).T
