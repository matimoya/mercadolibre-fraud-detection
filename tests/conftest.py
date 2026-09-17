"""Datos sintéticos compartidos: la suite no depende del CSV del enunciado."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.constants import AMOUNT, DATE, TARGET

FILAS = 300
DIAS = 40

# Las mismas columnas con nulos que el dataset real: de ahí sale cuántos
# indicadores de ausencia produce el preprocesador.
CON_NULOS = ["b", "c", "d", "f", "l", "m"]


@pytest.fixture(name="con_nulos")
def fixture_con_nulos() -> list[str]:
    """Columnas numéricas con nulos."""
    return CON_NULOS


@pytest.fixture(name="riesgo")
def fixture_riesgo() -> np.ndarray:
    """Variable latente que gobierna la etiqueta, `score` y `o`."""
    return np.random.default_rng(0).random(FILAS)


@pytest.fixture(name="etiqueta")
def fixture_etiqueta(riesgo: np.ndarray) -> pd.Series:
    """Etiqueta aprendible: el encoding por target necesita algo que codificar."""
    return pd.Series((riesgo > 0.7).astype(int), name=TARGET)


@pytest.fixture(name="predictores")
def fixture_predictores(riesgo: np.ndarray) -> pd.DataFrame:
    """Frame con el contrato de columnas del enunciado."""
    azar = np.random.default_rng(1)
    horas = azar.integers(0, DIAS * 24, FILAS)
    predictores = pd.DataFrame(
        {
            "a": azar.integers(1, 5, FILAS),
            "b": azar.random(FILAS),
            "c": azar.random(FILAS) * 1000,
            "d": azar.random(FILAS) * 50,
            "e": azar.random(FILAS),
            "f": azar.random(FILAS) * 100 - 5,
            "g": azar.choice(["AR", "BR"], FILAS),
            "h": azar.integers(0, 60, FILAS),
            "j": [f"cat_{numero % 12}" for numero in range(FILAS)],
            "k": azar.random(FILAS),
            "l": azar.random(FILAS) * 5000,
            "m": azar.random(FILAS) * 2000,
            "n": azar.integers(0, 2, FILAS),
            "o": pd.Series(np.where(riesgo > 0.8, "N", "Y")).where(riesgo > 0.5),
            "p": azar.choice(["Y", "N"], FILAS),
            DATE: pd.Timestamp("2020-03-01") + pd.to_timedelta(horas, "h"),
            AMOUNT: azar.random(FILAS) * 100 + 1,
            "score": (riesgo * 100).astype(int),
        }
    )
    predictores.loc[predictores.index[:30], CON_NULOS] = np.nan
    return predictores


@pytest.fixture(name="development")
def fixture_development(predictores: pd.DataFrame, etiqueta: pd.Series) -> pd.DataFrame:
    """Predictores y etiqueta juntos, ordenados por fecha."""
    return predictores.assign(**{TARGET: etiqueta}).sort_values(DATE).reset_index(drop=True)
