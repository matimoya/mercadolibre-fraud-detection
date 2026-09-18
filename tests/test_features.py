"""Preprocesamiento de features."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.constants import DATE
from fraud_detection.features import build_preprocessor


@pytest.fixture(name="preprocesador")
def fixture_preprocesador(predictores: pd.DataFrame, etiqueta: pd.Series):
    """Preprocesador ajustado sobre los datos sintéticos."""
    return build_preprocessor().fit(predictores, etiqueta)


def test_encoding_cruzado(predictores: pd.DataFrame, etiqueta: pd.Series, preprocesador):
    """`fit_transform` cruza internamente y `fit().transform()` no.

    Usar el segundo sobre el entrenamiento le da a cada fila la tasa de fraude
    que ella misma ayudó a calcular.
    """
    posicion = list(preprocesador.get_feature_names_out()).index("j")

    de_una_pasada = build_preprocessor().fit_transform(predictores, etiqueta)[:, posicion]
    en_dos_pasos = preprocesador.transform(predictores)[:, posicion]

    assert not np.allclose(de_una_pasada, en_dos_pasos)


def test_serializacion(preprocesador, predictores: pd.DataFrame):
    """El preprocesador ajustado sobrevive a joblib y transforma igual."""
    buffer = io.BytesIO()
    joblib.dump(preprocesador, buffer)
    buffer.seek(0)

    recuperado = joblib.load(buffer)

    np.testing.assert_allclose(
        recuperado.transform(predictores), preprocesador.transform(predictores)
    )


def test_categoria_nueva(preprocesador, predictores: pd.DataFrame):
    """Una categoría no vista al entrenar no genera nulos ni cambia las columnas."""
    nueva = predictores.iloc[[0]].copy()
    nueva["j"] = "cat_que_nunca_existio"
    nueva["g"] = "US"

    transformada = preprocesador.transform(nueva)

    assert not np.isnan(transformada).any()
    assert transformada.shape[1] == preprocesador.transform(predictores).shape[1]


def test_columnas(
    predictores: pd.DataFrame, etiqueta: pd.Series, preprocesador, con_nulos: list[str]
):
    """Indicadores de ausencia, categoría propia para el nulo de `o` y variante sin `score`."""
    con_score = list(preprocesador.get_feature_names_out())
    sin_score = list(
        build_preprocessor(use_score=False).fit(predictores, etiqueta).get_feature_names_out()
    )

    assert len(con_score) == len(sin_score) + 1
    assert "score" in con_score and "score" not in sin_score
    assert "o_Ausente" in con_score
    assert {f"missingindicator_{columna}" for columna in con_nulos} <= set(con_score)


def test_hora_ciclica(preprocesador, predictores: pd.DataFrame):
    """Las 23 quedan más cerca de las 0 que del mediodía."""
    columnas = list(preprocesador.get_feature_names_out())
    ejes = [columnas.index("hora_sin"), columnas.index("hora_cos")]

    def a_las(hora: int) -> np.ndarray:
        fila = predictores.iloc[[0]].copy()
        fila[DATE] = fila[DATE].dt.normalize() + pd.Timedelta(hours=hora)
        return preprocesador.transform(fila)[0, ejes]

    medianoche, once_de_la_noche, mediodia = a_las(0), a_las(23), a_las(12)

    assert np.linalg.norm(medianoche - once_de_la_noche) < np.linalg.norm(medianoche - mediodia)
