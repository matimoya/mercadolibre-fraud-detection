"""Métrica de negocio: ganancia esperada y umbral de decisión."""

import pandas as pd
import pytest

from fraud_detection.constants import AMOUNT, GAIN_RATE, TARGET
from fraud_detection.evaluation import (
    approve_all,
    expected_gain,
    gain_by_policy,
    gain_contribution,
    max_gain,
    optimal_probability_threshold,
)

UMBRAL = optimal_probability_threshold()


def frame(etiquetas: list[int], montos: list[float]) -> pd.DataFrame:
    """Las dos únicas columnas que la ganancia necesita."""
    return pd.DataFrame({TARGET: etiquetas, AMOUNT: montos})


def test_umbral():
    """El 0,2 va literal: es el valor con el que decide el modelo final."""
    assert UMBRAL == pytest.approx(0.2)


@pytest.mark.parametrize("desvio, conviene", [(-0.1, True), (-0.01, True), (0.0, False)])
@pytest.mark.parametrize("monto", [5.0, 3696.0])
def test_punto_de_indiferencia(desvio: float, conviene: bool, monto: float):
    """Aprobar rinde más que rechazar solo por debajo del umbral, y el monto no lo mueve.

    La tasa se construye a partir del umbral en vez de escribir 0,2, para que
    siga probando el borde correcto si cambiara la regla de negocio.
    """
    total, tasa = 1000, UMBRAL + desvio
    fraudes = round(total * tasa)
    grupo = frame([1] * fraudes + [0] * (total - fraudes), [monto] * total)

    assert bool(expected_gain(grupo, approve_all(grupo)) > 0) is conviene


def test_ganancia():
    """Las legítimas rinden su proporción; el fraude aprobado se lleva su monto entero."""
    grupo = frame([0, 0, 1], [100.0, 100.0, 200.0])
    legitimas = fraude = 200.0

    assert expected_gain(grupo, approve_all(grupo)) == pytest.approx(GAIN_RATE * legitimas - fraude)
    assert expected_gain(grupo, ~approve_all(grupo)) == pytest.approx(0.0)
    assert max_gain(grupo) == pytest.approx(GAIN_RATE * legitimas)


def test_tabla_de_politicas():
    """Cada fila tiene que dar lo mismo que calcular esa política por separado."""
    grupo = frame([0, 1, 0, 1], [10.0, 20.0, 30.0, 40.0])
    politicas = {
        "aprobar todo": approve_all(grupo),
        "rechazar todo": ~approve_all(grupo),
        "rechazar fraude": grupo[TARGET].eq(0),
    }

    tabla = gain_by_policy(grupo, politicas)

    for nombre, mascara in politicas.items():
        assert tabla.loc[nombre, "ganancia"] == pytest.approx(expected_gain(grupo, mascara))
    assert tabla.loc["rechazar fraude", "pct_del_maximo"] == pytest.approx(100.0)


def test_aporte_por_transaccion():
    """Abrir la ganancia fila por fila tiene que dar lo mismo que calcularla entera.

    Es la aserción que `05_evaluation` hacía a mano: si las dos formas de
    aplicar la regla de negocio se separan, el desglose del informe deja de
    sumar lo que dice la tabla.
    """
    grupo = frame([0, 1, 0, 1], [10.0, 20.0, 30.0, 40.0])
    rechaza = pd.Series([False, True, True, False], index=grupo.index)

    aporte = gain_contribution(grupo, rechaza)

    assert aporte.sum() == pytest.approx(
        expected_gain(grupo, ~rechaza) - expected_gain(grupo, approve_all(grupo))
    )
    # Rechazar evita el monto entero del fraude y cuesta la ganancia de la legítima.
    assert list(aporte) == pytest.approx([0.0, 20.0, -GAIN_RATE * 30.0, 0.0])


def test_mascara_desalineada():
    """Alinear mal la máscara mediría otra política sin avisar."""
    grupo = frame([0, 1], [10.0, 20.0])
    with pytest.raises(ValueError, match="alineada"):
        expected_gain(grupo, pd.Series([True], index=[0]))
