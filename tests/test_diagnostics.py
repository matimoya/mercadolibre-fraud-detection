"""Validaciones de datos y tablas por segmento."""

import pandas as pd
import pytest

from fraud_detection.constants import AMOUNT, TARGET
from fraud_detection.diagnostics import check_labels, iqr_report, rate_table


@pytest.fixture(name="segmentos")
def fixture_segmentos() -> pd.DataFrame:
    """Dos grupos con tasas de fraude conocidas: 50% y 25%."""
    return pd.DataFrame(
        {
            "grupo": ["alto"] * 4 + ["bajo"] * 8,
            TARGET: [1, 1, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0],
            AMOUNT: [100.0] * 12,
        }
    )


@pytest.mark.parametrize("etiquetas", [[0, 1, 2], [0, 1, None]])
def test_etiqueta_invalida(etiquetas: list):
    """Es el guardia de `expected_gain`: medir ganancia sobre esto daría un número inventado."""
    with pytest.raises(ValueError):
        check_labels(pd.DataFrame({TARGET: etiquetas}))


def test_tasas(segmentos: pd.DataFrame):
    """Tasa por grupo, lift contra la base y monto expuesto, que es la columna que decide."""
    tabla = rate_table(segmentos, "grupo")
    base = segmentos[TARGET].mean()

    assert tabla.loc["alto", "fraude_pct"] == pytest.approx(50.0)
    assert tabla.loc["bajo", "fraude_pct"] == pytest.approx(25.0)
    assert tabla.loc["alto", "lift"] == pytest.approx(0.5 / base)
    assert tabla.loc["alto", "monto_en_fraude"] == pytest.approx(200.0)


def test_tramos(segmentos: pd.DataFrame):
    """Una variable continua se agrupa por tramo sin perder filas."""
    continua = segmentos.assign(puntaje=range(12))

    tabla = rate_table(continua, "puntaje", bins=[-1, 5, 11])

    assert len(tabla) == 2
    assert tabla.transacciones.sum() == len(continua)


def test_cercas():
    """Recortar por este criterio se llevaba el 16,75% de los fraudes: por eso no se aplicó."""
    datos = pd.DataFrame({TARGET: [0] * 11 + [1], AMOUNT: [10.0] * 11 + [10_000.0]})

    cercas = iqr_report(datos, [AMOUNT]).loc[AMOUNT]

    assert cercas.limite_superior < 10_000.0
