"""Carga y reserva temporal del dataset."""

import pandas as pd
import pytest

from fraud_detection.constants import AMOUNT, DATE, TARGET, TEST_START
from fraud_detection.dataset import normalize_text, split_development, validate_transactions

UN_DIA = pd.Timedelta(days=1)


def frame(fechas: list[pd.Timestamp]) -> pd.DataFrame:
    """Un frame con la única columna que la partición mira."""
    return pd.DataFrame({DATE: pd.to_datetime(fechas), "monto": 10.0})


def test_particion():
    """Los períodos no se solapan, el corte cae del lado reservado y no se pierde ninguna fila."""
    datos = frame(
        [TEST_START + 7 * UN_DIA, TEST_START - 3 * UN_DIA, TEST_START, TEST_START - UN_DIA]
    )

    desarrollo, reservado = split_development(datos)

    assert len(desarrollo) + len(reservado) == len(datos)
    assert desarrollo[DATE].max() == TEST_START - UN_DIA
    assert reservado[DATE].min() == TEST_START


@pytest.mark.parametrize("desplazamiento", [UN_DIA, -UN_DIA])
def test_particion_vacia(desplazamiento: pd.Timedelta):
    """Fallar es mejor que devolver un conjunto vacío y medir sobre nada."""
    del_mismo_lado = frame([TEST_START + desplazamiento, TEST_START + 2 * desplazamiento])
    with pytest.raises(ValueError):
        split_development(del_mismo_lado)


def test_normalizacion():
    """Se limpian los códigos de texto y el vacío pasa a nulo, sin tocar los números."""
    datos = pd.DataFrame({"g": [" AR ", "BR", "  "], "monto": [10.5, 20.25, 30.0]})

    limpio = normalize_text(datos)

    assert limpio.g.tolist()[:2] == ["AR", "BR"]
    assert limpio.g.isna().sum() == 1
    pd.testing.assert_series_equal(limpio.monto, datos.monto)


def test_contrato_valido(development: pd.DataFrame):
    """Un frame con el contrato del enunciado pasa la validación sin cambios."""
    validate_transactions(development)


@pytest.mark.parametrize(
    "romper",
    [
        lambda datos: datos.assign(**{AMOUNT: -datos[AMOUNT]}),
        lambda datos: datos.assign(**{TARGET: 2}),
        lambda datos: datos.assign(**{DATE: pd.NaT}),
        lambda datos: datos.drop(columns=["score"]),
        lambda datos: datos.assign(o="X"),
        lambda datos: datos.assign(a=datos.a.astype(float)),
        lambda datos: datos.assign(score=150),
    ],
    ids=[
        "monto_negativo",
        "etiqueta_fuera_de_dominio",
        "fecha_nula",
        "columna_faltante",
        "codigo_nuevo",
        "a_como_float",
        "score_fuera_de_rango",
    ],
)
def test_contrato_invalido(development: pd.DataFrame, romper):
    """Un error de origen detiene el pipeline en vez de corregirse en silencio."""
    with pytest.raises(ValueError):
        validate_transactions(romper(development))
