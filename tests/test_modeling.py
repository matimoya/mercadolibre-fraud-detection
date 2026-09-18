"""Validación temporal."""

import pandas as pd
import pytest

from fraud_detection.constants import N_SPLITS
from fraud_detection.modeling import temporal_folds

DIAS = 40
POR_DIA = 5
HUECO = 5


def fechas() -> pd.Series:
    """Marcas de tiempo repartidas en días consecutivos."""
    calendario = pd.date_range("2020-03-01", periods=DIAS, freq="D")
    return pd.Series(calendario.repeat(POR_DIA))


def test_orden_temporal():
    """Ningún bloque entrena con datos posteriores a los que valida.

    El default tiene que coincidir con `N_SPLITS`: si difiriera, un llamado sin
    argumento partiría distinto al resto del proyecto.
    """
    marcas = fechas()
    bloques = list(temporal_folds(marcas))

    assert len(bloques) == N_SPLITS
    for entrenamiento, validacion in bloques:
        assert len(entrenamiento) and len(validacion)
        assert marcas.iloc[entrenamiento].max() < marcas.iloc[validacion].min()


def test_ventana_expansiva():
    """Cada bloque entrena con todo lo anterior, no con una ventana de tamaño fijo."""
    tamanos = [len(entrenamiento) for entrenamiento, _ in temporal_folds(fechas())]
    assert tamanos == sorted(tamanos)
    assert tamanos[0] < tamanos[-1]


@pytest.mark.parametrize("dias", [1, 3, 7])
def test_gap(dias: int):
    """Simula la demora en confirmar el fraude: esos días no se pueden usar."""
    marcas = fechas()
    for entrenamiento, validacion in temporal_folds(marcas, gap_days=dias):
        separacion = marcas.iloc[validacion].min() - marcas.iloc[entrenamiento].max()
        assert separacion >= pd.Timedelta(days=dias)


def test_gap_recorta_solo_el_entrenamiento():
    """Parte de la caída al simular demora es tener menos historia, no la demora en sí.

    Por eso la sensibilidad al `gap` se reporta como cota pesimista.
    """
    marcas = fechas()
    sin_hueco = list(temporal_folds(marcas))
    con_hueco = list(temporal_folds(marcas, gap_days=HUECO))

    assert [len(valid) for _, valid in sin_hueco] == [len(valid) for _, valid in con_hueco]
    assert all(
        len(recortado) < len(completo)
        for (completo, _), (recortado, _) in zip(sin_hueco, con_hueco)
    )


def test_fecha_invalida():
    """Una fecha que no parsea correría el orden temporal sin que nada avise."""
    marcas = fechas()
    marcas.iloc[0] = pd.NaT
    with pytest.raises(ValueError):
        list(temporal_folds(marcas))
