"""Cableado de la línea de comandos."""

import pytest

from fraud_detection import cli
from fraud_detection.constants import N_TRIALS


@pytest.mark.parametrize(
    "comando, funcion",
    [("tune", cli.tune), ("train", cli.train), ("evaluate", cli.evaluate), ("all", cli.run_all)],
)
def test_cada_comando_enruta_a_su_funcion(comando: str, funcion):
    """Un comando apuntando a la función equivocada no lo detecta ningún otro test."""
    argumentos = cli.construir_parser().parse_args([comando])
    assert argumentos.funcion is funcion
    assert argumentos.verbose is False


def test_trials_tiene_default_y_se_puede_pisar():
    """El default sale de `constants`, no de un número escrito en la CLI."""
    parser = cli.construir_parser()
    assert parser.parse_args(["tune"]).trials == N_TRIALS
    assert parser.parse_args(["tune", "--trials", "3"]).trials == 3


def test_sin_comando_no_hace_nada():
    """Sin subcomando argparse corta con error, en vez de correr algo por omisión."""
    with pytest.raises(SystemExit):
        cli.construir_parser().parse_args([])
