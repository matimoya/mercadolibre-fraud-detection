"""Registro de una corrida."""

import logging
from pathlib import Path

import pytest

from fraud_detection.logs import registrar_corrida


def test_registro(tmp_path: Path):
    """Si la corrida falla el traceback queda en el archivo, y el logging vuelve a como estaba.

    Lo segundo protege a los notebooks: sin handlers heredados, el INFO del
    paquete no aparece en sus salidas.
    """
    raiz, paquete = logging.getLogger(), logging.getLogger("fraud_detection")
    handlers_previos, nivel_previo = list(raiz.handlers), paquete.level

    with pytest.raises(RuntimeError):
        with registrar_corrida("prueba", carpeta=tmp_path) as archivo:
            logging.getLogger("fraud_detection.dataset").debug("detalle por bloque")
            raise RuntimeError("etapa rota")

    registro = archivo.read_text()
    assert "detalle por bloque" in registro, "el archivo guarda DEBUG aunque la consola no"
    assert "RuntimeError: etapa rota" in registro
    assert raiz.handlers == handlers_previos
    assert paquete.level == nivel_previo
