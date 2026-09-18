"""Registro de una corrida del pipeline: consola y un archivo por corrida.

Los demás módulos solo emiten con `logging.getLogger(__name__)` y nunca
configuran handlers. Por eso en los notebooks, donde nadie llama a esto, el
INFO del paquete no aparece y las salidas no cambian.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fraud_detection.constants import (
    FORMATO_HORA_LOG,
    FORMATO_LOG,
    N_SPLITS,
    RANDOM_STATE,
    TEST_START,
)
from fraud_detection.paths import LOGS_DIR, desde_raiz

logger = logging.getLogger(__name__)


@contextmanager
def registrar_corrida(
    comando: str, detallado: bool = False, carpeta: Path = LOGS_DIR
) -> Iterator[Path]:
    """Configurar los logs mientras dura el bloque y dejar la corrida en un archivo.

    Solo el logger del paquete baja de WARNING: de las otras librerías llegan
    los avisos, no su INFO. Si el bloque falla, el traceback queda en el
    archivo antes de propagarse.

    Parameters
    ----------
    comando : str
        Nombre de la corrida; encabeza el nombre del archivo.
    detallado : bool, default=False
        Mostrar DEBUG en consola. El archivo siempre guarda DEBUG.
    carpeta : pathlib.Path, default=LOGS_DIR
        Dónde escribir el archivo.

    Yields
    ------
    pathlib.Path
        El archivo con el registro completo.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"{comando}-{datetime.now():%Y%m%d-%H%M%S}.log"

    consola = logging.StreamHandler()
    consola.setLevel(logging.DEBUG if detallado else logging.INFO)
    consola.setFormatter(logging.Formatter(FORMATO_LOG, datefmt=FORMATO_HORA_LOG))
    registro = logging.FileHandler(archivo, encoding="utf-8")
    registro.setFormatter(logging.Formatter(FORMATO_LOG))

    raiz, paquete = logging.getLogger(), logging.getLogger(__package__)
    niveles_previos = raiz.level, paquete.level
    raiz.setLevel(logging.WARNING)
    for handler in (consola, registro):
        raiz.addHandler(handler)
    paquete.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    logger.info("corrida %s | log en %s", comando, desde_raiz(archivo))
    logger.debug(
        "RANDOM_STATE=%d | TEST_START=%s | N_SPLITS=%d",
        RANDOM_STATE,
        f"{TEST_START:%Y-%m-%d}",
        N_SPLITS,
    )
    inicio = time.perf_counter()
    try:
        yield archivo
    except Exception:
        logger.exception("falló tras %.0f s", time.perf_counter() - inicio)
        raise
    else:
        logger.info("terminado en %.0f s", time.perf_counter() - inicio)
    finally:
        # Devolver el logging como estaba: quien reutilice esto no hereda handlers abiertos.
        logging.captureWarnings(False)
        raiz.setLevel(niveles_previos[0])
        paquete.setLevel(niveles_previos[1])
        for handler in (consola, registro):
            raiz.removeHandler(handler)
            handler.close()
