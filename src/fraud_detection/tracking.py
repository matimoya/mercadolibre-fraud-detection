"""Registro de experimentos con MLflow.

Es una base SQLite en la raíz del proyecto: no hace falta levantar ningún
servidor, y MLflow 3.16 dejó de aceptar el store de archivos.
"""

import logging
import os
from pathlib import Path

from fraud_detection.paths import ARTIFACTS_DIR, TRACKING_DB

# El aviso de MLflow sobre su skill de tracing se emite al importar el paquete,
# así que la variable tiene que quedar antes del import y no junto al resto.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # pylint: disable=wrong-import-position,wrong-import-order

# MLflow avisa por INFO cada vez que crea un experimento o una corrida.
logging.getLogger("mlflow").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def configurar(
    experimento: str, tracking_db: Path = TRACKING_DB, artifacts_dir: Path = ARTIFACTS_DIR
) -> str:
    """Apuntar MLflow al store local y seleccionar el experimento.

    Returns
    -------
    str
        La URI del store, para poder mostrarla en el notebook.
    """
    base = Path(tracking_db).resolve()
    artefactos = Path(artifacts_dir).resolve()
    artefactos.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{base}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=artefactos.as_uri())
    mlflow.set_experiment(experimento)
    logger.debug("MLflow en %s, experimento %s", uri, experimento)
    return uri


def registrar(
    nombre: str,
    parametros: dict | None = None,
    metricas: dict | None = None,
    etiquetas: dict | None = None,
    anidada: bool = False,
) -> str:
    """Guardar una corrida con sus parámetros y métricas.

    Parameters
    ----------
    nombre : str
        Nombre de la corrida, el que se ve en la interfaz.
    parametros : dict, optional
        Configuración del experimento: hiperparámetros, variante, umbral.
    metricas : dict, optional
        Resultados numéricos. Solo acepta valores numéricos.
    etiquetas : dict, optional
        Texto libre para filtrar después, por ejemplo el modelo o la etapa.
    anidada : bool, default=False
        True para colgar la corrida de otra que ya esté abierta.

    Returns
    -------
    str
        El identificador de la corrida.
    """
    with mlflow.start_run(run_name=nombre, nested=anidada) as corrida:
        if parametros:
            mlflow.log_params(parametros)
        if metricas:
            mlflow.log_metrics(metricas)
        if etiquetas:
            mlflow.set_tags(etiquetas)
        return corrida.info.run_id
