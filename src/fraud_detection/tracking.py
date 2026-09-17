"""Registro de experimentos con MLflow.

El store queda en la raíz del proyecto y no en `notebooks/`, para que las tres
etapas escriban en el mismo lugar sin importar desde dónde se ejecuten. Es una
base SQLite: no hace falta levantar ningún servidor, y MLflow 3.16 dejó de
aceptar el store de archivos.
"""

import logging
import os
from pathlib import Path

# El aviso de MLflow sobre su skill de tracing se emite al importar el paquete,
# así que la variable tiene que quedar antes del import y no junto al resto.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # pylint: disable=wrong-import-position

TRACKING_DB = "../mlflow.db"
ARTIFACTS_DIR = "../mlartifacts"

# MLflow avisa por INFO cada vez que crea un experimento o una corrida.
logging.getLogger("mlflow").setLevel(logging.WARNING)


def configurar(
    experimento: str, tracking_db: str = TRACKING_DB, artifacts_dir: str = ARTIFACTS_DIR
) -> str:
    """Apuntar MLflow al store local y seleccionar el experimento.

    Parameters
    ----------
    experimento : str
        Nombre bajo el que se agrupan las corridas de esta etapa.
    tracking_db : str, default=TRACKING_DB
        Base SQLite con parámetros y métricas, relativa al directorio del kernel.
    artifacts_dir : str, default=ARTIFACTS_DIR
        Carpeta de artefactos —modelos, tablas—, que no entran en la base.

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
