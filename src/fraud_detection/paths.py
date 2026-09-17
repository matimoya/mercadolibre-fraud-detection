"""Rutas del proyecto, resueltas desde la raíz del repositorio.

Separado de `constants` a propósito: eso es el contrato de datos y esto es la
ubicación de los archivos. Antes cada módulo armaba rutas contra el directorio
del kernel, así que los notebooks solo corrían desde `notebooks/`.
"""

import os
from pathlib import Path

MARCA = "pyproject.toml"
VARIABLE_DE_ENTORNO = "FRAUD_DETECTION_ROOT"

DATASET = "MercadoLibre Data Scientist Technical Challenge - Dataset.csv"


def encontrar_raiz() -> Path:
    """Ubicar el directorio que contiene `pyproject.toml`.

    Raises
    ------
    RuntimeError
        Si el paquete quedó instalado fuera del repositorio y no se declaró
        `FRAUD_DETECTION_ROOT`.
    """
    declarada = os.environ.get(VARIABLE_DE_ENTORNO)
    raiz = Path(declarada).resolve() if declarada else Path(__file__).resolve().parents[2]
    if not (raiz / MARCA).is_file():
        raise RuntimeError(f"No hay {MARCA} en {raiz}. Revisar {VARIABLE_DE_ENTORNO}.")
    return raiz


ROOT = encontrar_raiz()

DATA_DIR = ROOT / "data"
DATASET_CSV = DATA_DIR / DATASET
MODELS_DIR = ROOT / "models"
BEST_PARAMS_JSON = MODELS_DIR / "mejores_parametros.json"
MODEL_JOBLIB = MODELS_DIR / "xgboost.joblib"
NOTEBOOKS_DIR = ROOT / "notebooks"
FIGURES_DIR = ROOT / "informe" / "figuras"

TRACKING_DB = ROOT / "mlflow.db"
ARTIFACTS_DIR = ROOT / "mlartifacts"
