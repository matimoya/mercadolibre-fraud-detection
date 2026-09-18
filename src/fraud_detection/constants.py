"""Contrato de datos, reglas de negocio y configuración de ejecución."""

import pandas as pd

# 25% de ganancia por transacción legítima aprobada, 100% de pérdida por fraude.
GAIN_RATE = 0.25

# Fechas posteriores reservadas para la evaluación final.
TEST_START = pd.Timestamp("2020-04-13")

# Folds temporales: 4 es la menor dispersión sobre los 36 días de desarrollo.
N_SPLITS = 4

# Pruebas de la búsqueda de hiperparámetros.
N_TRIALS = 40

# Semilla única, para que el corrector reproduzca los números del informe.
RANDOM_STATE = 42

TARGET = "fraude"
AMOUNT = "monto"
DATE = "fecha"

NUMERIC_COLUMNS = ["b", "c", "d", "e", "f", "h", "k", "l", "m", "monto", "score"]

# 'a' es nominal: su tasa de fraude no es monótona en el orden 1-4.
LOW_CARDINALITY_COLUMNS = ["a", "o", "p"]
HIGH_CARDINALITY_COLUMNS = ["g", "j"]
CATEGORICAL_COLUMNS = sorted(LOW_CARDINALITY_COLUMNS + HIGH_CARDINALITY_COLUMNS)

BINARY_COLUMNS = ["n"]

# Dominio cerrado: un código nuevo pasaría como fila de ceros en el one-hot, sin error.
CATEGORY_DOMAINS = {"a": {1, 2, 3, 4}, "o": {"N", "Y"}, "p": {"N", "Y"}}
SCORE_RANGE = (0, 100)

# 'k' es uniforme(0, 1) e independiente de la etiqueta: variable de control.
NOISE_COLUMNS = ["k"]

# --- Configuración de ejecución: dónde se registra cada corrida y cómo se ve. ---

# Un experimento de MLflow por notebook que registra; la CLI escribe en el de la búsqueda.
EXPERIMENTO_MODELOS = "03-comparacion-modelos"
EXPERIMENTO_BUSQUEDA = "04-busqueda-hiperparametros"
EXPERIMENTO_EVALUACION = "05-evaluacion-final"

FORMATO_LOG = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
FORMATO_HORA_LOG = "%H:%M:%S"
