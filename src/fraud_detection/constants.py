"""Contrato de datos y reglas de negocio compartidos entre notebooks."""

import pandas as pd

# 25% de ganancia por transacción legítima aprobada, 100% de pérdida por fraude.
GAIN_RATE = 0.25

# Fechas posteriores reservadas para la evaluación final.
TEST_START = pd.Timestamp("2020-04-13")

# Folds temporales: 4 es la menor dispersión sobre los 36 días de desarrollo.
N_SPLITS = 4

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

# 'k' es uniforme(0, 1) e independiente de la etiqueta: variable de control.
NOISE_COLUMNS = ["k"]
