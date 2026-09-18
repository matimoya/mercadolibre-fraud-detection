"""Construcción de features con disciplina temporal.

Todo lo que aprende parámetros de los datos se ajusta dentro del Pipeline, con
el train de cada fold. No se persiste un dataset procesado: el artefacto es el
Pipeline entrenado.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from fraud_detection.constants import (
    BINARY_COLUMNS,
    DATE,
    HIGH_CARDINALITY_COLUMNS,
    LOW_CARDINALITY_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
)

# El EDA mostró que la frecuencia de 'j' distingue el fraude por sí sola, aparte de su tasa.
FREQUENCY_COLUMNS = ["j"]
TIME_FEATURES = ["hora_sin", "hora_cos", "es_madrugada"]
MISSING_LABEL = "Ausente"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Codificar categorías por su frecuencia en entrenamiento.

    Parameters
    ----------
    normalize : bool, default=True
        Devolver la proporción en lugar del conteo, para que la escala no
        dependa del tamaño del fold.

    Attributes
    ----------
    frequencies_ : dict of str to pandas.Series
        Frecuencia por categoría aprendida en fit, una entrada por columna.

    Notes
    -----
    Las categorías no vistas reciben 0: no aparecieron en el pasado.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def fit(self, X: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        """Aprender la frecuencia de cada categoría observada en ``X``."""
        self.feature_names_in_ = list(X.columns)
        self.frequencies_ = {
            col: X[col].value_counts(normalize=self.normalize) for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transformar categorías en las frecuencias aprendidas durante ``fit``."""
        encoded = {
            col: X[col].map(self.frequencies_[col]).fillna(0.0) for col in self.feature_names_in_
        }
        return pd.DataFrame(encoded, index=X.index).to_numpy(dtype=float)

    def get_feature_names_out(
        self, input_features=None  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Devolver los nombres generados para las columnas transformadas."""
        return np.asarray([f"freq_{col}" for col in self.feature_names_in_])


def _to_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Pasar códigos categóricos a texto, con etiqueta explícita para nulos."""
    return frame.astype(object).map(lambda valor: MISSING_LABEL if pd.isna(valor) else str(valor))


def _time_feature_names(_transformer=None, _input_features=None) -> np.ndarray:
    """Nombres de salida de las features temporales.

    Notes
    -----
    Es una función del módulo y no un lambda a propósito: el Pipeline
    entrenado se serializa con joblib, y pickle no guarda lambdas.
    """
    return np.asarray(TIME_FEATURES)


def _time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derivar hora cíclica y flag de madrugada a partir de la fecha."""
    hour = frame[DATE].dt.hour
    angle = 2 * np.pi * hour / 24
    return pd.DataFrame(
        {
            "hora_sin": np.sin(angle),
            "hora_cos": np.cos(angle),
            "es_madrugada": hour.lt(6).astype(float),
        },
        index=frame.index,
    )


def build_preprocessor(scale: bool = False, use_score: bool = True) -> ColumnTransformer:
    """Armar el preprocesador compartido por todos los modelos.

    Parameters
    ----------
    scale : bool, default=False
        Estandarizar las ramas numérica, de target encoding y de frecuencia.
        True para modelos sensibles a escala, False para árboles.
    use_score : bool, default=True
        Incluir `score`. False arma el mismo pipeline sin esa columna, para
        medir cuánto depende el modelo de un dato cuya disponibilidad al momento
        de decidir el pago sigue sin confirmarse.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformador sin ajustar, para usar dentro de un Pipeline.

    Notes
    -----
    El target encoding y la frecuencia se ajustan solo con las filas que recibe
    en fit: dentro de un Pipeline, eso es el train de cada fold. Las one-hot no
    se escalan porque ya están en 0/1. El monto entra como feature pero nunca se
    modifica en el frame original, que es el que usa expected_gain.
    """

    def maybe_scale(steps: list) -> Pipeline:
        return Pipeline(steps + [("escalado", StandardScaler())] if scale else steps)

    numeric_columns = (
        NUMERIC_COLUMNS if use_score else [col for col in NUMERIC_COLUMNS if col != "score"]
    )
    numeric = maybe_scale(
        [
            ("imputacion", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    low_cardinality = Pipeline(
        [
            ("texto", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    high_cardinality = maybe_scale(
        [
            ("texto", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            # cv explícito en lugar de random_state, que sklearn 1.9 deprecó.
            (
                "target",
                TargetEncoder(
                    target_type="binary",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                ),
            ),
        ]
    )
    frequency = maybe_scale(
        [
            ("texto", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            ("frecuencia", FrequencyEncoder()),
        ]
    )
    time = FunctionTransformer(_time_features, feature_names_out=_time_feature_names)
    return ColumnTransformer(
        [
            ("numericas", numeric, numeric_columns),
            ("baja_cardinalidad", low_cardinality, LOW_CARDINALITY_COLUMNS),
            ("alta_cardinalidad", high_cardinality, HIGH_CARDINALITY_COLUMNS),
            ("frecuencia", frequency, FREQUENCY_COLUMNS),
            ("binarias", "passthrough", BINARY_COLUMNS),
            ("temporales", time, [DATE]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
