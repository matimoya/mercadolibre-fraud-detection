"""Diagnóstico de segmentos y calidad de datos sobre etiquetas conocidas.

No es un pipeline de preprocesamiento: no se ejecuta para predecir pagos
nuevos. Imputadores y escaladores del modelo se ajustan solo en el train de
cada fold. El monto siempre permanece en unidades originales.
"""

import numpy as np
import pandas as pd


def check_labels(frame: pd.DataFrame, target: str = "fraude") -> None:
    """Validar que la etiqueta sea binaria 0/1 sin nulos.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos a validar.
    target : str, default="fraude"
        Columna de la etiqueta.

    Raises
    ------
    ValueError
        Si la columna contiene nulos o valores fuera de {0, 1}.
    """
    if not frame[target].isin([0, 1]).all():
        raise ValueError(f"'{target}' debe ser binaria 0/1 y no contener nulos.")


def _grouping_keys(frame: pd.DataFrame, by, bins, missing_label: str) -> list[pd.Series]:
    """Normalizar by a una lista de series alineadas con frame."""
    if isinstance(by, pd.Series):
        if not by.index.equals(frame.index):
            raise ValueError("La serie de agrupación debe estar alineada con frame.")
        columns = [by if by.name is not None else by.rename("grupo")]
    else:
        names = [by] if isinstance(by, str) else list(by)
        columns = [frame[name] for name in names]
    if bins is not None:
        if len(columns) != 1:
            raise ValueError("bins requiere una única variable de agrupación.")
        columns = [pd.cut(columns[0], bins, include_lowest=True).rename(columns[0].name)]
    return [_label_missing(values, missing_label) for values in columns]


def _label_missing(values: pd.Series, missing_label: str) -> pd.Series:
    """Dar un grupo propio a los nulos, salvo en columnas numéricas."""
    if not values.isna().any():
        return values
    if isinstance(values.dtype, pd.CategoricalDtype):
        return values.cat.add_categories(missing_label).fillna(missing_label)
    if pd.api.types.is_numeric_dtype(values):
        return values
    return values.astype(object).fillna(missing_label)


def rate_table(
    frame: pd.DataFrame,
    by,
    bins: list | None = None,
    target: str = "fraude",
    amount: str = "monto",
    missing_label: str = "Ausente",
) -> pd.DataFrame:
    """Por grupo: cuántas transacciones, cuánto fraude y cuánta plata hay.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con etiqueta binaria y monto en unidades originales.
    by : str, list of str or pandas.Series
        Agrupación. Una serie alineada permite agrupar por máscara booleana.
    bins : list, optional
        Cortes para discretizar una continua antes de agrupar. Incluye el
        borde inferior; los valores fuera de rango caen en missing_label.
    target, amount : str
        Columnas de etiqueta y monto.
    missing_label : str, default="Ausente"
        Etiqueta del grupo de nulos, que nunca se descarta.

    Returns
    -------
    pandas.DataFrame
        transacciones, volumen_pct y fraudes_pct (participación en el total),
        fraude_pct (tasa del grupo), lift sobre la tasa base, monto_total y
        monto_en_fraude.

    Notes
    -----
    Los nulos forman su propio grupo: donde la ausencia es informativa,
    descartarlos borraría ese patrón. monto_en_fraude es exposición si se
    aprobara el segmento, no una pérdida observada.
    """
    check_labels(frame, target)
    keys = _grouping_keys(frame, by, bins, missing_label)
    work = frame[[target, amount]].copy()
    work["monto_en_fraude"] = work[amount].where(work[target].eq(1), 0)
    result = work.groupby(keys, dropna=False, observed=True).agg(
        transacciones=(target, "size"),
        fraudes=(target, "sum"),
        fraude_pct=(target, "mean"),
        monto_total=(amount, "sum"),
        monto_en_fraude=("monto_en_fraude", "sum"),
    )
    result["fraude_pct"] *= 100
    result["volumen_pct"] = 100 * result["transacciones"] / len(frame)
    total_fraud = frame[target].sum()
    result["fraudes_pct"] = 100 * result["fraudes"] / total_fraud if total_fraud else np.nan
    base_rate = 100 * frame[target].mean()
    result["lift"] = result["fraude_pct"] / base_rate if base_rate else np.nan
    return result[
        [
            "transacciones",
            "volumen_pct",
            "fraudes",
            "fraudes_pct",
            "fraude_pct",
            "lift",
            "monto_total",
            "monto_en_fraude",
        ]
    ]


def summarize_segments(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Resumir volumen, fraude y cobertura temporal por segmento.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con fecha, fraude binario y monto original.
    by : list of str
        Columnas de agrupación, por ejemplo país y período.

    Returns
    -------
    pandas.DataFrame
        transacciones, fraudes, fraude_pct, monto_mediano, desde, hasta y
        dias_observados por segmento.

    Notes
    -----
    Los días observados cuentan días con transacciones y no demuestran
    cobertura completa de la fuente. Sin dimensión temporal, usar rate_table.
    """
    work = frame.assign(dia=frame.fecha.dt.normalize())
    result = work.groupby(by, dropna=False, observed=True).agg(
        transacciones=("fraude", "size"),
        fraudes=("fraude", "sum"),
        fraude_pct=("fraude", "mean"),
        monto_mediano=("monto", "median"),
        desde=("dia", "min"),
        hasta=("dia", "max"),
        dias_observados=("dia", "nunique"),
    )
    result["fraude_pct"] *= 100
    return result


def iqr_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Marcar extremos con cercas de Tukey, sin eliminar observaciones.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos de entrenamiento con etiqueta binaria.
    columns : list of str
        Variables numéricas; excluir objetivo, indicadores e identificadores.

    Returns
    -------
    pandas.DataFrame
        Por variable: cuantiles, límites Q1 - 1.5 * IQR y Q3 + 1.5 * IQR,
        marcados, pct_observados (sobre valores finitos) y fraudes_marcados.
        aplicable=False significa que no hubo alertas, no que la columna sea
        válida.

    Notes
    -----
    Recalcula límites sobre el frame recibido: usar desarrollo, nunca test.
    En colas largas una alerta estadística no demuestra un error.
    """
    rows = []
    for col in columns:
        values = frame[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        applicable = pd.notna(iqr) and iqr > 0
        flags = (
            (frame[col].lt(low) | frame[col].gt(high)).fillna(False)
            if applicable
            else pd.Series(False, index=frame.index)
        )
        rows.append(
            {
                "variable": col,
                "min": values.min(),
                "mediana": values.median(),
                "p99": values.quantile(0.99),
                "max": values.max(),
                "iqr": iqr,
                "limite_inferior": low,
                "limite_superior": high,
                "aplicable": applicable,
                "marcados": int(flags.sum()),
                "pct_observados": 100 * flags.sum() / len(values) if len(values) else np.nan,
                "fraudes_marcados": int(frame.loc[flags, "fraude"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("variable")
