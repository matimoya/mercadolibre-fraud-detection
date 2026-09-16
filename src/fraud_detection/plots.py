"""Gráficos de apoyo para el análisis con etiquetas conocidas.

Las transformaciones aplicadas acá son visuales: no estandarizan el dataset,
no ajustan escaladores y no devuelven features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {"No fraude": "#287C9E", "Fraude": "#D38B28"}


def plot_class_distribution(
    frame: pd.DataFrame, column: str, log_scale: bool = False, target: str = "fraude"
):
    """Comparar una variable continua por clase con KDE y resumen original.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datos con objetivo binario 0/1.
    column : str
        Variable numérica continua.
    log_scale : bool, default=False
        Graficar log1p del valor; requiere observaciones no negativas.
        El resumen conserva las unidades originales.
    target : str, default="fraude"
        Columna de la etiqueta.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Curvas por clase y tabla de estadísticos.
    summary : pandas.DataFrame
        Conteos, nulos y estadísticos en unidades originales por clase.

    Raises
    ------
    ValueError
        Si faltan clases, hay etiquetas inválidas, infinitos, valores negativos
        con log_scale o alguna clase no tiene variación suficiente para KDE.

    Notes
    -----
    Cada KDE se normaliza por clase (common_norm=False): la densidad no
    representa P(fraude | valor) ni el volumen relativo de clases. Los nulos
    se excluyen de las curvas, se cuentan en la tabla y siguen en el dataset.
    """
    if not frame[target].isin([0, 1]).all() or set(frame[target].unique()) != {0, 1}:
        raise ValueError("Se requieren ambas clases, 0 y 1, sin etiquetas inválidas.")
    observed = frame[column].dropna()
    if not np.isfinite(observed).all():
        raise ValueError("Revisar valores infinitos antes de graficar.")
    if log_scale and observed.lt(0).any():
        raise ValueError("log1p en este gráfico requiere valores no negativos.")
    grouped = frame.groupby(target)[column]
    summary = grouped.agg(
        total="size",
        observados="count",
        media="mean",
        mediana="median",
        desviacion="std",
        minimo="min",
        maximo="max",
    ).T
    summary.loc["nulos"] = summary.loc["total"] - summary.loc["observados"]
    summary.loc["p25"] = grouped.quantile(0.25)
    summary.loc["p75"] = grouped.quantile(0.75)
    summary = summary.loc[
        [
            "total",
            "observados",
            "nulos",
            "media",
            "desviacion",
            "minimo",
            "p25",
            "mediana",
            "p75",
            "maximo",
        ]
    ]
    summary.columns = ["No fraude", "Fraude"]
    plot_data = frame[[column, target]].dropna().copy()
    if plot_data.groupby(target)[column].nunique().min() < 2:
        raise ValueError("KDE requiere variación en ambas clases; usar frecuencias.")
    plot_data["clase"] = plot_data[target].map({0: "No fraude", 1: "Fraude"})
    plot_data["valor"] = np.log1p(plot_data[column]) if log_scale else plot_data[column]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [2, 1]})
    sns.kdeplot(
        data=plot_data,
        x="valor",
        hue="clase",
        hue_order=["No fraude", "Fraude"],
        palette=PALETTE,
        common_norm=False,
        cut=0,
        linewidth=2.3,
        ax=axes[0],
    )
    axes[0].set(
        title=f"{column}: distribución por clase",
        xlabel=f"log(1 + {column})" if log_scale else column,
        ylabel="Densidad (área 1 por clase)",
    )
    axes[1].axis("off")
    values = [
        [
            f"{valor:,.0f}" if name in {"total", "observados", "nulos"} else f"{valor:,.3f}"
            for valor in row
        ]
        for name, row in summary.iterrows()
    ]
    table = axes[1].table(
        cellText=values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="right",
        loc="center",
        colWidths=[0.44, 0.44],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1].set_title("Resumen en unidades originales", fontsize=11)
    fig.tight_layout(w_pad=3)
    return fig, summary


def plot_gain_curve(
    curves: dict[str, pd.Series],
    baseline: float | None = None,
    title: str = "",
    xlabel: str = "Corte",
):
    """Graficar la ganancia que deja cada corte, con su máximo marcado.

    Parameters
    ----------
    curves : dict of str to pandas.Series
        Nombre de cada curva y su salida de evaluation.sweep_gain.
    baseline : float, optional
        Ganancia de referencia, típicamente la de aprobar todo, dibujada
        como línea horizontal.
    title : str, default=""
        Título del gráfico.
    xlabel : str, default="Corte"
        Nombre del eje de cortes.

    Returns
    -------
    matplotlib.figure.Figure
        Una línea por curva y un punto en el corte de mayor ganancia.

    Notes
    -----
    El punto marcado es el mejor corte para esos datos. Si cada curva viene
    de un período distinto y los puntos no coinciden, el mejor corte no es
    estable y elegirlo sobre todo el histórico sobrestima la ganancia.
    """
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for position, (name, curve) in enumerate(curves.items()):
        color = PALETTE["Fraude"] if position == 0 else None
        line = ax.plot(
            curve.index,
            curve.to_numpy(),
            linewidth=2.3 if position == 0 else 1.4,
            color=color,
            label=name,
        )[0]
        best = curve.idxmax()
        ax.plot(
            best,
            curve.loc[best],
            marker="o",
            markersize=7 if position == 0 else 5,
            color=line.get_color(),
            linestyle="none",
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#7A7A7A",
            linestyle="--",
            linewidth=1.2,
            label=f"Aprobar todo: {baseline:,.0f}",
        )
    ax.set(title=title, xlabel=xlabel, ylabel="Ganancia")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_rate_table(table: pd.DataFrame, title: str = "", ylabel: str = "Grupo"):
    """Graficar volumen en barras y tasa de fraude en línea sobre eje derecho.

    Parameters
    ----------
    table : pandas.DataFrame
        Salida de diagnostics.rate_table, con índice simple.
    title : str, default=""
        Título del gráfico.
    ylabel : str, default="Grupo"
        Nombre del eje de categorías.

    Returns
    -------
    matplotlib.figure.Figure
        Barras de transacciones y línea de fraude_pct con la tasa base.

    Notes
    -----
    La tasa base se recalcula sobre la tabla, por lo que asume que los grupos
    cubren el conjunto completo. Los grupos con poco volumen tienen tasas
    imprecisas: leer siempre la barra junto a la línea.
    """
    base_rate = 100 * table["fraudes"].sum() / table["transacciones"].sum()
    labels = table.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.bar(labels, table["transacciones"], color="#C9D9E2", label="Transacciones")
    ax.set(xlabel=ylabel, ylabel="Transacciones", title=title)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    rate_axis = ax.twinx()
    rate_axis.plot(
        labels,
        table["fraude_pct"],
        color=PALETTE["Fraude"],
        marker="o",
        linewidth=2.3,
        label="Fraude (%)",
    )
    rate_axis.axhline(
        base_rate,
        color="#7A7A7A",
        linestyle="--",
        linewidth=1.2,
        label=f"Tasa base {base_rate:.2f}%",
    )
    rate_axis.set_ylabel("Transacciones fraudulentas (%)")
    rate_axis.set_ylim(bottom=0)
    handles = ax.get_legend_handles_labels()[0] + rate_axis.get_legend_handles_labels()[0]
    labels_legend = ax.get_legend_handles_labels()[1] + rate_axis.get_legend_handles_labels()[1]
    ax.legend(handles, labels_legend, loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig
