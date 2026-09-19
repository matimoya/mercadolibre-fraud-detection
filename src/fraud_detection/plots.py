"""Gráficos de apoyo para el análisis con etiquetas conocidas.

Las transformaciones aplicadas acá son visuales: no estandarizan el dataset,
no ajustan escaladores y no devuelven features.
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from fraud_detection.modeling import temporal_folds

PALETTE = {"No fraude": "#287C9E", "Fraude": "#D38B28"}
WEEKDAYS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


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


INK, MUTED = "#1F1F1F", "#4A4A4A"
FOLD_COLORS = {"train": PALETTE["No fraude"], "valid": PALETTE["Fraude"], "final": "#5E5E5E"}


def _day_range(days: pd.Series) -> str:
    """Primer y último día con su día de semana, como 'lun 16/03 – dom 22/03'."""
    first, last = days.min(), days.max()
    return f"{WEEKDAYS[first.dayofweek]} {first:%d/%m} – {WEEKDAYS[last.dayofweek]} {last:%d/%m}"


def _thousands(value: int) -> str:
    """Formato del informe: punto para los miles."""
    return f"{value:,}".replace(",", ".")


def _signed(value: float) -> str:
    """Dinero con signo explícito y punto para los miles: '+71.156', '−23.589'."""
    return f"{'−' if value < 0 else '+'}{_thousands(int(round(abs(value))))}"


def _style_axis(ax) -> None:
    """Grilla vertical tenue, sin marcas y sin bordes salvo el de abajo."""
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD")


def _draw_days(ax, row: int, start, days: int, fill: tuple[str, str], text: str) -> None:
    """Una barra de `days` días desde `start`, con su etiqueta adentro o al costado.

    `fill` es el color de la barra y el del texto cuando entra adentro.
    """
    color, text_color = fill
    left = mdates.date2num(start)
    ax.barh(row, days, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.5)
    fits = days >= 5
    ax.annotate(
        text,
        (left + (days / 2 if fits else days), row),
        xytext=(0 if fits else 5, 0),
        textcoords="offset points",
        ha="center" if fits else "left",
        va="center",
        fontsize=8.5,
        color=text_color if fits else INK,
        bbox=None if fits else {"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"},
    )


def _draw_fold(ax, row: int, day: pd.Series, train_pos, valid_pos) -> dict:
    """Entrenamiento, hueco y validación de un fold; devuelve su fila del calendario."""
    train_days, valid_days = day.iloc[train_pos], day.iloc[valid_pos]
    n_train, n_valid = train_days.nunique(), valid_days.nunique()
    _draw_days(
        ax,
        row,
        train_days.min(),
        n_train,
        (FOLD_COLORS["train"], "white"),
        f"{n_train} {'día' if n_train == 1 else 'días'} · {_thousands(len(train_pos))}",
    )
    hole_start = train_days.max() + pd.Timedelta(days=1)
    hole = (valid_days.min() - hole_start).days
    if hole > 0:
        ax.barh(
            row,
            hole,
            left=mdates.date2num(hole_start),
            height=0.62,
            color="white",
            edgecolor="#9A9A9A",
            hatch="////",
            linewidth=0.8,
        )
    _draw_days(
        ax,
        row,
        valid_days.min(),
        n_valid,
        (FOLD_COLORS["valid"], INK),
        f"{n_valid} días · {_thousands(len(valid_pos))}",
    )
    return {
        "entrena": _day_range(train_days),
        "dias_train": n_train,
        "filas_train": len(train_pos),
        "valida": _day_range(valid_days),
        "dias_valid": n_valid,
        "filas_valid": len(valid_pos),
    }


def _draw_fold_panel(ax, folds: list, day: pd.Series, reserved: pd.Series, with_final: bool):
    """Un panel: la franja reservada, cada fold y, si corresponde, el modelo final."""
    ax.axvspan(
        mdates.date2num(reserved.min()),
        mdates.date2num(reserved.max() + pd.Timedelta(days=1)),
        color="#F1F1F1",
        zorder=0,
    )
    calendar = [
        {"fold": f"fold {number}", **_draw_fold(ax, -number, day, train_pos, valid_pos)}
        for number, (train_pos, valid_pos) in enumerate(folds)
    ]
    labels = [fold["fold"] for fold in calendar]
    if with_final:
        row = -len(folds)
        _draw_days(
            ax,
            row,
            day.min(),
            day.nunique(),
            (FOLD_COLORS["train"], "white"),
            f"{day.nunique()} días · {_thousands(len(day))}",
        )
        _draw_days(
            ax,
            row,
            reserved.min(),
            reserved.nunique(),
            (FOLD_COLORS["final"], "white"),
            f"{reserved.nunique()} días · {_thousands(len(reserved))}",
        )
        labels.append("modelo final")
    ax.set_yticks(range(0, -len(labels), -1), labels)
    return calendar


def plot_temporal_folds(
    dates: pd.Series, reserved_dates: pd.Series, gap_days: tuple[int, ...] = (0, 7)
):
    """Dibujar qué días entrena y qué días valida cada fold, un panel por gap.

    Parameters
    ----------
    dates : pandas.Series
        Fechas de desarrollo, las mismas que recibe temporal_folds.
    reserved_dates : pandas.Series
        Fechas del período reservado, dibujado como franja aparte.
    gap_days : tuple of int, default=(0, 7)
        Un panel por valor. El primero suma la fila del modelo final.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Entrenamiento, hueco y validación de cada fold sobre el calendario.
    calendar : pandas.DataFrame
        Días y filas de cada fold con el primer valor de gap_days.
    """
    day = pd.to_datetime(dates).dt.normalize()
    reserved = pd.to_datetime(reserved_dates).dt.normalize()
    panels = [list(temporal_folds(dates, gap_days=gap)) for gap in gap_days]
    rows_per_panel = [len(folds) + (1 if number == 0 else 0) for number, folds in enumerate(panels)]
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(11, 1.5 + 0.5 * sum(rows_per_panel)),
        gridspec_kw={"height_ratios": rows_per_panel},
        squeeze=False,
    )

    calendar: list[dict] = []
    for ax, gap, folds in zip(axes[:, 0], gap_days, panels):
        rows = _draw_fold_panel(ax, folds, day, reserved, with_final=gap == gap_days[0])
        if gap == gap_days[0]:
            calendar = rows
        ax.set_title(
            (
                "Sin hueco (gap = 0): lo que se usa en todo el trabajo"
                if gap == 0
                else f"Con hueco (gap = {gap}): si las etiquetas tardaran {gap} días en confirmarse"
            ),
            fontsize=10,
            loc="left",
            color=INK,
        )
        _style_axis(ax)

    bottom = axes[-1, 0]
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("lun %d/%m"))
    bottom.set_xlim(
        mdates.date2num(day.min()), mdates.date2num(reserved.max() + pd.Timedelta(days=1))
    )
    axes[0, 0].annotate(
        "período reservado",
        (mdates.date2num(reserved.min() + (reserved.max() - reserved.min()) / 2), 0.4),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(color=FOLD_COLORS["train"], label="Entrena"),
            Patch(color=FOLD_COLORS["valid"], label="Valida"),
            Patch(facecolor="white", edgecolor="#9A9A9A", hatch="////", label="Hueco sin usar"),
            Patch(color=FOLD_COLORS["final"], label="Evaluación final, una sola vez"),
        ],
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Validación temporal: días y filas de cada fold", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, pd.DataFrame(calendar).set_index("fold")


def _draw_balance_row(ax, row: int, table: pd.DataFrame, pad: float) -> None:
    """Costo de las legítimas a la izquierda, fraude frenado a la derecha y el neto marcado."""
    cost, saved, net = table.loc[["legitima rechazada", "fraude rechazado", "total"], "efecto"]
    legit, frauds = table.loc[["legitima rechazada", "fraude rechazado"], "transacciones"]
    for value, color in [(cost, PALETTE["No fraude"]), (saved, PALETTE["Fraude"])]:
        ax.barh(row, value, height=0.5, color=color, edgecolor="white", linewidth=1.5)
    ax.text(
        cost - pad,
        row,
        f"{_signed(cost)} · {_thousands(int(legit))} legítimas",
        ha="right",
        va="center",
        fontsize=9,
        color=INK,
    )
    ax.text(
        saved + pad,
        row,
        f"{_signed(saved)} · {_thousands(int(frauds))} fraudes",
        ha="left",
        va="center",
        fontsize=9,
        color=INK,
    )
    ax.plot([net, net], [row - 0.34, row + 0.34], color=INK, linewidth=2.2, solid_capstyle="butt")
    ax.text(
        net,
        row + 0.38,
        f"neto {_signed(net)}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
        fontweight="bold",
    )


def plot_rejection_balance(cells: dict[str, pd.DataFrame]):
    """Dibujar, por política, lo que cuestan las legítimas rechazadas y lo que salvan los fraudes.

    Parameters
    ----------
    cells : dict of str to pandas.DataFrame
        Nombre de cada política y sus celdas de la matriz de confusión, con
        filas "legitima rechazada", "fraude rechazado" y "total", y columnas
        transacciones y efecto (dinero frente a aprobar todo).

    Returns
    -------
    matplotlib.figure.Figure
        Una fila por política, en el orden recibido de arriba hacia abajo.

    Notes
    -----
    Las aprobadas no aparecen: frente a aprobar todo no cambian nada, así que
    el neto de cada fila es la ganancia de la política sobre aprobar todo.
    """
    effects = pd.concat(
        [
            table.loc[["legitima rechazada", "fraude rechazado"], "efecto"]
            for table in cells.values()
        ]
    )
    low, high = effects.min(), effects.max()
    fig, ax = plt.subplots(figsize=(10, 1.4 + 1.1 * len(cells)))
    for row, table in enumerate(cells.values()):
        _draw_balance_row(ax, -row, table, pad=0.012 * (high - low))

    ax.set_xlim(low - 0.55 * (high - low), high + 0.42 * (high - low))
    ax.set_ylim(-len(cells) + 0.45, 0.85)
    ax.axvline(0, color="#7A7A7A", linewidth=1)
    ax.set_yticks(range(0, -len(cells), -1), list(cells))
    ax.xaxis.set_major_formatter(
        lambda value, _: f"{'−' if value < 0 else ''}{_thousands(int(abs(value)))}"
    )
    ax.set_xlabel("Dinero frente a aprobar todo", color=MUTED, fontsize=9)
    _style_axis(ax)
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    fig.legend(
        handles=[
            Patch(
                color=PALETTE["No fraude"], label="Legítimas rechazadas: se deja de ganar el 25%"
            ),
            Patch(color=PALETTE["Fraude"], label="Fraudes frenados: se salva el monto entero"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.005, 0.93),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Qué gana y qué cuesta rechazar", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig
