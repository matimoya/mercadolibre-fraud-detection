"""Extrae a PNG las figuras embebidas en los notebooks ya ejecutados.

Las figuras del informe salen de acá y no de volver a graficar, para que sean
exactamente las que produjo el código que se entrega. Ejecutar desde la raíz:
`uv run python informe/extraer_figuras.py`.
"""

import base64
import json
from pathlib import Path

NOTEBOOKS = Path("notebooks")
DESTINO = Path("informe/figuras")

# Nombre del archivo por (notebook, id de celda, posición de la figura en esa celda).
# El id y no el número de celda: el número cambia cada vez que se agrega o se mueve
# una celda, y entonces el informe muestra la figura equivocada sin que nada falle.
NOMBRES = {
    ("01_eda", "3f9d6e02", 0): "fraude_por_dia",
    ("01_eda", "9e41f7d4", 0): "folds_temporales",
    ("01_eda", "e08ec84c", 0): "monto_por_clase",
    ("01_eda", "f8d9783b", 0): "fraude_por_tramo_de_score",
    ("01_eda", "f9f2f9ef", 0): "ganancia_por_corte_de_score",
    ("01_eda", "76f2d7e3", 0): "correlaciones",
    ("01_eda", "2236e4a9", 0): "fraude_semanal_por_pais",
    ("03_modeling", "4c94df36", 0): "ganancia_y_recall_por_pais",
    ("03_modeling", "65a2151b", 0): "ganancia_por_umbral",
    ("04_tuning", "c795bdbf", 0): "brecha_de_generalizacion",
    ("05_evaluation", "6b0cc508", 0): "ganancia_y_costo_de_rechazar",
    ("05_evaluation", "c2e4a188", 0): "aporte_por_dia",
    ("05_evaluation", "897be9fc", 0): "importancia_shap",
}


def figuras_disponibles() -> dict[tuple[str, str, int], str]:
    """Todas las figuras de los notebooks, indexadas por dónde aparecen."""
    encontradas = {}
    for ruta in sorted(NOTEBOOKS.glob("*.ipynb")):
        for celda in json.loads(ruta.read_text())["cells"]:
            imagenes = [
                salida["data"]["image/png"]
                for salida in celda.get("outputs", [])
                if "image/png" in salida.get("data", {})
            ]
            for posicion, imagen in enumerate(imagenes):
                encontradas[(ruta.stem, celda["id"], posicion)] = imagen
    return encontradas


def extraer() -> None:
    """Guardar las figuras con nombre y reportar las que no calzan."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    disponibles = figuras_disponibles()
    for clave, imagen in disponibles.items():
        if clave in NOMBRES:
            archivo = DESTINO / f"{NOMBRES[clave]}.png"
            archivo.write_bytes(base64.b64decode(imagen))
            print(f"{archivo}  <-  {clave[0]}, celda {clave[1]}")

    sin_nombre = sorted(set(disponibles) - set(NOMBRES))
    faltantes = sorted(set(NOMBRES) - set(disponibles))
    if sin_nombre:
        print("\nFiguras del notebook sin nombre asignado:", sin_nombre)
    if faltantes:
        print("\nNombres que no encontraron figura:", faltantes)


if __name__ == "__main__":
    extraer()
