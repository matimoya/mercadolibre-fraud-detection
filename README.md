# Mercado Libre Fraud Detection Challenge

## Ejecución

Usar Python 3.11 y uv desde la raíz del proyecto:

```bash
uv sync --locked
uv run --locked jupyter lab
```

En macOS, XGBoost necesita además el runtime de OpenMP, que sus ruedas no incluyen:

```bash
brew install libomp
```

El CSV del enunciado debe ubicarse en:
`data/MercadoLibre Data Scientist Technical Challenge - Dataset.csv`.
La carpeta `data/` está excluida de Git. La carga usa una ruta relativa sencilla;
el directorio de trabajo del kernel debe ser `notebooks/`.
En VS Code seleccionar `.venv/bin/python` como entorno del notebook.

## Notebooks

Se ejecutan en orden, cada uno con un kernel nuevo:

| Notebook | Contenido |
| --- | --- |
| `01_eda.ipynb` | Calidad, economía, señales y estabilidad temporal. |
| `02_feature_engineering.ipynb` | Transformaciones y controles de leakage. |
| `03_modeling.ipynb` | Modelos, calibración, desbalanceo y validación. |
| `04_tuning.ipynb` | Optuna, evaluación final, SHAP y persistencia. |

El EDA reserva como test el período desde el 13 de abril de 2020 y no lo usa para
ninguna decisión. Cada variable se mide en ganancia y no solo en tasa de fraude:
la vara de `score` se calcula eligiendo el corte con el pasado de cada fold y
midiéndolo en su futuro, y cada señal se reporta con su exposición en monto y con
la ganancia que produciría rechazar ese segmento. Evalúa con los datos las dos
preguntas abiertas del enunciado: si `score` es una fuga de información y si las
etiquetas tienen demora de confirmación.

Sobre el período reservado, el modelo elegido —XGBoost con umbral 0,2— rinde
**+47.567 sobre aprobar todo**, contra **+23.201** de la mejor regla de un corte
sobre `score`: captura el 40% de la ganancia que estaba en disputa y deja pasar
803 fraudes contra 1.571. Sin la columna `score`, cuya disponibilidad al decidir
el pago sigue sin confirmarse, rinde **+37.867**, todavía 1,6 veces la regla
actual.

Cada decisión de modelado se toma midiendo y se deja el resultado negativo
escrito: las cuatro técnicas de manejo del desbalanceo empeoran la ganancia, las
dos calibraciones también, Isolation Forest no aporta señal, y del ajuste de
hiperparámetros —+5.127 en validación— sobrevivieron **+164** sobre datos nuevos.

**No se persiste ningún dataset procesado.** El target encoding de las categóricas
de alta cardinalidad se ajusta con el train de cada fold, dentro del Pipeline. El
único artefacto que se guarda es el Pipeline entrenado —`models/`, excluida de Git—,
nunca la tabla de features; `02_feature_engineering.ipynb` verifica esa separación
con cuatro chequeos y `04_tuning.ipynb` comprueba que el artefacto recuperado
reproduce las mismas probabilidades.
`RANDOM_STATE` está fijado en `constants.py` para que los números sean
reproducibles entre corridas.

## Estructura del paquete

| Módulo | Contenido |
| --- | --- |
| `constants.py` | Contrato de datos y reglas de negocio. |
| `dataset.py` | Carga, normalización y reserva temporal. |
| `diagnostics.py` | Diagnósticos de riesgo y exposición. |
| `plots.py` | Gráficos de distribución, riesgo y ganancia. |
| `evaluation.py` | Ganancia, umbrales y barridos de cortes. |
| `features.py` | Preprocesamiento y `FrequencyEncoder`. |
| `modeling.py` | Folds temporales sobre días calendario. |

`diagnostics.py` y `plots.py` son herramientas de análisis e informes, no
transformadores de producción; `features.py` sí produce las entradas del modelo.
Los docstrings siguen el formato NumPy.

Ejecución reproducible sin interfaz:

```bash
uv run --locked jupyter execute notebooks/01_eda.ipynb --inplace
uv run --locked jupyter execute notebooks/02_feature_engineering.ipynb --inplace
uv run --locked jupyter execute notebooks/03_modeling.ipynb --inplace
uv run --locked jupyter execute notebooks/04_tuning.ipynb --inplace
```

Las dos últimas tardan alrededor de dos minutos cada una; las dos primeras, segundos.

## Controles de calidad

Instalar los hooks de Git una vez preparado el entorno:

```bash
uv run pre-commit install
```

Ejecutar todos los controles manualmente:

```bash
uv run pre-commit run --all-files
```

Black formatea el código Python y las celdas de los notebooks. Pylint y mypy se
aplican a `src/`; markdownlint revisa los archivos Markdown.
