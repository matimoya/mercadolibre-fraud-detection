# Mercado Libre Fraud Detection Challenge

## Ejecución con Docker

Docker es la forma recomendada de ejecutar el proyecto. Desde la raíz, construir
la imagen una vez:

```shell
docker build -t meli-fraud .
```

El CSV del enunciado debe ubicarse en:
`data/MercadoLibre Data Scientist Technical Challenge - Dataset.csv`.
La carpeta `data/` está excluida de Git y de la imagen Docker.

Iniciar Jupyter Lab con:

```shell
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  meli-fraud
```

El montaje `-v "$PWD:/workspace"` comparte el proyecto local con el contenedor.
Así puede leer el CSV y conservar notebooks, modelos y resultados generados sin
incorporarlos a la imagen. Abrir la URL con token que Jupyter muestra en la
terminal, normalmente sobre [http://localhost:8888](http://localhost:8888).

## Notebooks

Se ejecutan en orden, cada uno con un kernel nuevo:

| Notebook | Contenido |
| --- | --- |
| `01_eda.ipynb` | Calidad, economía, señales y estabilidad temporal. |
| `02_feature_engineering.ipynb` | Transformaciones y controles de leakage. |
| `03_modeling.ipynb` | Modelos, calibración, desbalanceo y validación. |
| `04_tuning.ipynb` | Optuna sin usar el período reservado. |
| `05_evaluation.ipynb` | Evaluación final, SHAP, persistencia y monitoreo. |

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
con cuatro chequeos y `05_evaluation.ipynb` comprueba que el artefacto recuperado
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
| `tracking.py` | Registro de experimentos en MLflow. |

`diagnostics.py` y `plots.py` son herramientas de análisis e informes, no
transformadores de producción; `features.py` sí produce las entradas del modelo.
Los docstrings siguen el formato NumPy.

## Ejecución completa

Ejecutar los cinco notebooks, en orden y sin abrir la interfaz:

```shell
docker run --rm -v "$PWD:/workspace" meli-fraud sh -c '
  set -e
  uv run --locked jupyter execute notebooks/01_eda.ipynb --inplace
  uv run --locked jupyter execute notebooks/02_feature_engineering.ipynb --inplace
  uv run --locked jupyter execute notebooks/03_modeling.ipynb --inplace
  uv run --locked jupyter execute notebooks/04_tuning.ipynb --inplace
  uv run --locked jupyter execute notebooks/05_evaluation.ipynb --inplace
'
```

`03` y `04` tardan alrededor de dos minutos cada una; el resto, segundos.

## Experimentos con MLflow

Las corridas de `03`, `04` y `05` quedan registradas en MLflow, con store SQLite
en `mlflow.db` y artefactos en `mlartifacts/`, los dos excluidos de Git porque se
regeneran al ejecutar los notebooks. Después de ejecutar los notebooks, iniciar
la interfaz con:

```shell
docker run --rm -it \
  -p 5001:5000 \
  -v "$PWD:/workspace" \
  meli-fraud \
  uv run --locked mlflow server \
    --backend-store-uri sqlite:////workspace/mlflow.db \
    --host 0.0.0.0 \
    --port 5000
```

Abrir [http://localhost:5001](http://localhost:5001). Desde **Experiments** se
pueden abrir las corridas, comparar parámetros y métricas, y revisar sus
etiquetas y artefactos. La documentación oficial describe estas vistas en
[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).

| Experimento | Qué contiene |
| --- | --- |
| `03-comparacion-modelos` | Modelos con/sin `score` y desbalanceo. |
| `04-busqueda-hiperparametros` | Búsqueda y pruebas de Optuna. |
| `05-evaluacion-final` | Evaluación reservada y modelo final. |

`04_tuning.ipynb` deja su configuración en `models/mejores_parametros.json`, que
es lo que `05_evaluation.ipynb` carga para entrenar el modelo final. Esa
separación es lo que hace que el período reservado se use una sola vez.

## Controles de calidad

Ejecutar todos los hooks dentro de la imagen:

```shell
docker run --rm -v "$PWD:/workspace" meli-fraud \
  uv run --locked pre-commit run --all-files
```

Pylint y mypy también pueden comprobarse individualmente:

```shell
docker run --rm -v "$PWD:/workspace" meli-fraud \
  uv run --locked pylint src/fraud_detection
docker run --rm -v "$PWD:/workspace" meli-fraud \
  uv run --locked mypy src/fraud_detection
```

Black formatea el código Python y las celdas de los notebooks. Pylint y mypy se
aplican a `src/`; markdownlint revisa los archivos Markdown.

## Ejecución local alternativa

Docker evita instalar Python y OpenMP directamente. Si se prefiere trabajar sin
contenedor, usar Python 3.11 y `uv` desde la raíz:

```shell
uv sync --locked
uv run --locked jupyter lab
```

En macOS, XGBoost necesita además el runtime de OpenMP:

```shell
brew install libomp
```

La carga usa una ruta relativa; el directorio de trabajo del kernel debe ser
`notebooks/`. En VS Code se puede seleccionar `.venv/bin/python` como entorno.
Para instalar y ejecutar los hooks localmente:

```shell
uv run pre-commit install
uv run pre-commit run --all-files
```
