"""Comandos para correr el pipeline desde la terminal.

Cada etapa lee de disco lo que dejó la anterior y escribe su propia salida:
`tune` la configuración, `train` el modelo y `evaluate` la medición final.
Separadas, reentrenar con datos nuevos no obliga a repetir la búsqueda.
"""

import argparse
import logging

import pandas as pd
from xgboost import XGBClassifier

from fraud_detection import training, tuning
from fraud_detection.constants import DATE, EXPERIMENTO_BUSQUEDA, N_TRIALS, TARGET
from fraud_detection.dataset import load_periods
from fraud_detection.evaluation import approve_all, gain_by_policy
from fraud_detection.logs import registrar_corrida
from fraud_detection.modeling import temporal_folds
from fraud_detection.paths import BEST_PARAMS_JSON, EVALUATION_CSV, desde_raiz
from fraud_detection.tracking import configurar

logger = logging.getLogger(__name__)


def tune(trials: int = N_TRIALS) -> None:
    """Buscar hiperparámetros y dejar la configuración en disco."""
    development, _ = load_periods()
    bloques = list(temporal_folds(development[DATE]))

    configurar(EXPERIMENTO_BUSQUEDA)
    estudio, ganancia_defecto = tuning.search_and_record(development, bloques, n_trials=trials)
    tuning.save_best(estudio, ganancia_validacion_defecto=ganancia_defecto)

    logger.info("%d pruebas, %d podadas", len(estudio.trials), tuning.pruned_trials(estudio))
    logger.info(
        "mejor ganancia en validación: %+.0f (%+.0f sobre la configuración por defecto)",
        estudio.best_value,
        estudio.best_value - ganancia_defecto,
    )


def train() -> None:
    """Entrenar con todo el desarrollo y guardar el artefacto."""
    development, _ = load_periods()
    parametros = tuning.load_best()["parametros"]

    modelo = training.train(development, XGBClassifier(**parametros))
    training.save_model(modelo, development)


def evaluate() -> None:
    """Medir el artefacto guardado sobre el período reservado."""
    _, test = load_periods()
    artefacto = training.load_model()

    probabilidad = artefacto["pipeline"].predict_proba(test.drop(columns=[TARGET]))[:, 1]
    politicas = {
        "aprobar todo": approve_all(test),
        "modelo": pd.Series(probabilidad, index=test.index).lt(artefacto["umbral"]),
    }
    tabla = gain_by_policy(test, politicas)
    tabla["sobre_aprobar_todo"] = tabla.ganancia - tabla.loc["aprobar todo", "ganancia"]

    EVALUATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(EVALUATION_CSV, index_label="politica")
    logger.info("evaluación guardada en %s", desde_raiz(EVALUATION_CSV))
    # La tabla es el resultado del comando: va por stdout, y los logs por stderr.
    print(tabla.round(2).to_string())


def run_all(trials: int = N_TRIALS) -> None:
    """Correr las tres etapas en orden: buscar, entrenar y evaluar."""
    tune(trials)
    train()
    evaluate()


def construir_parser() -> argparse.ArgumentParser:
    """Un subcomando por función, guardando la función misma como valor."""
    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument(
        "-v", "--verbose", action="store_true", help="mostrar también el detalle por bloque"
    )

    parser = argparse.ArgumentParser(prog="fraud-detection", description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)

    buscar = comandos.add_parser("tune", parents=[comunes], help=tune.__doc__)
    buscar.add_argument("--trials", type=int, default=N_TRIALS)
    buscar.set_defaults(funcion=tune)

    entrenar = comandos.add_parser(
        "train", parents=[comunes], help=f"entrenar con {BEST_PARAMS_JSON.name}"
    )
    entrenar.set_defaults(funcion=train)

    medir = comandos.add_parser("evaluate", parents=[comunes], help=evaluate.__doc__)
    medir.set_defaults(funcion=evaluate)

    todo = comandos.add_parser("all", parents=[comunes], help=run_all.__doc__)
    todo.add_argument("--trials", type=int, default=N_TRIALS)
    todo.set_defaults(funcion=run_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de `fraud-detection`."""
    argumentos = vars(construir_parser().parse_args(argv))
    funcion, comando = argumentos.pop("funcion"), argumentos.pop("comando")
    try:
        with registrar_corrida(comando, argumentos.pop("verbose")):
            logger.info("argumentos: %s", argumentos)
            funcion(**argumentos)
    except Exception:  # pylint: disable=broad-exception-caught
        # El traceback ya quedó en consola y en el archivo: no repetirlo al salir.
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
