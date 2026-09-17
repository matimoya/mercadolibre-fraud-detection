"""Comandos para correr el pipeline desde la terminal."""

import argparse
import logging

import pandas as pd
from xgboost import XGBClassifier

from fraud_detection import training, tuning
from fraud_detection.constants import DATE, N_TRIALS, TARGET
from fraud_detection.dataset import load_transactions, normalize_text, split_development
from fraud_detection.evaluation import gain_by_policy
from fraud_detection.modeling import temporal_folds
from fraud_detection.paths import BEST_PARAMS_JSON

logger = logging.getLogger(__name__)


def _periodos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Desarrollo y período reservado."""
    return split_development(normalize_text(load_transactions()))


def tune(trials: int = N_TRIALS) -> None:
    """Buscar hiperparámetros y dejar la configuración en disco."""
    development, _ = _periodos()
    bloques = list(temporal_folds(development[DATE]))

    estudio = tuning.run_search(development, bloques, n_trials=trials)
    destino = tuning.save_best(estudio)

    logger.info("%d pruebas, %d podadas", len(estudio.trials), tuning.pruned_trials(estudio))
    logger.info("mejor ganancia en validación: %.0f", estudio.best_value)
    logger.info("configuración escrita en %s", destino)


def train() -> None:
    """Entrenar con todo el desarrollo y guardar el artefacto."""
    development, _ = _periodos()
    parametros = tuning.load_best()["parametros"]

    modelo = training.train(development, XGBClassifier(**parametros))
    destino = training.save_model(modelo, development)

    logger.info("entrenado con %d transacciones", len(development))
    logger.info("artefacto en %s", destino)


def evaluate() -> None:
    """Medir el artefacto guardado sobre el período reservado."""
    _, test = _periodos()
    artefacto = training.load_model()

    probabilidad = artefacto["pipeline"].predict_proba(test.drop(columns=[TARGET]))[:, 1]
    politicas = {
        "aprobar todo": pd.Series(True, index=test.index),
        "modelo": pd.Series(probabilidad, index=test.index).lt(artefacto["umbral"]),
    }
    tabla = gain_by_policy(test, politicas)
    tabla["sobre_aprobar_todo"] = tabla.ganancia - tabla.loc["aprobar todo", "ganancia"]

    logger.info("período reservado:\n%s", tabla.round(2).to_string())


def construir_parser() -> argparse.ArgumentParser:
    """Un subcomando por función, guardando la función misma como valor."""
    parser = argparse.ArgumentParser(prog="fraud-detection", description=__doc__)
    comandos = parser.add_subparsers(required=True)

    buscar = comandos.add_parser("tune", help=tune.__doc__)
    buscar.add_argument("--trials", type=int, default=N_TRIALS)
    buscar.set_defaults(funcion=tune)

    entrenar = comandos.add_parser("train", help=f"entrenar con {BEST_PARAMS_JSON.name}")
    entrenar.set_defaults(funcion=train)

    medir = comandos.add_parser("evaluate", help=evaluate.__doc__)
    medir.set_defaults(funcion=evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de `fraud-detection`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argumentos = vars(construir_parser().parse_args(argv))
    argumentos.pop("funcion")(**argumentos)


if __name__ == "__main__":
    main()
