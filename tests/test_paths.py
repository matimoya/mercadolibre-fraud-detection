"""Resolución de rutas del proyecto."""

from pathlib import Path

import pytest

from fraud_detection import paths

RUTAS = [
    paths.DATA_DIR,
    paths.DATASET_CSV,
    paths.MODELS_DIR,
    paths.BEST_PARAMS_JSON,
    paths.MODEL_JOBLIB,
    paths.TRACKING_DB,
    paths.ARTIFACTS_DIR,
]


@pytest.mark.parametrize("ruta", RUTAS)
def test_rutas(ruta: Path):
    """Todas absolutas y colgando de la raíz: si alguna fuera relativa dependería del cwd."""
    assert (paths.ROOT / paths.MARCA).is_file()
    assert ruta.is_absolute()
    assert paths.ROOT in ruta.parents


def test_variable_de_entorno(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Es el escape para cuando el paquete se instala fuera del repositorio."""
    monkeypatch.setenv(paths.VARIABLE_DE_ENTORNO, str(tmp_path))
    assert paths.encontrar_raiz() == tmp_path.resolve()


def test_sin_raiz(monkeypatch: pytest.MonkeyPatch):
    """Fallar temprano es mejor que leer un CSV inexistente a mitad de un notebook."""
    monkeypatch.delenv(paths.VARIABLE_DE_ENTORNO, raising=False)
    monkeypatch.setattr(paths, "MARCA", "archivo-que-no-existe.toml")
    with pytest.raises(RuntimeError, match=paths.VARIABLE_DE_ENTORNO):
        paths.encontrar_raiz()
