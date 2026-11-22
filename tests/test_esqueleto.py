"""Testes de fumaça da Fase 0: configuração e abertura da interface."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.config import CLASSES_VEICULO, PROJECT_ROOT, AppConfig


def test_config_padrao_consistente() -> None:
    cfg = AppConfig()
    assert set(cfg.modelo.classes) == {2, 3, 5, 7}
    assert set(CLASSES_VEICULO) == set(cfg.modelo.classes)
    assert len(cfg.geometria.roi) >= 3
    assert len(cfg.geometria.homografia_origem) == 4
    assert 0 < cfg.geometria.ema_alpha <= 1
    assert cfg.preproc.kernel_ruido % 2 == 1


def test_instancias_nao_compartilham_listas() -> None:
    a, b = AppConfig(), AppConfig()
    a.geometria.roi.append((0.5, 0.5))
    assert len(b.geometria.roi) == 4


def test_caminhos_relativos_a_raiz() -> None:
    assert (PROJECT_ROOT / "app.py").is_file()


def test_app_abre_sem_erros() -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not at.exception
    assert at.title[0].value.endswith("FluxoVC")
