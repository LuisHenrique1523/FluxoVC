"""Marcação com o mouse: linha, ROI e calibração clicando na prévia."""

from __future__ import annotations

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.acquisition import fontes
from src.aplicacao.marcacao import (
    COR_MARCACAO,
    MODOS,
    completa,
    desenhar_marcacao,
    finalizar,
    ordenar_quadrilatero,
    ponto_do_clique,
)
from src.config import PROJECT_ROOT
from src.geometry.erros import ErroGeometria


def test_clique_vira_ponto_normalizado() -> None:
    # imagem exibida com 800×450 px: clique no meio
    assert ponto_do_clique({"x": 400, "y": 225, "width": 800, "height": 450, "unix_time": 1}) == (0.5, 0.5)
    assert ponto_do_clique({"x": 900, "y": -5, "width": 800, "height": 450}) == (1.0, 0.0)  # limitado
    assert ponto_do_clique(None) is None
    assert ponto_do_clique({"x": 1, "y": 1, "width": 0, "height": 0}) is None


def test_quantos_cliques_cada_modo_precisa() -> None:
    assert completa("linha", [(0, 0), (1, 1)]) and not completa("linha", [(0, 0)])
    assert completa("calibracao", [(0, 0)] * 4)
    assert not completa("roi", [(0, 0)] * 10)  # ROI termina com o botão "Concluir"
    assert MODOS["roi"].minimo == 3


def test_calibracao_aceita_cliques_em_qualquer_ordem() -> None:
    trapezio = [(0.35, 0.4), (0.65, 0.4), (0.95, 0.95), (0.05, 0.95)]  # sup.esq, sup.dir, inf.dir, inf.esq
    embaralhado = [trapezio[2], trapezio[0], trapezio[3], trapezio[1]]
    assert ordenar_quadrilatero(embaralhado) == trapezio
    assert finalizar("calibracao", embaralhado, 1280, 720) == trapezio


def test_finalizar_valida_com_as_regras_da_analise() -> None:
    assert finalizar("linha", [(0.1, 0.5), (0.9, 0.5)], 640, 360) == [(0.1, 0.5), (0.9, 0.5)]
    with pytest.raises(ErroGeometria):
        finalizar("linha", [(0.5, 0.5), (0.5, 0.5)], 640, 360)          # pontos iguais
    with pytest.raises(ErroGeometria):
        finalizar("roi", [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)], 640, 360)  # alinhados
    with pytest.raises(ErroGeometria):
        finalizar("calibracao", [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1)], 640, 360)


def test_desenha_os_pontos_marcados() -> None:
    previa = np.zeros((360, 640, 3), dtype=np.uint8)
    saida = desenhar_marcacao(previa, "roi", [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)])
    assert previa.sum() == 0 and saida.sum() > 0  # não altera a original
    assert tuple(saida[72, 128]) == COR_MARCACAO  # centro do 1º ponto em laranja
    assert np.array_equal(desenhar_marcacao(previa, "linha", []), previa)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, fabrica_video) -> AppTest:
    video = fabrica_video(nome="demo.avi", n_frames=5)
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [video])
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)


def test_cliques_aplicados_preenchem_a_barra_lateral(app: AppTest) -> None:
    app.session_state["_aplicar_marcacao"] = ("linha", [(0.104, 0.3), (0.9, 0.7)])
    app.run()
    assert not app.exception
    valores = {k: app.slider(key=k).value for k in ("lx1", "ly1", "lx2", "ly2")}
    assert valores == {"lx1": 0.1, "ly1": 0.3, "lx2": 0.9, "ly2": 0.7}
    assert app.session_state["cfg"].contagem.linha_p1 == (0.1, 0.3)

    roi = [(0.1, 0.4), (0.9, 0.4), (0.9, 1.0), (0.1, 1.0)]
    app.session_state["_aplicar_marcacao"] = ("roi", roi)
    app.run()
    assert app.session_state["cfg"].geometria.roi == roi

    calib = [(0.3, 0.4), (0.7, 0.4), (0.9, 0.9), (0.1, 0.9)]
    app.session_state["_aplicar_marcacao"] = ("calibracao", calib)
    app.run()
    assert app.session_state["cfg"].geometria.homografia_origem == calib
    assert app.session_state["modo_marcacao"].startswith("Nenhum")
    # calibração mexida: o aviso de "calibração padrão" some
    assert not any("ainda é a padrão" in w.value for w in app.warning)


def test_modo_de_marcacao_mostra_instrucoes(app: AppTest) -> None:
    app.run()
    assert app.image  # sem modo: prévia normal
    app.radio(key="modo_marcacao").set_value("🚀 Calibração").run()
    assert not app.exception
    assert any("4 cantos" in i.value for i in app.info)
    assert app.button(key="marc_desfazer").disabled  # nenhum ponto ainda
