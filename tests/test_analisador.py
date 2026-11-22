"""Testes da Fase 3 — pipeline de ponta a ponta com detector falso (sem pesos)."""

from __future__ import annotations

import numpy as np
import pytest

from src.acquisition.fontes import Frame, abrir_arquivo
from src.config import AppConfig
from src.geometry.erros import ErroGeometria, RoiInvalidaError
from src.models.analisador import (
    MSG_SEM_DETECCOES,
    AnalisadorFluxo,
    analisar_imagem,
    ponto_referencia,
)
from src.models.contagem import SENTIDO_AB, SENTIDO_BA
from src.models.tipos import DeteccaoBruta
from tests.falsos import DetectorPorCor, DetectorRoteirizado


def _cfg_linha_no_meio() -> AppConfig:
    cfg = AppConfig()
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.5), (1.0, 0.5)
    return cfg


def _rodar_video(caminho, cfg: AppConfig, detector) -> tuple[AnalisadorFluxo, list]:
    with abrir_arquivo(caminho) as fonte:
        analisador = AnalisadorFluxo(cfg, detector, fonte.info.largura, fonte.info.altura)
        resultados = [analisador.processar(frame) for frame in fonte]
    return analisador, resultados


# ---------------------------------------------------------------------------
# Ponta a ponta: vídeo sintético → pré-processamento → detector → contagem
# ---------------------------------------------------------------------------
def test_ponta_a_ponta_conta_carro_descendo_e_caminhao_subindo(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=40, largura=640, altura=360, fps=15.0)
    detector = DetectorPorCor()
    analisador, resultados = _rodar_video(caminho, _cfg_linha_no_meio(), detector)

    assert detector.reinicios == 1 and detector.chamadas == 40
    tabela = analisador.contador.por_classe()
    assert tabela["carro"] == {SENTIDO_AB: 1, SENTIDO_BA: 0, "total": 1}
    assert tabela["caminhão"] == {SENTIDO_AB: 0, SENTIDO_BA: 1, "total": 1}
    assert analisador.contador.total == 2

    eventos = [e for r in resultados for e in r.eventos]
    assert [(e.id_rastreio, e.sentido) for e in eventos] == [(1, SENTIDO_AB), (2, SENTIDO_BA)]
    assert all(e.timestamp_s == pytest.approx(e.frame / 15.0) for e in eventos)

    # tempos por etapa medidos em todo frame processado
    assert all(set(r.tempos_ms) == {"preprocessamento", "inferencia", "interpretacao"} for r in resultados)
    assert not analisador.sem_deteccoes


def test_caixas_voltam_ao_espaco_original(fabrica_video) -> None:
    """O retângulo azul do frame 12 está em x=160..205, y=108..153 no vídeo 640×360."""
    caminho = fabrica_video(n_frames=13, largura=640, altura=360)
    cfg = _cfg_linha_no_meio()
    cfg.preproc.usar_roi = False
    _, resultados = _rodar_video(caminho, cfg, DetectorPorCor())
    carro = next(v for v in resultados[12].veiculos if v.id_rastreio == 1)
    np.testing.assert_allclose(carro.caixa, (160, 108, 205, 153), atol=2)


def test_ponta_a_ponta_com_frame_skip(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=40, largura=640, altura=360)
    cfg = _cfg_linha_no_meio()
    cfg.preproc.frame_skip = 1
    detector = DetectorPorCor()
    analisador, resultados = _rodar_video(caminho, cfg, detector)
    assert detector.chamadas == 20                           # metade dos frames
    assert [r.processado for r in resultados[:4]] == [True, False, True, False]
    assert resultados[1].veiculos == resultados[0].veiculos  # frame pulado repete o anterior
    assert analisador.contador.total == 2


def test_ponta_a_ponta_com_todos_os_preprocessamentos(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=40, largura=640, altura=360)
    cfg = _cfg_linha_no_meio()
    cfg.preproc.usar_clahe = True
    cfg.preproc.usar_reducao_ruido = True
    analisador, _ = _rodar_video(caminho, cfg, DetectorPorCor(tolerancia=70))
    assert analisador.contador.total == 2


def test_video_sem_deteccoes(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=5)
    analisador, resultados = _rodar_video(caminho, AppConfig(), DetectorRoteirizado([]))
    assert analisador.sem_deteccoes
    assert analisador.mensagem_sem_deteccoes == MSG_SEM_DETECCOES
    assert all(not r.veiculos for r in resultados)


# ---------------------------------------------------------------------------
# Filtros da interpretação (detector roteirizado num frame 640×640 sem faixas)
# ---------------------------------------------------------------------------
def _frame(indice: int = 0, fps: float = 10.0) -> Frame:
    return Frame(indice, indice / fps, np.zeros((640, 640, 3), dtype=np.uint8))


def _cfg_sem_roi() -> AppConfig:
    cfg = AppConfig()
    cfg.preproc.usar_roi = False
    return cfg


def test_filtra_pessoa_confianca_baixa_e_caixa_na_faixa() -> None:
    roteiro = [[
        DeteccaoBruta((10, 10, 50, 50), 2, 0.9, 1),      # carro válido
        DeteccaoBruta((100, 10, 150, 50), 0, 0.9, 2),    # pessoa → ignorada
        DeteccaoBruta((200, 10, 250, 50), 2, 0.1, 3),    # confiança baixa → ignorada
        DeteccaoBruta((300, 10, 350, 50), 5, 0.8, None), # ônibus ainda sem ID → exibido, não rastreado
    ]]
    analisador = AnalisadorFluxo(_cfg_sem_roi(), DetectorRoteirizado(roteiro), 640, 640)
    resultado = analisador.processar(_frame())
    assert [(v.id_rastreio, v.nome_classe) for v in resultado.veiculos] == [(1, "carro"), (None, "ônibus")]
    assert 3 not in analisador.historico and len(analisador.historico) == 1


def test_caixa_dentro_da_faixa_do_letterbox_e_descartada() -> None:
    # frame 1280×720 → faixa de 140 px no topo do quadrado 640×640
    roteiro = [[DeteccaoBruta((10, 10, 60, 120), 2, 0.9, 1)]]
    analisador = AnalisadorFluxo(_cfg_sem_roi(), DetectorRoteirizado(roteiro), 1280, 720)
    frame = Frame(0, 0.0, np.zeros((720, 1280, 3), dtype=np.uint8))
    assert analisador.processar(frame).veiculos == []


def test_roi_filtra_pelo_ponto_de_referencia() -> None:
    cfg = AppConfig()
    cfg.geometria.roi = [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]  # metade de baixo
    roteiro = [[
        DeteccaoBruta((10, 100, 60, 200), 2, 0.9, 1),    # base em y=200 (< 320) → fora
        DeteccaoBruta((10, 300, 60, 400), 2, 0.9, 2),    # base em y=400 → dentro
    ]]
    analisador = AnalisadorFluxo(cfg, DetectorRoteirizado(roteiro), 640, 640)
    assert [v.id_rastreio for v in analisador.processar(_frame()).veiculos] == [2]


def test_velocidade_e_contagem_com_detector_roteirizado() -> None:
    """Frame 640×640 calibrado como 64 m × 64 m (10 px = 1 m). 10 px/frame a 10 fps = 36 km/h."""
    cfg = _cfg_sem_roi()
    cfg.geometria.homografia_origem = [(0, 0), (1, 0), (1, 1), (0, 1)]
    cfg.geometria.largura_pista_m = cfg.geometria.comprimento_pista_m = 64.0
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.6), (1.0, 0.6)   # y = 384 px
    roteiro = [[DeteccaoBruta((300, 200 + 10 * i, 340, 240 + 10 * i), 2, 0.9, 7)] for i in range(30)]
    analisador = AnalisadorFluxo(cfg, DetectorRoteirizado(roteiro), 640, 640)
    resultados = [analisador.processar(_frame(i)) for i in range(30)]

    assert resultados[-1].veiculos[0].velocidade_kmh == pytest.approx(36.0, rel=0.02)
    (evento,) = analisador.contador.eventos
    assert evento.sentido == SENTIDO_AB and evento.velocidade_kmh == pytest.approx(36.0, rel=0.05)
    assert not analisador.avisos


def test_homografia_degenerada_so_desativa_velocidade() -> None:
    cfg = _cfg_sem_roi()
    cfg.geometria.homografia_origem = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9)]
    roteiro = [[DeteccaoBruta((300, 10 * i, 340, 40 + 10 * i), 2, 0.9, 1)] for i in range(10)]
    analisador = AnalisadorFluxo(cfg, DetectorRoteirizado(roteiro), 640, 640)
    resultados = [analisador.processar(_frame(i)) for i in range(10)]
    assert analisador.homografia is None
    assert analisador.avisos and "Velocidade desativada" in analisador.avisos[0]
    assert resultados[-1].veiculos[0].velocidade_kmh is None


def test_roi_invalida_e_linha_degenerada_impedem_analise() -> None:
    cfg = AppConfig()
    cfg.geometria.roi = [(0.1, 0.1), (0.2, 0.2)]
    with pytest.raises(RoiInvalidaError):
        AnalisadorFluxo(cfg, DetectorRoteirizado([]), 640, 480)
    cfg = _cfg_sem_roi()
    cfg.contagem.linha_p1 = cfg.contagem.linha_p2 = (0.5, 0.5)
    with pytest.raises(ErroGeometria):
        AnalisadorFluxo(cfg, DetectorRoteirizado([]), 640, 480)


def test_ponto_referencia() -> None:
    assert ponto_referencia((10, 20, 30, 60), "base") == (20, 60)
    assert ponto_referencia((10, 20, 30, 60), "centro") == (20, 40)
    with pytest.raises(ValueError):
        ponto_referencia((0, 0, 1, 1), "topo")


def test_analisar_imagem_unica() -> None:
    imagem = np.full((360, 640, 3), 90, dtype=np.uint8)
    imagem[200:245, 100:145] = (255, 170, 0)
    veiculos, preproc = analisar_imagem(imagem, _cfg_sem_roi(), DetectorPorCor())
    assert preproc.imagem.shape == (640, 640, 3)
    (carro,) = veiculos
    assert carro.nome_classe == "carro" and carro.id_rastreio is None
    np.testing.assert_allclose(carro.caixa, (100, 200, 145, 245), atol=2)
