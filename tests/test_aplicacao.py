"""Testes da Fase 5 — sessão de análise, relatórios, gravador e interface Streamlit."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.acquisition import fontes
from src.acquisition.fontes import abrir_arquivo
from src.aplicacao import sessao as modulo_sessao
from src.aplicacao.relatorios import (
    deteccoes_para_dataframe,
    eventos_para_dataframe,
    para_csv,
    resumo_contagem,
    serie_contagem,
)
from src.aplicacao.sessao import (
    SessaoAnalise,
    analisar_e_compor_imagem,
    codificar_png,
    config_da_sessao,
    previa_configuracao,
    primeiro_frame,
    validar_configuracao,
)
from src.config import PROJECT_ROOT, AppConfig
from src.geometry.erros import RoiInvalidaError
from src.models.contagem import SENTIDO_AB, SENTIDO_BA, ContadorLinha, EventoCruzamento
from src.visualization.gravador import GravadorVideo
from tests.falsos import DetectorPorCor


def _cfg_linha_no_meio() -> AppConfig:
    cfg = AppConfig()
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.5), (1.0, 0.5)
    return cfg


# ---------------------------------------------------------------------------
# Sessão de análise
# ---------------------------------------------------------------------------
def test_sessao_processa_video_inteiro_e_grava_resultado(fabrica_video, tmp_path: Path) -> None:
    caminho = fabrica_video(n_frames=40, largura=640, altura=360, fps=15.0)
    sessao = SessaoAnalise(abrir_arquivo(caminho), _cfg_linha_no_meio(), DetectorPorCor(), tmp_path)
    passos = 0
    while (passo := sessao.passo()) is not None:
        passos += 1
        assert passo.imagem_final.shape == (360, 640, 3)
    assert passos == 40 and sessao.encerrada and sessao.progresso == 1.0
    assert sessao.contador.total == 2
    assert sessao.passo() is None  # depois de encerrada não lê mais nada

    assert sessao.caminho_video is not None and sessao.caminho_video.parent == tmp_path
    with abrir_arquivo(sessao.caminho_video) as video:  # o MP4 gravado abre e tem todos os frames
        assert video.info.total_frames == 40

    tempos = sessao.tempos_dataframe()
    assert len(tempos) == 40
    assert set(tempos.columns) == {"preprocessamento", "inferencia", "interpretacao", "desenho"}
    assert sessao.fps_medio() is not None and sessao.fps_medio() > 0


def test_sessao_interrompida_finaliza_sem_erro(fabrica_video, tmp_path: Path) -> None:
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=20)), AppConfig(), DetectorPorCor(), tmp_path)
    for _ in range(5):
        sessao.passo()
    sessao.finalizar()
    sessao.finalizar()  # idempotente
    assert sessao.frames_lidos == 5 and not sessao.fonte.aberta
    assert sessao.progresso == pytest.approx(5 / 20)


def test_sessao_sem_gravar_video(fabrica_video) -> None:
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=3)), AppConfig(), DetectorPorCor(), None)
    while sessao.passo() is not None:
        pass
    assert sessao.caminho_video is None


def test_sessao_com_roi_invalida_libera_a_fonte(fabrica_video) -> None:
    cfg = AppConfig()
    cfg.geometria.roi = [(0.1, 0.1), (0.2, 0.2)]
    fonte = abrir_arquivo(fabrica_video(n_frames=3))
    with pytest.raises(RoiInvalidaError):
        SessaoAnalise(fonte, cfg, DetectorPorCor(), None)
    assert not fonte.aberta


def test_parametros_ao_vivo_e_geometria_congelada() -> None:
    cfg = AppConfig()
    copia = config_da_sessao(cfg)
    cfg.modelo.confianca = 0.8
    cfg.preproc.usar_clahe = True
    cfg.visualizacao.mostrar_heatmap = True
    cfg.contagem.linha_p1 = (0.3, 0.3)
    cfg.geometria.roi.append((0.5, 0.5))
    assert copia.modelo.confianca == 0.8 and copia.preproc.usar_clahe and copia.visualizacao.mostrar_heatmap
    assert copia.contagem.linha_p1 == AppConfig().contagem.linha_p1
    assert len(copia.geometria.roi) == 4


def test_toggle_de_roi_durante_a_analise(fabrica_video) -> None:
    cfg = AppConfig()
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=4)), cfg, DetectorPorCor(), None)
    assert sessao.analisador.roi_ativa is not None
    cfg.preproc.usar_roi = False  # desligado na interface, com a análise rodando
    assert sessao.analisador.roi_ativa is None
    assert sessao.passo() is not None


# ---------------------------------------------------------------------------
# Imagem única, prévia e validação
# ---------------------------------------------------------------------------
def test_imagem_unica_composta() -> None:
    imagem = np.full((360, 640, 3), 90, dtype=np.uint8)
    imagem[250:295, 100:145] = (255, 170, 0)
    veiculos, preproc, final = analisar_e_compor_imagem(imagem, AppConfig(), DetectorPorCor())
    assert [v.nome_classe for v in veiculos] == ["carro"]
    assert preproc.imagem.shape == (640, 640, 3) and final.shape == (360, 640, 3)
    assert codificar_png(final)[:4] == b"\x89PNG"


def test_previa_e_primeiro_frame(fabrica_video) -> None:
    cfg = AppConfig()
    imagem = primeiro_frame(fabrica_video(n_frames=5, largura=1280, altura=720), cfg)
    assert imagem.shape == (720, 1280, 3)
    previa = previa_configuracao(imagem, cfg)
    assert previa.shape == (540, 960, 3)
    assert not np.array_equal(previa, cv2.resize(imagem, (960, 540), interpolation=cv2.INTER_AREA))


def test_validar_configuracao() -> None:
    erros, avisos = validar_configuracao(AppConfig(), 1280, 720)
    assert erros == [] and len(avisos) == 1 and "calibração da velocidade ainda é a padrão" in avisos[0]
    calibrada = AppConfig()
    calibrada.geometria.comprimento_pista_m = 35.0  # medida real informada: sem aviso
    assert validar_configuracao(calibrada, 1280, 720) == ([], [])
    cfg = AppConfig()
    cfg.geometria.roi = [(0.1, 0.1)]
    cfg.contagem.linha_p1 = cfg.contagem.linha_p2 = (0.5, 0.5)
    cfg.geometria.homografia_origem = [(0, 0), (1, 0), (0, 1), (1, 1)]  # cruzado
    erros, avisos = validar_configuracao(cfg, 1280, 720)
    assert len(erros) == 2 and len(avisos) == 1
    cfg.preproc.usar_roi = False  # ROI desligada: polígono inválido não é erro
    assert len(validar_configuracao(cfg, 1280, 720)[0]) == 1


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------
def _eventos() -> list[EventoCruzamento]:
    return [
        EventoCruzamento(1, 2, SENTIDO_AB, 10, 0.5, 42.123),
        EventoCruzamento(2, 7, SENTIDO_BA, 40, 2.0, None),
        EventoCruzamento(3, 2, SENTIDO_BA, 70, 3.5, 30.0),
    ]


def test_tabela_de_eventos() -> None:
    tabela = eventos_para_dataframe(_eventos())
    assert list(tabela.columns) == ["id", "classe", "sentido", "frame", "instante_s", "velocidade_kmh"]
    assert tabela.loc[0, "classe"] == "carro" and tabela.loc[0, "velocidade_kmh"] == 42.1
    assert pd.isna(tabela.loc[1, "velocidade_kmh"])
    assert eventos_para_dataframe([]).empty


def test_resumo_com_total() -> None:
    contador = ContadorLinha((0, 100), (200, 100))
    contador.atualizar(1, 2, (100, 90), 0, 0.0)
    contador.atualizar(1, 2, (100, 110), 1, 0.1)
    resumo = resumo_contagem(contador)
    assert resumo.loc["carro", SENTIDO_AB] == 1
    assert resumo.loc["TOTAL", "total"] == 1


def test_serie_acumulada() -> None:
    serie = serie_contagem(_eventos(), duracao_s=5.0)
    assert list(serie.index) == [0, 1, 2, 3, 4, 5]
    assert list(serie["carro"]) == [0, 1, 1, 1, 2, 2]
    assert list(serie["caminhão"]) == [0, 0, 1, 1, 1, 1]
    assert serie["moto"].sum() == 0
    assert len(serie_contagem([])) == 1  # sem eventos: só o instante 0


def test_csv_abre_no_excel_brasileiro() -> None:
    dados = para_csv(eventos_para_dataframe(_eventos()))
    assert dados.startswith(b"\xef\xbb\xbf")                      # BOM UTF-8
    texto = dados.decode("utf-8-sig")
    assert texto.splitlines()[0] == "id;classe;sentido;frame;instante_s;velocidade_kmh"
    assert "42,1" in texto and "caminhão" in texto
    relido = pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
    assert relido["velocidade_kmh"].iloc[0] == pytest.approx(42.1)


def test_deteccoes_de_imagem() -> None:
    from src.models.analisador import VeiculoRastreado

    tabela = deteccoes_para_dataframe([VeiculoRastreado(None, 5, 0.91234, (1.4, 2.6, 30, 40), (0, 0))])
    assert tabela.iloc[0].tolist() == ["ônibus", 0.912, 1, 3, 30, 40]


# ---------------------------------------------------------------------------
# Gravador
# ---------------------------------------------------------------------------
def test_gravador_usa_codec_disponivel_e_redimensiona(tmp_path: Path) -> None:
    gravador = GravadorVideo(tmp_path / "saida.mp4", 10.0)
    gravador.escrever(np.zeros((240, 320, 3), dtype=np.uint8))
    gravador.escrever(np.zeros((480, 640, 3), dtype=np.uint8))  # tamanho diferente: ajustado
    assert gravador.codec in {"avc1", "mp4v"}
    caminho = gravador.fechar()
    assert caminho is not None
    with abrir_arquivo(caminho) as video:
        assert (video.info.largura, video.info.altura) == (320, 240)


def test_gravador_sem_frames(tmp_path: Path) -> None:
    assert GravadorVideo(tmp_path / "vazio.mp4", 10.0).fechar() is None


# ---------------------------------------------------------------------------
# Interface (AppTest roda o app.py de verdade, sem navegador)
# ---------------------------------------------------------------------------
def _app() -> AppTest:
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)


def test_app_abre_com_controles_e_sem_videos_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [])
    at = _app().run()
    assert not at.exception
    assert at.title[0].value.endswith("FluxoVC")
    assert {"Limiar de confiança", "Limiar de IoU (NMS)", "Frame skip"} <= {s.label for s in at.sidebar.slider}
    assert any("Nenhum vídeo" in i.value for i in at.sidebar.info)
    assert at.button(key="analisar").disabled  # sem fonte, não dá para analisar


def test_app_analisa_video_demo_de_ponta_a_ponta(
    monkeypatch: pytest.MonkeyPatch, fabrica_video, tmp_path: Path
) -> None:
    video = fabrica_video(nome="demo.avi", n_frames=40, largura=640, altura=360)
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [video])
    monkeypatch.setattr(modulo_sessao, "PASTA_VIDEOS", tmp_path / "videos")

    at = _app()
    at.session_state["detectores"] = {"yolo11n.pt": DetectorPorCor()}  # sem baixar pesos
    at.run()
    assert not at.exception
    assert at.image  # prévia da configuração com o primeiro frame

    # linha no meio do frame
    at.slider(key="lx1").set_value(0.0)
    at.slider(key="ly1").set_value(0.5)
    at.slider(key="lx2").set_value(1.0)
    at.slider(key="ly2").set_value(0.5).run()
    at.button(key="analisar").click().run()
    assert not at.exception

    sessao = at.session_state["sessao"]
    assert sessao.encerrada and not at.session_state["rodando"]
    assert sessao.contador.total == 2
    assert {m.label: m.value for m in at.metric}["Veículos contados"] == "2"
    assert any(s.value == "Resultado da análise" for s in at.subheader)
    assert (tmp_path / "videos").exists()

    at.button(key="limpar").click().run()
    assert at.session_state["sessao"] is None and not at.exception


def test_app_mostra_erro_de_roi_invalida_na_previa(monkeypatch: pytest.MonkeyPatch, fabrica_video) -> None:
    video = fabrica_video(nome="demo.avi", n_frames=5)
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [video])
    at = _app().run()
    at.slider(key="lx1").set_value(0.5)
    at.slider(key="lx2").set_value(0.5)
    at.slider(key="ly1").set_value(0.5)
    at.slider(key="ly2").set_value(0.5).run()
    assert any("linha de contagem" in e.value for e in at.error)
