"""Testes da Fase 7 — tratamento de erros de ponta a ponta.

Cobre os casos pedidos no enunciado (arquivo inválido, sem câmera, sem
detecções, ROI inválida, homografia degenerada, pesos sem internet) e falhas
no meio da análise, tanto na sessão quanto na interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.acquisition import fontes
from src.acquisition.erros import WebcamIndisponivelError
from src.acquisition.fontes import abrir_arquivo
from src.aplicacao import sessao as modulo_sessao
from src.aplicacao.sessao import ErroAnalise, SessaoAnalise
from src.config import PROJECT_ROOT, AppConfig
from src.models import detector_yolo
from src.models.analisador import MSG_SEM_DETECCOES
from src.models.tipos import DeteccaoBruta, PesosIndisponiveisError
from tests.falsos import DetectorPorCor


class DetectorQueFalha(DetectorPorCor):
    """Funciona nas primeiras chamadas e depois lança um erro (ex.: falta de memória)."""

    def __init__(self, falhar_na_chamada: int = 5) -> None:
        super().__init__()
        self.falhar_na_chamada = falhar_na_chamada

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        if self.chamadas + 1 >= self.falhar_na_chamada:
            raise MemoryError("memória insuficiente (simulado)")
        return super().rastrear(imagem)


def _cfg_linha_no_meio() -> AppConfig:
    cfg = AppConfig()
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.5), (1.0, 0.5)
    return cfg


# ---------------------------------------------------------------------------
# Sessão
# ---------------------------------------------------------------------------
def test_erro_no_meio_da_analise_encerra_a_sessao_e_guarda_o_video(fabrica_video, tmp_path: Path) -> None:
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=20)), AppConfig(), DetectorQueFalha(5), tmp_path)
    for _ in range(4):
        assert sessao.passo() is not None
    with pytest.raises(ErroAnalise, match="interrompida no frame 5"):
        sessao.passo()
    assert sessao.encerrada and not sessao.fonte.aberta
    assert sessao.erro is not None and "MemoryError" in sessao.erro
    assert not sessao.fonte_terminou
    assert sessao.caminho_video is not None and sessao.caminho_video.exists()  # vídeo parcial
    assert sessao.passo() is None  # depois do erro não processa mais nada


def test_falha_na_gravacao_nao_interrompe_a_analise(fabrica_video, tmp_path: Path) -> None:
    arquivo = tmp_path / "nao_e_pasta.txt"
    arquivo.write_text("x")  # a "pasta de saída" é um arquivo: mkdir falha
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=40)), _cfg_linha_no_meio(), DetectorPorCor(), arquivo)
    while sessao.passo() is not None:
        pass
    assert sessao.contador.total == 2  # a contagem foi até o fim
    assert sessao.caminho_video is None and sessao.erro is None
    assert len(sessao.avisos) == 1 and "não será salvo" in sessao.avisos[0]


def test_fim_da_fonte_e_registrado(fabrica_video) -> None:
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=3)), AppConfig(), DetectorPorCor(), None)
    while sessao.passo() is not None:
        pass
    assert sessao.fonte_terminou and not sessao.ao_vivo


def test_parar_nao_marca_fim_da_fonte(fabrica_video) -> None:
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=10)), AppConfig(), DetectorPorCor(), None)
    sessao.passo()
    sessao.finalizar()
    assert not sessao.fonte_terminou


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def _app_com_video(monkeypatch: pytest.MonkeyPatch, video: Path, tmp_path: Path, detector=None) -> AppTest:
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [video])
    monkeypatch.setattr(modulo_sessao, "PASTA_VIDEOS", tmp_path / "videos")
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    if detector is not None:
        at.session_state["detectores"] = {"yolo11n.pt": detector}
    return at.run()


def test_app_erro_no_meio_do_video_mostra_mensagem_e_resultados(
    monkeypatch: pytest.MonkeyPatch, fabrica_video, tmp_path: Path
) -> None:
    at = _app_com_video(monkeypatch, fabrica_video(n_frames=20), tmp_path, DetectorQueFalha(8))
    at.button(key="analisar").click().run()
    assert not at.exception
    assert not at.session_state["rodando"]
    assert any("interrompida no frame" in e.value for e in at.error)
    assert any(s.value == "Resultado da análise" for s in at.subheader)  # downloads continuam
    assert at.button(key="parar").disabled  # barra lateral redesenhada no estado "parado"


def test_app_video_corrompido_mostra_erro(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    falso = tmp_path / "quebrado.mp4"
    falso.write_bytes(b"isto nao e um video" * 100)
    at = _app_com_video(monkeypatch, falso, tmp_path)
    assert not at.exception
    assert at.error and not at.image


def test_app_sem_camera_mostra_erro_amigavel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def sem_camera(*_a, **_k):
        raise WebcamIndisponivelError("Não foi possível acessar a câmera 0 (simulado).")

    monkeypatch.setattr(fontes, "abrir_webcam", sem_camera)
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [])
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.session_state["detectores"] = {"yolo11n.pt": DetectorPorCor()}
    at.run()
    at.sidebar.radio(key="tipo_fonte").set_value("Webcam").run()
    at.button(key="analisar").click().run()
    assert not at.exception
    assert any("câmera" in e.value for e in at.error)
    assert not at.session_state["rodando"]


def test_app_sem_deteccoes_avisa_para_reduzir_a_confianca(
    monkeypatch: pytest.MonkeyPatch, fabrica_video, tmp_path: Path
) -> None:
    at = _app_com_video(monkeypatch, fabrica_video(n_frames=35), tmp_path, DetectorPorCor(alvos=()))
    at.button(key="analisar").click().run()
    assert not at.exception
    assert any(w.value == MSG_SEM_DETECCOES for w in at.warning)


def test_app_pesos_sem_internet_orienta_o_download(
    monkeypatch: pytest.MonkeyPatch, fabrica_video, tmp_path: Path
) -> None:
    class SemInternet:
        def __init__(self, *_a, **_k) -> None:
            raise PesosIndisponiveisError("Não foi possível baixar os pesos 'yolo11n.pt'. ... internet ...")

    monkeypatch.setattr(detector_yolo, "DetectorYOLO", SemInternet)
    at = _app_com_video(monkeypatch, fabrica_video(n_frames=5), tmp_path)
    at.button(key="analisar").click().run()
    assert not at.exception
    assert any("internet" in e.value for e in at.error)
    assert at.session_state["sessao"] is None


def test_homografia_degenerada_so_desativa_a_velocidade(fabrica_video) -> None:
    cfg = _cfg_linha_no_meio()
    cfg.geometria.homografia_origem = [(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)]  # cruzado
    sessao = SessaoAnalise(abrir_arquivo(fabrica_video(n_frames=40)), cfg, DetectorPorCor(), None)
    while sessao.passo() is not None:
        pass
    assert any("Velocidade desativada" in a for a in sessao.analisador.avisos)
    assert sessao.contador.total == 2  # a contagem funciona normalmente
    assert all(e.velocidade_kmh is None for e in sessao.eventos)
