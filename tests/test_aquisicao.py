"""Testes da Fase 1 — aquisição e validação de entradas."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.acquisition import fontes
from src.acquisition.erros import (
    ArquivoInvalidoError,
    FonteInvalidaError,
    WebcamIndisponivelError,
)
from src.acquisition.fontes import (
    abrir_arquivo,
    abrir_webcam,
    carregar_imagem,
    decodificar_imagem,
    listar_videos_demo,
    salvar_upload,
    validar_url,
)
from src.acquisition.validacao import (
    fps_valido,
    normalizar_total_frames,
    validar_extensao,
    validar_resolucao,
    validar_tamanho,
)
from src.config import AquisicaoConfig

EXT_VIDEO = (".mp4", ".avi", ".mov")


# ---------------------------------------------------------------------------
# Regras de validação (funções puras)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nome", ["a.mp4", "pasta/B.MP4", "c.Avi", "d.mov"])
def test_extensao_aceita_maiusculas_e_minusculas(nome: str) -> None:
    assert validar_extensao(nome, EXT_VIDEO) == Path(nome).suffix.lower()


@pytest.mark.parametrize("nome", ["video.txt", "video.mkv", "sem_extensao", "foto.png"])
def test_extensao_errada_e_rejeitada(nome: str) -> None:
    with pytest.raises(ArquivoInvalidoError, match="não suportado"):
        validar_extensao(nome, EXT_VIDEO)


def test_tamanho() -> None:
    validar_tamanho(1024, max_mb=1)
    with pytest.raises(ArquivoInvalidoError, match="vazio"):
        validar_tamanho(0, max_mb=1)
    with pytest.raises(ArquivoInvalidoError, match="muito grande"):
        validar_tamanho(2 * 1024 * 1024, max_mb=1)


def test_resolucao_minima() -> None:
    validar_resolucao(320, 240, (320, 240))
    with pytest.raises(ArquivoInvalidoError, match="Resolução"):
        validar_resolucao(319, 480, (320, 240))
    with pytest.raises(ArquivoInvalidoError, match="Resolução"):
        validar_resolucao(640, 200, (320, 240))


@pytest.mark.parametrize(
    ("bruto", "esperado", "usou_fallback"),
    [
        (29.97, 29.97, False),
        (60.0, 60.0, False),
        (0.0, 30.0, True),
        (-1.0, 30.0, True),
        (math.nan, 30.0, True),
        (math.inf, 30.0, True),
        (90000.0, 30.0, True),
        (None, 30.0, True),
    ],
)
def test_fps_com_fallback(bruto: float | None, esperado: float, usou_fallback: bool) -> None:
    assert fps_valido(bruto, fallback=30.0) == (pytest.approx(esperado), usou_fallback)


@pytest.mark.parametrize(
    ("bruto", "esperado"), [(120.0, 120), (0.0, None), (-1.0, None), (math.nan, None), (None, None)]
)
def test_total_frames(bruto: float | None, esperado: int | None) -> None:
    assert normalizar_total_frames(bruto) == esperado


# ---------------------------------------------------------------------------
# Arquivo de vídeo
# ---------------------------------------------------------------------------
def test_abre_video_valido_e_le_todos_os_frames(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=12, largura=640, altura=360, fps=15.0)
    with abrir_arquivo(caminho) as fonte:
        info = fonte.info
        assert (info.largura, info.altura) == (640, 360)
        assert info.fps == pytest.approx(15.0)
        assert not info.fps_estimado
        assert info.total_frames == 12
        assert info.duracao_s == pytest.approx(12 / 15.0)
        assert info.tipo == "arquivo" and not info.ao_vivo

        frames = list(fonte)
    assert [f.indice for f in frames] == list(range(12))
    assert frames[3].timestamp_s == pytest.approx(3 / 15.0)
    assert frames[0].imagem.shape == (360, 640, 3)
    assert frames[0].imagem.dtype == np.uint8
    assert not fonte.aberta  # o ``with`` liberou a captura
    assert fonte.ler_frame() is None


def test_abre_mp4(fabrica_video) -> None:
    caminho = fabrica_video(nome="sintetico.mp4", fourcc="mp4v", n_frames=8)
    with abrir_arquivo(caminho) as fonte:
        assert sum(1 for _ in fonte) == 8


def test_ler_frame_a_frame(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=3)
    fonte = abrir_arquivo(caminho)
    lidos = [fonte.ler_frame() for _ in range(5)]
    fonte.liberar()
    fonte.liberar()  # liberar duas vezes não dá erro
    assert [f.indice if f else None for f in lidos] == [0, 1, 2, None, None]
    assert fonte.frames_lidos == 3


def test_arquivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ArquivoInvalidoError, match="não encontrado"):
        abrir_arquivo(tmp_path / "nao_existe.mp4")


def test_pasta_no_lugar_de_arquivo(tmp_path: Path) -> None:
    pasta = tmp_path / "video.mp4"
    pasta.mkdir()
    with pytest.raises(ArquivoInvalidoError, match="não é um arquivo"):
        abrir_arquivo(pasta)


def test_extensao_errada_no_arquivo(tmp_path: Path) -> None:
    caminho = tmp_path / "notas.txt"
    caminho.write_text("não é vídeo", encoding="utf-8")
    with pytest.raises(ArquivoInvalidoError, match="não suportado"):
        abrir_arquivo(caminho)


def test_arquivo_vazio(tmp_path: Path) -> None:
    caminho = tmp_path / "vazio.mp4"
    caminho.write_bytes(b"")
    with pytest.raises(ArquivoInvalidoError, match="vazio"):
        abrir_arquivo(caminho)


def test_arquivo_corrompido(tmp_path: Path) -> None:
    caminho = tmp_path / "corrompido.mp4"
    caminho.write_bytes(np.random.default_rng(0).integers(0, 256, 50_000, dtype=np.uint8).tobytes())
    with pytest.raises(ArquivoInvalidoError):
        abrir_arquivo(caminho)


def test_arquivo_acima_do_limite(fabrica_video) -> None:
    caminho = fabrica_video(n_frames=5)
    with pytest.raises(ArquivoInvalidoError, match="muito grande"):
        abrir_arquivo(caminho, AquisicaoConfig(tamanho_max_mb=0.001))


def test_video_com_resolucao_baixa(fabrica_video) -> None:
    caminho = fabrica_video(largura=160, altura=120, n_frames=5)
    with pytest.raises(ArquivoInvalidoError, match="Resolução"):
        abrir_arquivo(caminho)


# ---------------------------------------------------------------------------
# Webcam (simulada — os testes não dependem de câmera real)
# ---------------------------------------------------------------------------
class CapturaFalsa:
    """Imita ``cv2.VideoCapture`` para testar a webcam sem hardware."""

    def __init__(self, aberta: bool = True, frames: int = 3, fps: float = 0.0) -> None:
        self._aberta = aberta
        self._restantes = frames
        self._fps = fps
        self.liberada = False

    def isOpened(self) -> bool:  # noqa: N802 (mesmo nome da API do OpenCV)
        return self._aberta

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._restantes <= 0:
            return False, None
        self._restantes -= 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_FPS: self._fps, cv2.CAP_PROP_FRAME_COUNT: -1.0}.get(prop, 0.0)

    def release(self) -> None:
        self.liberada = True


def _instalar_captura(monkeypatch: pytest.MonkeyPatch, captura: CapturaFalsa) -> None:
    monkeypatch.setattr(fontes.cv2, "VideoCapture", lambda *args, **kwargs: captura)


def test_webcam_ausente(monkeypatch: pytest.MonkeyPatch) -> None:
    captura = CapturaFalsa(aberta=False)
    _instalar_captura(monkeypatch, captura)
    with pytest.raises(WebcamIndisponivelError, match="Nenhuma câmera"):
        abrir_webcam(0)
    assert captura.liberada


def test_webcam_abre_mas_nao_envia_imagem(monkeypatch: pytest.MonkeyPatch) -> None:
    captura = CapturaFalsa(frames=0)
    _instalar_captura(monkeypatch, captura)
    with pytest.raises(WebcamIndisponivelError, match="não enviou imagens"):
        abrir_webcam(1)
    assert captura.liberada


def test_webcam_funcionando_usa_fps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _instalar_captura(monkeypatch, CapturaFalsa(frames=3, fps=0.0))
    with abrir_webcam(0, AquisicaoConfig(fps_fallback=25.0)) as fonte:
        assert fonte.info.ao_vivo
        assert fonte.info.total_frames is None and fonte.info.duracao_s is None
        assert fonte.info.fps == 25.0 and fonte.info.fps_estimado
        frames = list(fonte)
    assert len(frames) == 3
    tempos = [f.timestamp_s for f in frames]
    assert tempos == sorted(tempos) and tempos[0] >= 0


@pytest.mark.parametrize("indice", [-1, 1.5, "0"])
def test_indice_de_webcam_invalido(indice: object) -> None:
    with pytest.raises(WebcamIndisponivelError, match="inválido"):
        abrir_webcam(indice)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", ["", "   ", "ftp://x.com/v.mp4", "www.site.com/v.mp4", "http://"])
def test_url_invalida(url: str) -> None:
    with pytest.raises(FonteInvalidaError):
        validar_url(url)


@pytest.mark.parametrize("url", ["https://exemplo.com/v.mp4", " rtsp://10.0.0.5:554/stream "])
def test_url_valida(url: str) -> None:
    assert validar_url(url) == url.strip()


# ---------------------------------------------------------------------------
# Vídeos de demonstração e upload
# ---------------------------------------------------------------------------
def test_listar_videos_demo(tmp_path: Path) -> None:
    for nome in ["b.mp4", "A.MOV", "leia.txt", "c.avi"]:
        (tmp_path / nome).write_bytes(b"x")
    (tmp_path / "subpasta.mp4").mkdir()
    assert [p.name for p in listar_videos_demo(tmp_path)] == ["A.MOV", "b.mp4", "c.avi"]


def test_listar_videos_demo_pasta_inexistente(tmp_path: Path) -> None:
    assert listar_videos_demo(tmp_path / "nao_existe") == []


def test_upload_salvo_e_reaberto(fabrica_video, tmp_path: Path) -> None:
    dados = fabrica_video(n_frames=6).read_bytes()
    destino = tmp_path / "uploads"
    caminho = salvar_upload(dados, "Câmera da Avenida #1.AVI", destino=destino)
    assert caminho.parent == destino
    assert caminho.suffix == ".avi"
    assert caminho.name.isascii() and " " not in caminho.name
    with abrir_arquivo(caminho) as fonte:
        assert fonte.info.total_frames == 6


def test_upload_invalido_nao_grava_nada(tmp_path: Path) -> None:
    destino = tmp_path / "uploads"
    with pytest.raises(ArquivoInvalidoError):
        salvar_upload(b"conteudo", "planilha.xlsx", destino=destino)
    with pytest.raises(ArquivoInvalidoError, match="vazio"):
        salvar_upload(b"", "video.mp4", destino=destino)
    assert not destino.exists()


# ---------------------------------------------------------------------------
# Imagem única
# ---------------------------------------------------------------------------
def _png(largura: int, altura: int) -> bytes:
    ok, buffer = cv2.imencode(".png", np.full((altura, largura, 3), 128, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_decodifica_imagem_valida() -> None:
    imagem = decodificar_imagem(_png(640, 480), "rua.png")
    assert imagem.shape == (480, 640, 3)


def test_imagem_corrompida() -> None:
    with pytest.raises(ArquivoInvalidoError, match="Não foi possível ler"):
        decodificar_imagem(b"isto nao e uma imagem", "rua.jpg")


def test_imagem_pequena_demais() -> None:
    with pytest.raises(ArquivoInvalidoError, match="Resolução"):
        decodificar_imagem(_png(100, 80), "mini.png")


def test_imagem_com_extensao_de_video() -> None:
    with pytest.raises(ArquivoInvalidoError, match="não suportado"):
        decodificar_imagem(_png(640, 480), "rua.mp4")


def test_carregar_imagem_de_caminho_acentuado(tmp_path: Path) -> None:
    caminho = tmp_path / "interseção.png"
    caminho.write_bytes(_png(640, 480))
    assert carregar_imagem(caminho).shape == (480, 640, 3)
