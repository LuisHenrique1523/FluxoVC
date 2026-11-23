"""Sessão de análise: liga fonte → analisador → compositor → gravador, frame a frame.

A interface cria uma :class:`SessaoAnalise` ao clicar em "Analisar" e chama
:meth:`SessaoAnalise.passo` num laço, atualizando a tela a cada frame. Como a
sessão fica guardada no estado do Streamlit, o laço pode ser interrompido
(botão "Parar" ou mudança de parâmetro) e retomado sem perder nada.

Parâmetros ao vivo × fixos: confiança, IoU, pré-processamento e camadas são
**compartilhados** com a configuração da interface (mudam durante a análise).
Linha, ROI, calibração e α da suavização são **copiados** no início, para que
uma mesma análise não misture geometrias diferentes.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.acquisition.fontes import FonteVideo, Frame, abrir_arquivo
from src.config import (
    AVISO_CALIBRACAO_PADRAO,
    OUTPUTS_DIR,
    AppConfig,
    VisualizacaoConfig,
    calibracao_e_padrao,
)
from src.geometry.coordenadas import normalizados_para_pixels
from src.geometry.erros import ErroGeometria
from src.geometry.homografia import calcular_homografia
from src.geometry.roi import roi_em_pixels
from src.models.analisador import AnalisadorFluxo, ResultadoFrame, VeiculoRastreado, analisar_imagem
from src.models.contagem import ContadorLinha, EventoCruzamento
from src.models.tipos import Detector
from src.preprocessing.pipeline import ResultadoPreproc
from src.visualization.camadas import desenhar_calibracao
from src.visualization.compositor import CompositorCamadas
from src.visualization.gravador import GravadorVideo

logger = logging.getLogger(__name__)

PASTA_VIDEOS: Path = OUTPUTS_DIR / "videos"
_PASTA_PADRAO = object()  # sentinela: usa PASTA_VIDEOS no momento da chamada


class ErroAnalise(RuntimeError):
    """Falha inesperada no meio da análise (a sessão já foi encerrada e o vídeo salvo)."""


def config_da_sessao(cfg: AppConfig) -> AppConfig:
    """Cópia da configuração: compartilha os parâmetros "ao vivo" e congela a geometria."""
    return AppConfig(
        aquisicao=cfg.aquisicao,
        preproc=cfg.preproc,
        modelo=cfg.modelo,
        contagem=copy.deepcopy(cfg.contagem),
        geometria=copy.deepcopy(cfg.geometria),
        visualizacao=cfg.visualizacao,
    )


@dataclass
class PassoAnalise:
    """O que a tela precisa mostrar de um frame."""

    frame: Frame
    resultado: ResultadoFrame
    imagem_final: np.ndarray        # BGR, já no tamanho de exibição


class SessaoAnalise:
    """Análise de um vídeo (arquivo, demonstração, webcam ou URL)."""

    def __init__(
        self,
        fonte: FonteVideo,
        cfg: AppConfig,
        detector: Detector,
        pasta_saida: Path | None | object = _PASTA_PADRAO,
    ) -> None:
        """
        Args:
            pasta_saida: onde gravar o vídeo processado (``None`` = não gravar).

        Raises:
            RoiInvalidaError / ErroGeometria: configuração geométrica inválida
                (a fonte é liberada antes de relançar).
        """
        self.fonte = fonte
        self.info = fonte.info
        self.cfg = config_da_sessao(cfg)
        try:
            self.analisador = AnalisadorFluxo(self.cfg, detector, self.info.largura, self.info.altura)
        except Exception:
            fonte.liberar()
            raise
        self.compositor = CompositorCamadas(self.cfg, self.info.largura, self.info.altura)
        self.inicio = datetime.now()
        pasta = PASTA_VIDEOS if pasta_saida is _PASTA_PADRAO else pasta_saida
        self.gravador = (
            GravadorVideo(Path(pasta) / f"fluxovc_{self.inicio:%Y%m%d_%H%M%S}.mp4", self.info.fps)  # type: ignore[arg-type]
            if pasta is not None
            else None
        )
        self.tempos: list[dict[str, float]] = []
        self.ultimo_passo: PassoAnalise | None = None
        self.ultimo_preproc: ResultadoPreproc | None = None
        self.caminho_video: Path | None = None
        self.encerrada = False
        self.fonte_terminou = False     # True = a fonte acabou (fim do arquivo ou câmera desconectada)
        self.erro: str | None = None    # mensagem da falha que interrompeu a análise
        self.avisos: list[str] = []     # problemas não fatais (ex.: gravação do vídeo)
        self._relogio_inicio: float | None = None
        self._relogio_fim: float | None = None

    # ------------------------------------------------------------------
    def passo(self) -> PassoAnalise | None:
        """Processa o próximo frame. Devolve ``None`` quando a fonte acaba (e encerra).

        Raises:
            ErroAnalise: falha inesperada ao ler, analisar ou desenhar o frame. A
                sessão é encerrada antes (fonte liberada, vídeo parcial salvo), e o
                que já foi contado continua disponível.
        """
        if self.encerrada:
            return None
        if self._relogio_inicio is None:
            self._relogio_inicio = time.perf_counter()
        try:
            frame = self.fonte.ler_frame()
            if frame is None:
                self.fonte_terminou = True
                self.finalizar()
                return None
            resultado = self.analisador.processar(frame)
            final = self.compositor.compor(
                frame.imagem, resultado, self.analisador.contador, self.analisador.historico
            )
        except Exception as erro:
            logger.exception("Falha ao processar o frame %d", self.fonte.frames_lidos)
            self.erro = (
                f"A análise foi interrompida no frame {self.fonte.frames_lidos} por um erro "
                f"inesperado ({type(erro).__name__}: {erro}). O que já foi contado foi mantido."
            )
            self.finalizar()
            raise ErroAnalise(self.erro) from erro
        self._gravar(final)
        if resultado.processado:
            self.ultimo_preproc = resultado.preproc
            self.tempos.append({**resultado.tempos_ms, "desenho": self.compositor.tempo_desenho_ms})
        self._relogio_fim = time.perf_counter()
        self.ultimo_passo = PassoAnalise(frame, resultado, final)
        return self.ultimo_passo

    def _gravar(self, imagem: np.ndarray) -> None:
        """Grava o frame no MP4. Se a gravação falhar, a análise continua sem o vídeo."""
        if self.gravador is None:
            return
        try:
            self.gravador.escrever(imagem)
        except (OSError, RuntimeError, cv2.error) as erro:
            aviso = f"O vídeo processado não será salvo: {erro}"
            logger.warning(aviso)
            self.avisos.append(aviso)
            try:
                self.gravador.fechar()
            except Exception:  # pragma: no cover - só limpeza
                pass
            self.gravador = None

    @property
    def ao_vivo(self) -> bool:
        """Webcam ou stream (sem fim previsto)."""
        return bool(self.info.ao_vivo)

    def finalizar(self) -> None:
        """Libera a fonte e fecha o vídeo gravado (pode ser chamado mais de uma vez)."""
        if self.encerrada:
            return
        self.encerrada = True
        self.fonte.liberar()
        if self.gravador is not None:
            self.caminho_video = self.gravador.fechar()
        logger.info(
            "Análise encerrada: %d frames, %d veículos contados.",
            self.fonte.frames_lidos, self.analisador.contador.total,
        )

    # ------------------------------------------------------------------
    @property
    def contador(self) -> ContadorLinha:
        return self.analisador.contador

    @property
    def eventos(self) -> list[EventoCruzamento]:
        return self.analisador.contador.eventos

    @property
    def frames_lidos(self) -> int:
        return self.fonte.frames_lidos

    @property
    def progresso(self) -> float | None:
        """Fração do vídeo já lida (``None`` para webcam/stream)."""
        total = self.info.total_frames
        return None if not total else min(1.0, self.frames_lidos / total)

    @property
    def duracao_processada_s(self) -> float:
        passo = self.ultimo_passo
        return 0.0 if passo is None else passo.frame.timestamp_s

    @property
    def video_no_navegador(self) -> bool:
        return self.gravador is not None and self.gravador.reproduz_no_navegador

    def tempos_dataframe(self) -> pd.DataFrame:
        """Tempo (ms) de cada etapa em cada frame processado."""
        return pd.DataFrame(self.tempos)

    def fps_medio(self) -> float | None:
        """Frames por segundo efetivos (todos os frames lidos / tempo de parede)."""
        if self._relogio_inicio is None or self._relogio_fim is None or self.frames_lidos < 2:
            return None
        decorrido = self._relogio_fim - self._relogio_inicio
        return self.frames_lidos / decorrido if decorrido > 0 else None


# ---------------------------------------------------------------------------
# Imagem única e prévia da configuração
# ---------------------------------------------------------------------------
def analisar_e_compor_imagem(
    imagem: np.ndarray, cfg: AppConfig, detector: Detector
) -> tuple[list[VeiculoRastreado], ResultadoPreproc, np.ndarray]:
    """Analisa uma imagem única e devolve (veículos, pré-processamento, imagem final)."""
    veiculos, preproc = analisar_imagem(imagem, cfg, detector)
    altura, largura = imagem.shape[:2]
    compositor = CompositorCamadas(cfg, largura, altura)
    resultado = ResultadoFrame(0, 0.0, True, veiculos, [], preproc, dict(preproc.tempos_ms))
    return veiculos, preproc, compositor.compor(imagem, resultado)


def validar_configuracao(cfg: AppConfig, largura: int, altura: int) -> tuple[list[str], list[str]]:
    """Confere linha, ROI e calibração antes de analisar.

    Returns:
        ``(erros, avisos)``: erros impedem a análise; avisos só desativam a velocidade.
    """
    erros: list[str] = []
    avisos: list[str] = []
    linha = normalizados_para_pixels([cfg.contagem.linha_p1, cfg.contagem.linha_p2], largura, altura)
    try:
        ContadorLinha(linha[0], linha[1])
    except ErroGeometria as erro:
        erros.append(str(erro))
    if cfg.preproc.usar_roi:
        try:
            roi_em_pixels(cfg.geometria.roi, largura, altura)
        except ErroGeometria as erro:
            erros.append(str(erro))
    geo = cfg.geometria
    try:
        calcular_homografia(
            normalizados_para_pixels(geo.homografia_origem, largura, altura),
            geo.largura_pista_m, geo.comprimento_pista_m,
        )
    except ErroGeometria as erro:
        avisos.append(f"Velocidade será desativada: {erro}")
    else:
        if calibracao_e_padrao(geo):
            avisos.append(AVISO_CALIBRACAO_PADRAO)
    return erros, avisos


def previa_configuracao(imagem: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Frame com ROI, linha de contagem e quadrilátero de calibração, para conferir antes de analisar."""
    altura, largura = imagem.shape[:2]
    cfg_previa = copy.deepcopy(cfg)
    cfg_previa.visualizacao = VisualizacaoConfig(
        mostrar_caixas=False, mostrar_trajetorias=False, mostrar_heatmap=False,
        mostrar_birdseye=False, mostrar_hud=False, mostrar_linha=True, mostrar_roi=True,
        opacidade_roi=cfg.visualizacao.opacidade_roi,
        largura_exibicao=cfg.visualizacao.largura_exibicao,
    )
    compositor = CompositorCamadas(cfg_previa, largura, altura)
    saida = compositor.compor(imagem, ResultadoFrame(0, 0.0, False))
    pontos = normalizados_para_pixels(cfg.geometria.homografia_origem, largura, altura)
    if len(pontos) == 4:
        desenhar_calibracao(saida, pontos, compositor.escala)
    return saida


def primeiro_frame(caminho: Path, cfg: AppConfig) -> np.ndarray:
    """Lê só o primeiro frame de um vídeo (para a prévia).

    Raises:
        ArquivoInvalidoError: vídeo inválido.
    """
    with abrir_arquivo(caminho, cfg.aquisicao) as fonte:
        frame = fonte.ler_frame()
    assert frame is not None  # abrir_arquivo já garante ao menos um frame
    return frame.imagem


def codificar_png(imagem: np.ndarray) -> bytes:
    """Imagem BGR → bytes PNG (download)."""
    ok, buffer = cv2.imencode(".png", imagem)
    if not ok:
        raise ValueError("Não foi possível codificar a imagem.")
    return buffer.tobytes()
