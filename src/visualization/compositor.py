"""Composição por camadas do frame exibido.

Ordem (de baixo para cima)::

    frame reduzido p/ exibição → mapa de calor → ROI → linha de contagem
        → caixas → trajetórias → miniatura bird's-eye → painel HUD

As trajetórias ficam acima das caixas porque o ponto de referência é o
centro da base da caixa: num movimento horizontal, a borda inferior da
caixa esconderia a trajetória.

Cada camada liga/desliga em ``VisualizacaoConfig``. O frame é reduzido para
a largura de exibição **antes** de desenhar: é mais barato e o traço fica com
espessura constante na tela. Todas as coordenadas passam por
``original → exibição`` (a etapa final do mapeamento modelo → original → interface).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from src.config import CLASSES_VEICULO, AppConfig
from src.geometry.coordenadas import escala_exibicao, normalizados_para_pixels
from src.geometry.erros import ErroGeometria
from src.geometry.roi import roi_em_pixels
from src.metrics.desempenho import MedidorFPS
from src.models.analisador import ResultadoFrame, VeiculoRastreado
from src.models.contagem import SENTIDOS, ContadorLinha
from src.models.rastreamento import HistoricoTrajetorias
from src.visualization.birdseye import VistaBirdseye, colar_miniatura
from src.visualization.camadas import (
    desenhar_caixas,
    desenhar_hud,
    desenhar_linha_contagem,
    desenhar_roi,
    desenhar_trajetorias,
    linha_piscando,
)
from src.visualization.desenho import BRANCO, escrever
from src.visualization.mapa_calor import MapaCalor

logger = logging.getLogger(__name__)


def contagem_estatica(veiculos: Iterable[VeiculoRastreado]) -> dict[str, dict[str, int]]:
    """Tabela do HUD para imagem única: só a coluna "total" (não há sentido sem movimento)."""
    tabela = {nome: {**{s: 0 for s in SENTIDOS}, "total": 0} for nome in CLASSES_VEICULO.values()}
    for v in veiculos:
        tabela[v.nome_classe]["total"] += 1
    return tabela


class CompositorCamadas:
    """Mantém o estado da visualização de um vídeo (mapa de calor, FPS, bird's-eye)."""

    def __init__(self, cfg: AppConfig, largura: int, altura: int) -> None:
        self.cfg = cfg
        vis, geo = cfg.visualizacao, cfg.geometria
        self.largura, self.altura = largura, altura
        self.escala = escala_exibicao(largura, vis.largura_exibicao)
        self.tamanho_exibicao = (max(1, round(largura * self.escala)), max(1, round(altura * self.escala)))
        self.mapa_calor = MapaCalor(largura, altura)
        self.medidor_fps = MedidorFPS()
        self.tempo_desenho_ms = 0.0

        linha = normalizados_para_pixels([cfg.contagem.linha_p1, cfg.contagem.linha_p2], largura, altura)
        self.linha_px: tuple[Sequence[float], Sequence[float]] = (linha[0], linha[1])
        self.roi_px = self._roi()
        self.birdseye = self._birdseye()

    def _roi(self) -> np.ndarray | None:
        if not self.cfg.preproc.usar_roi:
            return None
        try:
            return roi_em_pixels(self.cfg.geometria.roi, self.largura, self.altura)
        except ErroGeometria:
            return None  # o analisador já avisa o usuário

    def _birdseye(self) -> VistaBirdseye | None:
        geo = self.cfg.geometria
        try:
            return VistaBirdseye(
                normalizados_para_pixels(geo.homografia_origem, self.largura, self.altura),
                geo.largura_pista_m,
                geo.comprimento_pista_m,
                altura_px=int(self.cfg.visualizacao.altura_birdseye * self.tamanho_exibicao[1]),
            )
        except ErroGeometria as erro:
            logger.info("Miniatura bird's-eye desativada: %s", erro)
            return None

    # ------------------------------------------------------------------
    def compor(
        self,
        imagem: np.ndarray,
        resultado: ResultadoFrame,
        contador: ContadorLinha | None = None,
        historico: HistoricoTrajetorias | None = None,
    ) -> np.ndarray:
        """Desenha todas as camadas ligadas e devolve o frame de exibição (BGR).

        Args:
            imagem: frame **original** (não é alterado).
            resultado: saída do analisador para este frame.
            contador / historico: estado acumulado do vídeo (``None`` para imagem única).
        """
        inicio = time.perf_counter()
        vis = self.cfg.visualizacao
        if resultado.processado:
            self.mapa_calor.acumular(v.caixa for v in resultado.veiculos)

        if self.escala == 1.0:
            saida = imagem.copy()
        else:
            saida = cv2.resize(imagem, self.tamanho_exibicao, interpolation=cv2.INTER_AREA)

        if vis.mostrar_heatmap:
            self.mapa_calor.sobrepor(saida, vis.opacidade_heatmap, vis.colormap)
        if vis.mostrar_roi and self.roi_px is not None:
            desenhar_roi(saida, self.roi_px, self.escala, vis.opacidade_roi)
        if vis.mostrar_linha:
            p1, p2 = (contador.p1, contador.p2) if contador else self.linha_px
            ativa = contador is not None and linha_piscando(
                resultado.frame, contador.frame_ultimo_cruzamento, vis.duracao_piscar_frames
            )
            desenhar_linha_contagem(saida, p1, p2, self.escala, ativa)
        if vis.mostrar_caixas:
            desenhar_caixas(saida, resultado.veiculos, self.escala)
        if vis.mostrar_trajetorias and historico is not None:
            recentes = max(5, 2 * (self.cfg.preproc.frame_skip + 1))
            ativas = [t for t in historico if resultado.frame - t.ultimo_frame <= recentes]
            desenhar_trajetorias(saida, ativas, self.escala)
        if vis.mostrar_birdseye and self.birdseye is not None:
            colar_miniatura(saida, self.birdseye.renderizar(imagem, resultado.veiculos))

        self.tempo_desenho_ms = (time.perf_counter() - inicio) * 1000
        fps = self.medidor_fps.registrar()
        if vis.mostrar_hud:
            tabela = contador.por_classe() if contador else contagem_estatica(resultado.veiculos)
            latencia = resultado.tempo_total_ms + self.tempo_desenho_ms if resultado.processado else None
            desenhar_hud(saida, tabela, fps, latencia)
        return saida

    def reiniciar(self) -> None:
        self.mapa_calor.reiniciar()
        self.medidor_fps.reiniciar()


# ---------------------------------------------------------------------------
# Comparação original × pré-processado × final
# ---------------------------------------------------------------------------
def _ajustar_altura(imagem: np.ndarray, altura: int) -> np.ndarray:
    escala = altura / imagem.shape[0]
    largura = max(1, round(imagem.shape[1] * escala))
    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    return cv2.resize(imagem, (largura, altura), interpolation=interpolacao)


def painel_comparacao(
    original: np.ndarray,
    preprocessado: np.ndarray,
    final: np.ndarray,
    altura: int = 320,
    titulos: Sequence[str] = ("Original", "Pre-processado (entrada do modelo)", "Resultado"),
) -> np.ndarray:
    """Três imagens lado a lado, na mesma altura, com títulos e separadores."""
    faixa_titulo, separador = 28, 6
    paineis = []
    for imagem, titulo in zip((original, preprocessado, final), titulos):
        ajustada = _ajustar_altura(imagem, altura)
        cabecalho = np.full((faixa_titulo, ajustada.shape[1], 3), 30, dtype=np.uint8)
        escrever(cabecalho, titulo, (8, 19), 0.5, BRANCO, fundo=None)
        paineis.append(np.vstack((cabecalho, ajustada)))
    barra = np.full((altura + faixa_titulo, separador, 3), 255, dtype=np.uint8)
    partes: list[np.ndarray] = []
    for i, painel in enumerate(paineis):
        if i:
            partes.append(barra)
        partes.append(painel)
    return np.hstack(partes)
