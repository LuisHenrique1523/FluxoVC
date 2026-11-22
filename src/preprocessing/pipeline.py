"""Encadeamento do pré-processamento conforme os toggles da configuração.

Ordem: **letterbox → CLAHE → redução de ruído → máscara de ROI**.

O letterbox vem primeiro para que as demais operações trabalhem sempre num
quadrado de ``tamanho_entrada`` pixels (ex.: 640×640), com custo constante
mesmo para vídeos 4K. A ROI vem por último para que o contraste e o filtro
não "vazem" a borda artificial da máscara para dentro da pista. A imagem
resultante é exatamente a que o modelo recebe — é ela que a interface mostra
na comparação original × pré-processado × final.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.config import PontoNorm, PreprocConfig
from src.geometry.coordenadas import ParametrosLetterbox, original_para_modelo
from src.geometry.roi import roi_em_pixels
from src.preprocessing.operacoes import (
    aplicar_clahe,
    aplicar_letterbox,
    aplicar_mascara,
    criar_mascara_roi,
    reduzir_ruido,
)

# Textos exibidos na interface ao lado de cada toggle (e resumidos em docs/pipeline.md).
EXPLICACOES: dict[str, str] = {
    "letterbox": (
        "Redimensiona o frame para o quadrado de entrada do modelo mantendo a proporção; "
        "as sobras viram faixas cinzas. Sem isso, os veículos ficam esticados e o detector erra mais."
    ),
    "roi": (
        "Pinta de cinza tudo o que está fora do polígono da pista, para o modelo ignorar "
        "calçadas, estacionamentos e outras vias."
    ),
    "clahe": (
        "Equaliza o contraste por regiões, só na luminância (canal L do espaço LAB). "
        "Ajuda em cenas escuras, com sombra ou contraluz, sem alterar as cores."
    ),
    "ruido": (
        "Suaviza o ruído do sensor e da compressão. O filtro bilateral preserva as bordas; "
        "o gaussiano é mais rápido, mas borra mais."
    ),
    "frame_skip": (
        "Processa 1 frame e pula os N seguintes. Aumenta o FPS, mas pode perder veículos rápidos "
        "e piora o rastreamento."
    ),
}


@dataclass
class ResultadoPreproc:
    """Saída do pré-processamento de um frame."""

    imagem: np.ndarray                  # entrada do modelo (tamanho × tamanho, BGR)
    letterbox: ParametrosLetterbox      # para desfazer o letterbox nas detecções
    roi_modelo: np.ndarray | None       # polígono da ROI no espaço do modelo (px), se aplicada
    tempos_ms: dict[str, float] = field(default_factory=dict)  # custo de cada etapa

    @property
    def tempo_total_ms(self) -> float:
        return sum(self.tempos_ms.values())


def preprocessar(
    imagem: np.ndarray,
    cfg: PreprocConfig,
    roi_norm: Sequence[PontoNorm] | None = None,
) -> ResultadoPreproc:
    """Aplica as operações ligadas em ``cfg`` ao frame original.

    Args:
        imagem: frame original BGR.
        cfg: toggles e parâmetros do pré-processamento.
        roi_norm: polígono da ROI em coordenadas normalizadas (usado se ``cfg.usar_roi``).

    Raises:
        RoiInvalidaError: se a ROI estiver ligada e o polígono for inválido.
    """
    tempos: dict[str, float] = {}

    inicio = time.perf_counter()
    saida, params = aplicar_letterbox(imagem, cfg.tamanho_entrada, cfg.manter_proporcao)
    tempos["letterbox"] = (time.perf_counter() - inicio) * 1000

    if cfg.usar_clahe:
        inicio = time.perf_counter()
        saida = aplicar_clahe(saida, cfg.clahe_clip, cfg.clahe_grade)
        tempos["clahe"] = (time.perf_counter() - inicio) * 1000

    if cfg.usar_reducao_ruido:
        inicio = time.perf_counter()
        saida = reduzir_ruido(saida, cfg.tipo_filtro, cfg.kernel_ruido)
        tempos["ruido"] = (time.perf_counter() - inicio) * 1000

    roi_modelo: np.ndarray | None = None
    if cfg.usar_roi and roi_norm is not None:
        inicio = time.perf_counter()
        roi_original = roi_em_pixels(roi_norm, params.largura_orig, params.altura_orig)
        roi_modelo = np.round(original_para_modelo(roi_original, params)).astype(np.int32)
        saida = aplicar_mascara(saida, criar_mascara_roi(saida.shape[:2], roi_modelo))
        tempos["roi"] = (time.perf_counter() - inicio) * 1000

    return ResultadoPreproc(saida, params, roi_modelo, tempos)
