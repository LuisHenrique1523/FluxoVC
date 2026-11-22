"""Operações de pré-processamento, cada uma isolada e testável.

Todas recebem e devolvem imagens BGR ``uint8`` e **não alteram** a imagem de
entrada (devolvem uma nova).
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from src.geometry.coordenadas import ParametrosLetterbox, calcular_letterbox

COR_PREENCHIMENTO: tuple[int, int, int] = (114, 114, 114)  # cinza padrão do YOLO
FILTROS_RUIDO: tuple[str, ...] = ("bilateral", "gaussiano")
SIGMA_COR_BILATERAL: float = 75.0
SIGMA_ESPACO_BILATERAL: float = 75.0


# ---------------------------------------------------------------------------
# 1. Redimensionamento com letterbox
# ---------------------------------------------------------------------------
def aplicar_letterbox(
    imagem: np.ndarray,
    tamanho: int,
    manter_proporcao: bool = True,
    cor: tuple[int, int, int] = COR_PREENCHIMENTO,
) -> tuple[np.ndarray, ParametrosLetterbox]:
    """Redimensiona para ``tamanho × tamanho`` mantendo a proporção (faixas nas sobras).

    Returns:
        A imagem quadrada e os parâmetros necessários para desfazer o letterbox
        nas coordenadas das detecções.
    """
    altura, largura = imagem.shape[:2]
    p = calcular_letterbox(largura, altura, tamanho, manter_proporcao)
    reduzindo = p.escala_x < 1 or p.escala_y < 1
    # INTER_AREA evita serrilhado ao reduzir; INTER_LINEAR é melhor ao ampliar.
    interpolacao = cv2.INTER_AREA if reduzindo else cv2.INTER_LINEAR
    redimensionada = cv2.resize(imagem, (p.nova_largura, p.nova_altura), interpolation=interpolacao)

    saida = np.full((tamanho, tamanho, 3), cor, dtype=np.uint8)
    saida[p.pad_y : p.pad_y + p.nova_altura, p.pad_x : p.pad_x + p.nova_largura] = redimensionada
    return saida, p


# ---------------------------------------------------------------------------
# 2. Máscara de ROI
# ---------------------------------------------------------------------------
def criar_mascara_roi(forma: tuple[int, int], poligono_px: np.ndarray) -> np.ndarray:
    """Máscara ``uint8`` (255 dentro do polígono, 0 fora) com a ``forma`` (altura, largura)."""
    mascara = np.zeros(forma[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [np.asarray(poligono_px, dtype=np.int32).reshape(-1, 1, 2)], 255)
    return mascara


def aplicar_mascara(
    imagem: np.ndarray, mascara: np.ndarray, cor: tuple[int, int, int] = COR_PREENCHIMENTO
) -> np.ndarray:
    """Pinta com ``cor`` tudo o que está fora da máscara (o modelo passa a ignorar essa área)."""
    saida = imagem.copy()
    saida[mascara == 0] = cor
    return saida


# ---------------------------------------------------------------------------
# 3. CLAHE no canal L (LAB)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _clahe(clip: float, grade: int) -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grade, grade))


def aplicar_clahe(imagem: np.ndarray, clip: float = 2.0, grade: int = 8) -> np.ndarray:
    """Equalização adaptativa de contraste só na luminância.

    Converter para LAB e equalizar apenas o canal L melhora cenas escuras ou
    em contraluz sem distorcer as cores (equalizar R, G e B separadamente
    mudaria a tonalidade dos veículos).

    Raises:
        ValueError: se ``clip`` ou ``grade`` não forem positivos.
    """
    if clip <= 0 or grade < 1:
        raise ValueError("Parâmetros do CLAHE devem ser positivos.")
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    luminancia, canal_a, canal_b = cv2.split(lab)
    luminancia = _clahe(float(clip), int(grade)).apply(luminancia)
    return cv2.cvtColor(cv2.merge((luminancia, canal_a, canal_b)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 4. Redução de ruído
# ---------------------------------------------------------------------------
def kernel_impar(kernel: int) -> int:
    """Os filtros exigem kernel ímpar ≥ 3; arredonda para o ímpar seguinte."""
    k = max(3, int(kernel))
    return k if k % 2 == 1 else k + 1


def reduzir_ruido(imagem: np.ndarray, tipo: str = "bilateral", kernel: int = 5) -> np.ndarray:
    """Suaviza ruído de sensor/compressão.

    * ``bilateral``: média ponderada pela distância **e** pela diferença de cor;
      preserva as bordas dos veículos. Mais lento.
    * ``gaussiano``: média ponderada só pela distância; rápido, mas borra bordas.

    Raises:
        ValueError: tipo de filtro desconhecido.
    """
    k = kernel_impar(kernel)
    if tipo == "bilateral":
        return cv2.bilateralFilter(imagem, k, SIGMA_COR_BILATERAL, SIGMA_ESPACO_BILATERAL)
    if tipo == "gaussiano":
        return cv2.GaussianBlur(imagem, (k, k), 0)
    raise ValueError(f"Filtro '{tipo}' desconhecido. Use: {', '.join(FILTROS_RUIDO)}.")


# ---------------------------------------------------------------------------
# 5. Frame skipping (opcional)
# ---------------------------------------------------------------------------
def deve_processar_frame(indice: int, frame_skip: int) -> bool:
    """Com ``frame_skip = n``, processa 1 frame e pula os ``n`` seguintes."""
    if frame_skip < 0:
        raise ValueError("frame_skip não pode ser negativo.")
    return indice % (frame_skip + 1) == 0
