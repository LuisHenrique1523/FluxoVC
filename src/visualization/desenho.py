"""Primitivas de desenho: cores, texto com fundo e escala de traço.

Observação sobre acentos: ``cv2.putText`` usa fontes Hershey, que só têm
caracteres ASCII ("ô" vira "?"). Desenhar texto com Pillow exigiria converter
o frame inteiro a cada quadro, o que custa caro em tempo real; por isso o texto
**sobre o vídeo** é escrito sem acentos (:func:`texto_ascii`). A interface
Streamlit continua com acentuação normal.
"""

from __future__ import annotations

import colorsys
import unicodedata
from functools import lru_cache

import cv2
import numpy as np

from src.config import CORES_CLASSE

Cor = tuple[int, int, int]  # BGR

FONTE = cv2.FONT_HERSHEY_SIMPLEX
BRANCO: Cor = (255, 255, 255)
PRETO: Cor = (0, 0, 0)
COR_PADRAO: Cor = (200, 200, 200)
RAZAO_AUREA = 0.618033988749895


def texto_ascii(texto: str) -> str:
    """Remove acentos e troca setas para caber nas fontes do OpenCV."""
    texto = texto.replace("→", "->").replace("←", "<-").replace("×", "x")
    normalizado = unicodedata.normalize("NFKD", texto)
    return normalizado.encode("ascii", "ignore").decode("ascii")


def cor_classe(classe: int) -> Cor:
    """Cor fixa da classe COCO (carro azul, moto laranja, ônibus verde, caminhão vermelho)."""
    return CORES_CLASSE.get(classe, COR_PADRAO)


@lru_cache(maxsize=512)
def cor_id(id_rastreio: int) -> Cor:
    """Cor estável e bem distinta por ID: matiz avançando pela razão áurea."""
    matiz = (id_rastreio * RAZAO_AUREA) % 1.0
    r, g, b = colorsys.hsv_to_rgb(matiz, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def espessura(largura_imagem: int, base: float = 1.0) -> int:
    """Espessura de traço proporcional à largura da imagem (2 px em 960 px)."""
    return max(1, round(base * largura_imagem / 480))


def escrever(
    imagem: np.ndarray,
    texto: str,
    origem: tuple[int, int],
    escala: float = 0.5,
    cor: Cor = BRANCO,
    fundo: Cor | None = PRETO,
    espessura_texto: int = 1,
    margem: int = 3,
) -> tuple[int, int]:
    """Escreve ``texto`` com a base esquerda em ``origem``, com retângulo de fundo opcional.

    Returns:
        ``(largura, altura)`` do texto desenhado, incluindo a margem.
    """
    texto = texto_ascii(texto)
    (largura, altura), linha_base = cv2.getTextSize(texto, FONTE, escala, espessura_texto)
    x, y = int(origem[0]), int(origem[1])
    if fundo is not None:
        cv2.rectangle(
            imagem,
            (x - margem, y - altura - margem),
            (x + largura + margem, y + linha_base + margem - 1),
            fundo,
            -1,
        )
    cv2.putText(imagem, texto, (x, y), FONTE, escala, cor, espessura_texto, cv2.LINE_AA)
    return largura + 2 * margem, altura + linha_base + 2 * margem


def misturar(base: np.ndarray, camada: np.ndarray, opacidade: float, mascara: np.ndarray | None = None) -> None:
    """Mistura ``camada`` sobre ``base`` (in-place): ``base = (1−α)·base + α·camada``.

    Com ``mascara`` (bool ou uint8, mesma altura × largura), só os pixels marcados
    mudam. ``cv2.copyTo`` com máscara é ~10× mais rápido que indexação booleana
    do NumPy em imagens grandes.
    """
    opacidade = float(np.clip(opacidade, 0.0, 1.0))
    misturada = cv2.addWeighted(camada, opacidade, base, 1.0 - opacidade, 0.0)
    if mascara is None:
        base[:] = misturada
    else:
        cv2.copyTo(misturada, mascara.astype(np.uint8, copy=False), base)
