"""Mapeamento de coordenadas entre os três espaços da aplicação.

* **Original** — pixels do frame como saiu da câmera/arquivo (ex.: 1920×1080).
* **Modelo** — quadrado ``tamanho × tamanho`` (ex.: 640×640) gerado pelo
  letterbox: a imagem é reduzida mantendo a proporção e centralizada, e as
  sobras viram faixas de preenchimento.
* **Exibição** — o frame original reduzido para caber na interface.

E ainda as coordenadas **normalizadas** (0 a 1), usadas na configuração da
linha, da ROI e da homografia para valerem em qualquer resolução.

Convenção: coordenadas contínuas, em que o pixel de índice ``i`` ocupa o
intervalo ``[i, i+1)``. Assim, as caixas ``(x1, y1, x2, y2)`` do detector e os
pontos convertem-se pela mesma fórmula afim:

    x_modelo = x_original · escala_x + pad_x
    x_original = (x_modelo − pad_x) / escala_x
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ParametrosLetterbox:
    """Tudo o que é preciso para ir e voltar entre o espaço original e o do modelo."""

    largura_orig: int
    altura_orig: int
    tamanho: int            # lado do quadrado de entrada do modelo
    escala_x: float         # nova_largura / largura_orig
    escala_y: float         # nova_altura / altura_orig
    nova_largura: int       # largura da imagem redimensionada (sem as faixas)
    nova_altura: int
    pad_x: int              # faixa à esquerda
    pad_y: int              # faixa no topo


def calcular_letterbox(
    largura: int, altura: int, tamanho: int, manter_proporcao: bool = True
) -> ParametrosLetterbox:
    """Calcula escala e faixas para levar um frame ``largura × altura`` ao quadrado ``tamanho``.

    Com ``manter_proporcao=True`` usa a mesma escala nos dois eixos
    (``min(tamanho/largura, tamanho/altura)``) e centraliza a imagem, como o
    YOLO espera. Com ``False`` simplesmente estica a imagem para o quadrado
    (distorce os veículos — útil só para comparação na interface).

    Raises:
        ValueError: se alguma dimensão não for positiva.
    """
    if largura <= 0 or altura <= 0 or tamanho <= 0:
        raise ValueError(f"Dimensões inválidas: {largura}×{altura} → {tamanho}.")

    if not manter_proporcao:
        return ParametrosLetterbox(
            largura, altura, tamanho, tamanho / largura, tamanho / altura, tamanho, tamanho, 0, 0
        )

    escala = min(tamanho / largura, tamanho / altura)
    nova_largura = min(tamanho, max(1, round(largura * escala)))
    nova_altura = min(tamanho, max(1, round(altura * escala)))
    return ParametrosLetterbox(
        largura_orig=largura,
        altura_orig=altura,
        tamanho=tamanho,
        # escala efetiva de cada eixo (o arredondamento pode diferir em frações de pixel)
        escala_x=nova_largura / largura,
        escala_y=nova_altura / altura,
        nova_largura=nova_largura,
        nova_altura=nova_altura,
        pad_x=(tamanho - nova_largura) // 2,
        pad_y=(tamanho - nova_altura) // 2,
    )


def _pontos(pontos: ArrayLike) -> np.ndarray:
    """Converte qualquer sequência de pontos em array ``(N, 2)`` de float64."""
    return np.asarray(pontos, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Original ↔ modelo (letterbox e letterbox inverso)
# ---------------------------------------------------------------------------
def original_para_modelo(pontos: ArrayLike, p: ParametrosLetterbox) -> np.ndarray:
    """Leva pontos do frame original para o espaço do modelo."""
    arr = _pontos(pontos)
    return np.column_stack((arr[:, 0] * p.escala_x + p.pad_x, arr[:, 1] * p.escala_y + p.pad_y))


def modelo_para_original(
    pontos: ArrayLike, p: ParametrosLetterbox, limitar: bool = True
) -> np.ndarray:
    """Letterbox inverso: leva pontos do espaço do modelo de volta ao frame original.

    Args:
        limitar: se ``True``, pontos que caem nas faixas de preenchimento são
            trazidos para a borda do frame (``0 … largura``/``0 … altura``).
    """
    arr = _pontos(pontos)
    saida = np.column_stack(
        ((arr[:, 0] - p.pad_x) / p.escala_x, (arr[:, 1] - p.pad_y) / p.escala_y)
    )
    if limitar:
        saida[:, 0] = np.clip(saida[:, 0], 0, p.largura_orig)
        saida[:, 1] = np.clip(saida[:, 1], 0, p.altura_orig)
    return saida


def caixas_modelo_para_original(caixas: ArrayLike, p: ParametrosLetterbox) -> np.ndarray:
    """Converte caixas ``(N, 4)`` no formato ``x1, y1, x2, y2`` do modelo para o original."""
    arr = np.asarray(caixas, dtype=np.float64).reshape(-1, 4)
    cantos = modelo_para_original(arr.reshape(-1, 2), p)  # (2N, 2): (x1,y1),(x2,y2),...
    return cantos.reshape(-1, 4)


# ---------------------------------------------------------------------------
# Original ↔ exibição
# ---------------------------------------------------------------------------
def escala_exibicao(largura_orig: int, largura_max: int) -> float:
    """Fator para caber em ``largura_max`` pixels na interface (nunca amplia)."""
    if largura_orig <= 0 or largura_max <= 0:
        raise ValueError("Larguras devem ser positivas.")
    return min(1.0, largura_max / largura_orig)


def original_para_exibicao(pontos: ArrayLike, escala: float) -> np.ndarray:
    """Leva pontos do frame original para a imagem exibida."""
    return _pontos(pontos) * escala


def exibicao_para_original(pontos: ArrayLike, escala: float) -> np.ndarray:
    """Leva pontos clicados/desenhados na imagem exibida de volta ao frame original."""
    return _pontos(pontos) / escala


# ---------------------------------------------------------------------------
# Normalizado ↔ pixels
# ---------------------------------------------------------------------------
def normalizados_para_pixels(
    pontos: Sequence[Sequence[float]] | ArrayLike, largura: int, altura: int
) -> np.ndarray:
    """Converte pontos em [0, 1] para pixels do frame ``largura × altura``."""
    return _pontos(pontos) * np.array([largura, altura], dtype=np.float64)


def pixels_para_normalizados(pontos: ArrayLike, largura: int, altura: int) -> np.ndarray:
    """Converte pixels do frame para coordenadas normalizadas em [0, 1]."""
    return _pontos(pontos) / np.array([largura, altura], dtype=np.float64)
