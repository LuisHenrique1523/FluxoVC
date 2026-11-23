"""Anonimização do frame antes de enviá-lo a um VLM.

O VLM nunca recebe o frame original. Antes, este módulo:

1. **reduz a resolução** (lado maior ≤ 768 px): placas e rostos distantes deixam
   de ser legíveis, e o VLM fica mais rápido;
2. **pixeliza as pessoas** detectadas pelo YOLO (classe "pessoa" do COCO);
3. **pixeliza a região da placa** de cada veículo: a faixa central da parte de
   baixo da caixa, onde ficam as placas dianteira e traseira.

Limitação: não há um detector de placas. A região é estimada pela geometria da
caixa, o que cobre a placa na maioria das vistas de frente e de trás, mas não
garante 100 %. Por isso a resolução também é reduzida.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

LADO_MAXIMO_PX: int = 768
FRACAO_ALTURA_PLACA: float = 0.40     # parte de baixo da caixa
FRACAO_LARGURA_PLACA: float = 0.70    # faixa central
MARGEM_PESSOA: float = 0.10           # aumenta a caixa da pessoa em 10 % de cada lado
BLOCO_PIXELIZACAO: int = 10           # px por bloco depois da redução

Caixa = Sequence[float]  # (x1, y1, x2, y2)


def reduzir(imagem: np.ndarray, lado_maximo: int = LADO_MAXIMO_PX) -> tuple[np.ndarray, float]:
    """Reduz a imagem para o lado maior caber em ``lado_maximo``. Devolve a imagem e a escala."""
    altura, largura = imagem.shape[:2]
    escala = min(1.0, lado_maximo / max(altura, largura))
    if escala == 1.0:
        return imagem.copy(), 1.0
    tamanho = (max(1, round(largura * escala)), max(1, round(altura * escala)))
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA), escala


def regiao_placa(caixa: Caixa) -> tuple[float, float, float, float]:
    """Retângulo onde a placa provavelmente está: parte de baixo e centro da caixa do veículo."""
    x1, y1, x2, y2 = caixa
    largura, altura = x2 - x1, y2 - y1
    folga = largura * (1 - FRACAO_LARGURA_PLACA) / 2
    return (x1 + folga, y2 - altura * FRACAO_ALTURA_PLACA, x2 - folga, y2)


def expandir(caixa: Caixa, fracao: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = caixa
    dx, dy = (x2 - x1) * fracao, (y2 - y1) * fracao
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def pixelizar(imagem: np.ndarray, caixa: Caixa, bloco: int = BLOCO_PIXELIZACAO) -> None:
    """Pixeliza (in-place) o retângulo ``caixa``, recortado aos limites da imagem."""
    altura, largura = imagem.shape[:2]
    x1, y1 = max(0, int(np.floor(caixa[0]))), max(0, int(np.floor(caixa[1])))
    x2, y2 = min(largura, int(np.ceil(caixa[2]))), min(altura, int(np.ceil(caixa[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    # no máximo ~8 blocos no lado menor: rostos grandes (perto da câmera) também ficam ilegíveis
    bloco = max(bloco, min(x2 - x1, y2 - y1) // 8)
    recorte = imagem[y1:y2, x1:x2]
    pequeno = cv2.resize(
        recorte, (max(1, (x2 - x1) // bloco), max(1, (y2 - y1) // bloco)), interpolation=cv2.INTER_AREA
    )
    imagem[y1:y2, x1:x2] = cv2.resize(pequeno, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def anonimizar(
    imagem: np.ndarray,
    pessoas: Iterable[Caixa] = (),
    veiculos: Iterable[Caixa] = (),
    lado_maximo: int = LADO_MAXIMO_PX,
) -> np.ndarray:
    """Cópia reduzida da imagem com pessoas e placas pixelizadas.

    Args:
        imagem: frame original BGR (não é alterado).
        pessoas / veiculos: caixas ``(x1, y1, x2, y2)`` em pixels do frame original.
    """
    saida, escala = reduzir(imagem, lado_maximo)
    for caixa in pessoas:
        pixelizar(saida, [v * escala for v in expandir(caixa, MARGEM_PESSOA)])
    for caixa in veiculos:
        pixelizar(saida, [v * escala for v in regiao_placa(caixa)])
    return saida
