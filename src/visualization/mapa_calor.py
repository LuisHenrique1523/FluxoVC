"""Mapa de calor acumulado de ocupação da via.

A cada frame processado, a **pegada** de cada veículo (terço inferior da caixa,
a parte que toca o asfalto) soma +1 numa grade de acumulação em resolução
reduzida. Para exibir, a grade é normalizada (raiz quadrada, para realçar
áreas pouco ocupadas), suavizada, colorida com uma colormap (TURBO ou JET) e
misturada ao frame com transparência — só onde houve ocupação.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from src.visualization.desenho import misturar

COLORMAPS: dict[str, int] = {"TURBO": cv2.COLORMAP_TURBO, "JET": cv2.COLORMAP_JET}
FRACAO_PEGADA: float = 1 / 3
LIMIAR_EXIBICAO: int = 8  # valores normalizados (0–255) abaixo disso ficam transparentes


class MapaCalor:
    """Acumulador de ocupação no espaço do frame original."""

    def __init__(self, largura: int, altura: int, reducao: int = 4, decaimento: float = 1.0) -> None:
        """
        Args:
            reducao: fator de redução da grade (4 → 1 célula para 4×4 pixels).
            decaimento: multiplicador por frame (1,0 = acumula para sempre;
                0,99 = o passado "esfria" aos poucos).
        """
        self.largura, self.altura = largura, altura
        self.reducao = max(1, reducao)
        self.decaimento = decaimento
        self.acumulado = np.zeros(
            (math.ceil(altura / self.reducao), math.ceil(largura / self.reducao)), dtype=np.float32
        )

    def acumular(self, caixas: Iterable[tuple[float, float, float, float]]) -> None:
        """Soma a pegada de cada caixa ``(x1, y1, x2, y2)`` (px do original)."""
        if self.decaimento < 1.0:
            self.acumulado *= self.decaimento
        r = self.reducao
        alt, larg = self.acumulado.shape
        for x1, y1, x2, y2 in caixas:
            y_pegada = y2 - (y2 - y1) * FRACAO_PEGADA
            c1, c2 = max(0, int(x1 // r)), min(larg, math.ceil(x2 / r))
            l1, l2 = max(0, int(y_pegada // r)), min(alt, math.ceil(y2 / r))
            if c2 > c1 and l2 > l1:
                self.acumulado[l1:l2, c1:c2] += 1.0

    @property
    def vazio(self) -> bool:
        return float(self.acumulado.max()) <= 0.0

    def normalizado(self) -> np.ndarray:
        """Grade em ``uint8`` (0–255), com raiz quadrada para realçar áreas pouco ocupadas."""
        maximo = float(self.acumulado.max())
        if maximo <= 0:
            return np.zeros(self.acumulado.shape, dtype=np.uint8)
        return (np.sqrt(self.acumulado / maximo) * 255).astype(np.uint8)

    def colorir(self, tamanho: tuple[int, int], colormap: str = "TURBO") -> tuple[np.ndarray, np.ndarray]:
        """Imagem colorida ``tamanho = (largura, altura)`` e máscara bool de onde há calor."""
        grade = cv2.GaussianBlur(self.normalizado(), (5, 5), 0)
        grade = cv2.resize(grade, tamanho, interpolation=cv2.INTER_LINEAR)
        mapa = COLORMAPS.get(colormap.upper(), cv2.COLORMAP_TURBO)
        return cv2.applyColorMap(grade, mapa), grade > LIMIAR_EXIBICAO

    def sobrepor(self, imagem: np.ndarray, opacidade: float, colormap: str = "TURBO") -> None:
        """Mistura o mapa de calor sobre ``imagem`` (in-place), só onde houve ocupação."""
        if self.vazio:
            return
        colorido, mascara = self.colorir((imagem.shape[1], imagem.shape[0]), colormap)
        misturar(imagem, colorido, opacidade, mascara)

    def reiniciar(self) -> None:
        self.acumulado.fill(0)
