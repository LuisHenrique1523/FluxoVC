"""Miniatura da vista de cima (bird's-eye) com os veículos plotados.

A homografia do trecho calibrado (ver :mod:`src.geometry.homografia`),
escalada em pixels por metro, "desentorta" a pista com ``cv2.warpPerspective``.
Os pontos de referência dos veículos passam pela mesma matriz e são
desenhados como círculos na cor da classe — na vista de cima as distâncias
são proporcionais às reais.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from numpy.typing import ArrayLike

from src.geometry.homografia import gerar_birdseye, matriz_birdseye, transformar_pontos
from src.models.analisador import VeiculoRastreado
from src.visualization.desenho import BRANCO, cor_classe, escrever


class VistaBirdseye:
    """Gera a miniatura bird's-eye com altura fixa em pixels."""

    def __init__(
        self, origem_px: ArrayLike, largura_m: float, comprimento_m: float, altura_px: int
    ) -> None:
        """
        Raises:
            HomografiaInvalidaError: pontos de calibração degenerados.
        """
        self.px_por_metro = max(1, altura_px) / comprimento_m
        self.matriz, self.tamanho = matriz_birdseye(origem_px, largura_m, comprimento_m, self.px_por_metro)
        self.comprimento_m = comprimento_m

    def renderizar(self, imagem_original: np.ndarray, veiculos: Iterable[VeiculoRastreado]) -> np.ndarray:
        """Vista de cima do trecho com os veículos e uma régua de escala."""
        vista = gerar_birdseye(imagem_original, self.matriz, self.tamanho)
        largura, altura = self.tamanho
        raio = max(3, largura // 20)
        for v in veiculos:
            x, y = transformar_pontos(self.matriz, [v.ponto])[0]
            if 0 <= x < largura and 0 <= y < altura:
                cv2.circle(vista, (int(x), int(y)), raio, cor_classe(v.classe), -1, cv2.LINE_AA)
                cv2.circle(vista, (int(x), int(y)), raio, BRANCO, 1, cv2.LINE_AA)
        # régua de 5 m no canto inferior esquerdo
        cinco_m = int(round(5 * self.px_por_metro))
        if cinco_m < altura - 10:
            cv2.line(vista, (4, altura - 6), (4, altura - 6 - cinco_m), BRANCO, 2)
            escrever(vista, "5 m", (8, altura - 8), 0.35, BRANCO)
        return vista


def colar_miniatura(imagem: np.ndarray, miniatura: np.ndarray, margem: int = 8, titulo: str = "vista de cima") -> None:
    """Cola ``miniatura`` no canto superior direito de ``imagem`` (in-place), com borda."""
    altura, largura = imagem.shape[:2]
    m_alt, m_larg = miniatura.shape[:2]
    if m_alt + 2 * margem > altura or m_larg + 2 * margem > largura:
        escala = min((altura - 2 * margem) / m_alt, (largura - 2 * margem) / m_larg)
        miniatura = cv2.resize(miniatura, (max(1, int(m_larg * escala)), max(1, int(m_alt * escala))))
        m_alt, m_larg = miniatura.shape[:2]
    x1, y1 = largura - m_larg - margem, margem
    imagem[y1 : y1 + m_alt, x1 : x1 + m_larg] = miniatura
    cv2.rectangle(imagem, (x1 - 1, y1 - 1), (x1 + m_larg, y1 + m_alt), BRANCO, 1)
    # título alinhado à direita para nunca sair da imagem
    largura_titulo = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
    escrever(imagem, titulo, (largura - margem - largura_titulo, y1 + m_alt + 16), 0.4, BRANCO)
