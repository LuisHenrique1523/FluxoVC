"""Transformação de perspectiva (homografia): vista de cima e velocidade.

O usuário marca 4 pontos da pista na imagem — superior esquerdo, superior
direito, inferior direito, inferior esquerdo — e informa as medidas reais desse
trecho (largura e comprimento em metros). A homografia ``H`` (3×3) leva
qualquer ponto da imagem para o plano da pista, em metros::

    [x', y', w']ᵀ = H · [x, y, 1]ᵀ      →      (X, Y) = (x'/w', y'/w')

Com a posição em metros ao longo do tempo, a velocidade é ``distância / Δt``.
Multiplicando ``H`` por uma escala (pixels por metro) obtemos a imagem
"bird's-eye" com ``cv2.warpPerspective``.

Limitação: a homografia só vale para pontos **no plano da pista**. Por isso a
posição do veículo deve ser o ponto de contato com o chão (centro da base da
caixa), não o centro da caixa.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import ArrayLike

from src.geometry.erros import HomografiaInvalidaError

AREA_MINIMA_PX: float = 100.0
EPS_ALINHAMENTO: float = 1e-6
MSG_ORDEM = (
    "Os 4 pontos da pista devem formar um quadrilátero convexo, sem três pontos "
    "alinhados, na ordem: superior esquerdo, superior direito, inferior direito, "
    "inferior esquerdo."
)


def validar_quadrilatero(pontos_px: ArrayLike) -> np.ndarray:
    """Confere se os 4 pontos formam um quadrilátero convexo e não degenerado.

    O teste usa o sinal do produto vetorial entre arestas consecutivas: num
    polígono convexo, sem autointerseção, todos têm o mesmo sinal.

    Returns:
        Array ``(4, 2)`` float32.

    Raises:
        HomografiaInvalidaError: quantidade errada, pontos repetidos/alinhados,
            quadrilátero cruzado ou área muito pequena.
    """
    try:
        arr = np.asarray(pontos_px, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as erro:
        raise HomografiaInvalidaError("Os pontos da pista devem ser pares (x, y).") from erro
    if arr.shape != (4, 2):
        raise HomografiaInvalidaError(
            f"A calibração precisa de exatamente 4 pontos (recebeu {len(arr)})."
        )
    if not np.all(np.isfinite(arr)):
        raise HomografiaInvalidaError("Os pontos da pista contêm valores inválidos.")

    arestas = np.roll(arr, -1, axis=0) - arr                 # p[i+1] − p[i]
    proximas = np.roll(arestas, -1, axis=0)                  # p[i+2] − p[i+1]
    cruzados = arestas[:, 0] * proximas[:, 1] - arestas[:, 1] * proximas[:, 0]
    escala = max(1.0, float(np.abs(arr).max())) ** 2
    if np.any(np.abs(cruzados) <= EPS_ALINHAMENTO * escala) or not (
        np.all(cruzados > 0) or np.all(cruzados < 0)
    ):
        raise HomografiaInvalidaError(MSG_ORDEM)

    if abs(cv2.contourArea(arr.astype(np.float32))) < AREA_MINIMA_PX:
        raise HomografiaInvalidaError("Os 4 pontos da pista estão próximos demais.")
    return arr.astype(np.float32)


def _validar_medidas(largura_m: float, comprimento_m: float) -> None:
    for nome, valor in (("largura", largura_m), ("comprimento", comprimento_m)):
        if not math.isfinite(valor) or valor <= 0:
            raise HomografiaInvalidaError(
                f"A {nome} real do trecho deve ser um valor positivo em metros."
            )


def calcular_homografia(
    origem_px: ArrayLike, largura_m: float, comprimento_m: float
) -> np.ndarray:
    """Homografia que leva pixels da imagem para metros no plano da pista.

    O destino é o retângulo ``(0,0) (L,0) (L,C) (0,C)``: ``X`` cruza a pista
    (largura ``L``) e ``Y`` segue a pista (comprimento ``C``).

    Raises:
        HomografiaInvalidaError: pontos degenerados, medidas inválidas ou
            matriz não inversível.
    """
    origem = validar_quadrilatero(origem_px)
    _validar_medidas(largura_m, comprimento_m)
    destino = np.array(
        [[0, 0], [largura_m, 0], [largura_m, comprimento_m], [0, comprimento_m]],
        dtype=np.float32,
    )
    matriz = cv2.getPerspectiveTransform(origem, destino).astype(np.float64)
    if not np.all(np.isfinite(matriz)) or abs(np.linalg.det(matriz)) < 1e-12:
        raise HomografiaInvalidaError("Não foi possível calcular a homografia com esses pontos.")
    return matriz


def transformar_pontos(matriz: np.ndarray, pontos: ArrayLike) -> np.ndarray:
    """Aplica a homografia a pontos ``(N, 2)`` (divisão perspectiva incluída)."""
    arr = np.asarray(pontos, dtype=np.float64).reshape(-1, 1, 2)
    if len(arr) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(arr, matriz).reshape(-1, 2)


def matriz_birdseye(
    origem_px: ArrayLike, largura_m: float, comprimento_m: float, px_por_metro: float = 10.0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Homografia imagem → vista de cima em pixels, e o tamanho ``(largura, altura)`` da saída."""
    if not math.isfinite(px_por_metro) or px_por_metro <= 0:
        raise HomografiaInvalidaError("A escala da vista de cima deve ser positiva.")
    escala = np.diag([px_por_metro, px_por_metro, 1.0])
    matriz = escala @ calcular_homografia(origem_px, largura_m, comprimento_m)
    tamanho = (max(1, math.ceil(largura_m * px_por_metro)), max(1, math.ceil(comprimento_m * px_por_metro)))
    return matriz, tamanho


def gerar_birdseye(imagem: np.ndarray, matriz_px: np.ndarray, tamanho: tuple[int, int]) -> np.ndarray:
    """Gera a imagem "vista de cima" do trecho calibrado (``cv2.warpPerspective``)."""
    return cv2.warpPerspective(imagem, matriz_px, tamanho, flags=cv2.INTER_LINEAR)


def estimar_velocidade_kmh(
    posicoes_m: ArrayLike, tempos_s: Sequence[float], dt_minimo_s: float = 0.25
) -> float | None:
    """Velocidade (km/h) pelo ajuste de uma reta posição × tempo (mínimos quadrados).

    A inclinação da reta em X e em Y é o vetor velocidade (m/s). Usar todas as
    posições da janela, e não só a primeira e a última, reduz o efeito do
    "tremor" das caixas, que geraria picos irreais de velocidade. As posições
    devem ser **brutas** (sem EMA): a suavização atrasa o ponto e subestima a
    velocidade, principalmente logo depois de o ID aparecer.

    Returns:
        Velocidade em km/h, ou ``None`` se houver menos de 2 posições ou se o
        intervalo de tempo for curto demais para uma estimativa confiável.
    """
    arr = np.asarray(posicoes_m, dtype=np.float64).reshape(-1, 2)
    tempos = np.asarray(tempos_s, dtype=np.float64).ravel()
    if len(arr) < 2 or len(tempos) != len(arr):
        return None
    if float(tempos[-1] - tempos[0]) < dt_minimo_s:
        return None
    centrados = tempos - tempos.mean()
    variancia = float(centrados @ centrados)
    if variancia <= 0:
        return None
    velocidade = centrados @ (arr - arr.mean(axis=0)) / variancia  # (vx, vy) em m/s
    return float(np.linalg.norm(velocidade)) * 3.6
