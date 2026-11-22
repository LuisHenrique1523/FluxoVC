"""Região de interesse (ROI): polígono que delimita a pista.

O polígono é configurado em coordenadas normalizadas. Detecções cujo ponto de
referência fica fora dele são ignoradas (calçadas, estacionamentos, outra via).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import ArrayLike

from src.geometry.coordenadas import normalizados_para_pixels
from src.geometry.erros import RoiInvalidaError

MINIMO_PONTOS: int = 3
AREA_MINIMA_NORM: float = 0.001  # 0,1% do frame
TOLERANCIA: float = 1e-6


def validar_roi(pontos_norm: Sequence[Sequence[float]] | ArrayLike) -> np.ndarray:
    """Valida o polígono da ROI em coordenadas normalizadas.

    Returns:
        Array ``(N, 2)`` de float64.

    Raises:
        RoiInvalidaError: menos de 3 pontos, valores fora de [0, 1] ou área quase nula.
    """
    try:
        arr = np.asarray(pontos_norm, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as erro:
        raise RoiInvalidaError("A ROI deve ser uma lista de pontos (x, y).") from erro

    if len(arr) < MINIMO_PONTOS:
        raise RoiInvalidaError(
            f"A ROI precisa de pelo menos {MINIMO_PONTOS} pontos (recebeu {len(arr)})."
        )
    if not np.all(np.isfinite(arr)) or arr.min() < -TOLERANCIA or arr.max() > 1 + TOLERANCIA:
        raise RoiInvalidaError("Os pontos da ROI devem estar dentro do frame (valores entre 0 e 1).")
    if abs(cv2.contourArea(arr.astype(np.float32))) < AREA_MINIMA_NORM:
        raise RoiInvalidaError(
            "A ROI não tem área: os pontos estão alinhados ou muito próximos."
        )
    return np.clip(arr, 0.0, 1.0)


def roi_em_pixels(
    pontos_norm: Sequence[Sequence[float]] | ArrayLike, largura: int, altura: int
) -> np.ndarray:
    """Valida a ROI e converte para pixels inteiros ``(N, 2)`` (formato do ``cv2.fillPoly``)."""
    return np.round(normalizados_para_pixels(validar_roi(pontos_norm), largura, altura)).astype(
        np.int32
    )


def ponto_na_roi(ponto: Sequence[float], poligono_px: np.ndarray) -> bool:
    """Indica se ``ponto`` (x, y) está dentro ou na borda do polígono em pixels."""
    contorno = np.asarray(poligono_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contorno, (float(ponto[0]), float(ponto[1])), False) >= 0
