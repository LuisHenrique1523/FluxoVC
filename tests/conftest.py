"""Fixtures compartilhadas pelos testes.

``fabrica_video`` gera vídeos sintéticos pequenos (retângulos coloridos se
movendo sobre fundo cinza), para testar a pipeline sem vídeos reais nem pesos
de modelo.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

FabricaVideo = Callable[..., Path]


def desenhar_frame_sintetico(indice: int, largura: int, altura: int) -> np.ndarray:
    """Frame cinza com dois "veículos" (retângulos) andando em sentidos opostos."""
    frame = np.full((altura, largura, 3), 90, dtype=np.uint8)
    lado = max(10, min(largura, altura) // 8)
    passo = max(2, altura // 40)
    # veículo 1: desce pela faixa da esquerda
    x1, y1 = largura // 4, (indice * passo) % altura
    cv2.rectangle(frame, (x1, y1), (x1 + lado, y1 + lado), (255, 170, 0), -1)
    # veículo 2: sobe pela faixa da direita
    x2, y2 = 3 * largura // 4 - lado, altura - lado - (indice * passo) % altura
    cv2.rectangle(frame, (x2, y2), (x2 + lado, y2 + lado), (60, 60, 230), -1)
    return frame


@pytest.fixture
def fabrica_video(tmp_path: Path) -> FabricaVideo:
    """Devolve uma função que cria um vídeo sintético em ``tmp_path``."""

    def criar(
        nome: str = "sintetico.avi",
        n_frames: int = 20,
        largura: int = 640,
        altura: int = 360,
        fps: float = 15.0,
        fourcc: str = "MJPG",
    ) -> Path:
        caminho = tmp_path / nome
        escritor = cv2.VideoWriter(
            str(caminho), cv2.VideoWriter.fourcc(*fourcc), fps, (largura, altura)
        )
        if not escritor.isOpened():
            pytest.skip(f"Codec {fourcc} indisponível neste OpenCV.")
        for i in range(n_frames):
            escritor.write(desenhar_frame_sintetico(i, largura, altura))
        escritor.release()
        return caminho

    return criar
