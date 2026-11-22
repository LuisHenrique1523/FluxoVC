"""Detectores falsos que cumprem o contrato ``Detector`` sem YOLO nem pesos."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.models.tipos import DeteccaoBruta

# Mesmas cores BGR usadas em ``conftest.desenhar_frame_sintetico``
COR_CARRO = (255, 170, 0)       # veículo 1: desce pela faixa da esquerda
COR_CAMINHAO = (60, 60, 230)    # veículo 2: sobe pela faixa da direita


class DetectorPorCor:
    """Acha retângulos de cores conhecidas na imagem (já no espaço do modelo).

    Cada cor vira uma classe e um ID fixo, imitando o que YOLO + ByteTrack fariam.
    """

    def __init__(
        self,
        alvos: Sequence[tuple[tuple[int, int, int], int, int]] = (
            (COR_CARRO, 2, 1),
            (COR_CAMINHAO, 7, 2),
        ),
        tolerancia: int = 50,
        pixels_minimos: int = 30,
    ) -> None:
        self.alvos = alvos
        self.tolerancia = tolerancia
        self.pixels_minimos = pixels_minimos
        self.chamadas = 0
        self.reinicios = 0

    def _achar(self, imagem: np.ndarray, com_id: bool) -> list[DeteccaoBruta]:
        self.chamadas += 1
        deteccoes = []
        for cor, classe, id_ in self.alvos:
            cor_arr = np.array(cor, dtype=np.int16)
            mascara = cv2.inRange(
                imagem,
                np.clip(cor_arr - self.tolerancia, 0, 255).astype(np.uint8),
                np.clip(cor_arr + self.tolerancia, 0, 255).astype(np.uint8),
            )
            ys, xs = np.nonzero(mascara)
            if len(xs) < self.pixels_minimos:
                continue
            caixa = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            deteccoes.append(DeteccaoBruta(caixa, classe, 0.9, id_ if com_id else None))
        return deteccoes

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        return self._achar(imagem, com_id=True)

    def detectar(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        return self._achar(imagem, com_id=False)

    def reiniciar(self) -> None:
        self.reinicios += 1


class DetectorRoteirizado:
    """Devolve, a cada chamada, a próxima lista de detecções do roteiro."""

    def __init__(self, roteiro: Sequence[Sequence[DeteccaoBruta]]) -> None:
        self.roteiro = list(roteiro)
        self.chamadas = 0

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        indice = self.chamadas
        self.chamadas += 1
        return list(self.roteiro[indice]) if indice < len(self.roteiro) else []

    def detectar(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        return [DeteccaoBruta(d.caixa, d.classe, d.confianca, None) for d in self.rastrear(imagem)]

    def reiniciar(self) -> None:
        self.chamadas = 0
