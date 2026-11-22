"""Tipos compartilhados da etapa de modelo e o contrato (``Protocol``) do detector.

O :class:`AnalisadorFluxo` depende só do contrato :class:`Detector`, não do
YOLO. Assim os testes injetam um detector falso e rodam a pipeline inteira sem
baixar pesos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ErroModelo(RuntimeError):
    """Falha ao carregar ou executar o modelo (mensagem amigável)."""


class PesosIndisponiveisError(ErroModelo):
    """Pesos ausentes e sem internet para baixá-los."""


@dataclass(frozen=True)
class DeteccaoBruta:
    """Uma detecção como sai do detector, **no espaço do modelo** (ex.: 640×640)."""

    caixa: tuple[float, float, float, float]   # x1, y1, x2, y2
    classe: int                                # id COCO (2 carro, 3 moto, 5 ônibus, 7 caminhão)
    confianca: float
    id_rastreio: int | None = None             # None quando o rastreador ainda não confirmou


class Detector(Protocol):
    """Contrato mínimo de um detector/rastreador."""

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        """Detecta e associa IDs entre frames consecutivos do mesmo vídeo."""
        ...

    def detectar(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        """Detecta sem rastrear (imagem única)."""
        ...

    def reiniciar(self) -> None:
        """Zera o estado do rastreador (chamar ao trocar de vídeo)."""
        ...
