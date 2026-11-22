"""Medidas de desempenho em tempo real (FPS)."""

from __future__ import annotations

import time


class MedidorFPS:
    """FPS suavizado por média móvel exponencial entre chamadas consecutivas.

    O FPS instantâneo oscila muito (um frame lento derruba o valor); a EMA
    deixa o número legível no painel.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._anterior: float | None = None
        self.fps: float | None = None

    def registrar(self, agora: float | None = None) -> float | None:
        """Marca o fim de um frame e devolve o FPS suavizado (``None`` no 1º frame)."""
        agora = time.perf_counter() if agora is None else agora
        if self._anterior is not None:
            intervalo = agora - self._anterior
            if intervalo > 0:
                instantaneo = 1.0 / intervalo
                self.fps = (
                    instantaneo
                    if self.fps is None
                    else self.alpha * instantaneo + (1 - self.alpha) * self.fps
                )
        self._anterior = agora
        return self.fps

    def reiniciar(self) -> None:
        self._anterior = None
        self.fps = None
