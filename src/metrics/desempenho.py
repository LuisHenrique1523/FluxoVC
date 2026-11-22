"""Medidas de desempenho: FPS, memória e resumo dos tempos por etapa."""

from __future__ import annotations

import time

import pandas as pd
import psutil

BYTES_POR_MB = 1024 * 1024


class MonitorMemoria:
    """Acompanha a memória residente (RSS) do processo e guarda o pico."""

    def __init__(self) -> None:
        self._processo = psutil.Process()
        self.inicial_mb = self.atual_mb()
        self.pico_mb = self.inicial_mb

    def atual_mb(self) -> float:
        return self._processo.memory_info().rss / BYTES_POR_MB

    def amostrar(self) -> float:
        atual = self.atual_mb()
        self.pico_mb = max(self.pico_mb, atual)
        return atual


def resumo_tempos(tempos: pd.DataFrame) -> dict[str, float]:
    """Média de cada etapa (ms), total médio e percentil 95 do total por frame."""
    if tempos.empty:
        return {}
    resumo = {f"{coluna}_ms": float(tempos[coluna].mean()) for coluna in tempos.columns}
    total = tempos.sum(axis=1)
    resumo["total_ms"] = float(total.mean())
    resumo["total_p95_ms"] = float(total.quantile(0.95))
    return resumo


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
