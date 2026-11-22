"""Suavização de trajetórias por média móvel exponencial (EMA).

As caixas do detector "tremem" alguns pixels de um frame para o outro. A EMA
filtra esse ruído mantendo a resposta rápida a mudanças reais::

    s_t = α · p_t + (1 − α) · s_{t−1}        (s_0 = p_0)

``α`` perto de 1 segue o ponto bruto (pouca suavização); perto de 0 suaviza
muito, mas atrasa a trajetória em relação ao veículo.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


def validar_alpha(alpha: float) -> float:
    """Garante ``0 < α ≤ 1``.

    Raises:
        ValueError: se ``alpha`` estiver fora do intervalo.
    """
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError(f"α da suavização deve estar em (0, 1]; recebido {alpha}.")
    return float(alpha)


def ema(anterior: ArrayLike, atual: ArrayLike, alpha: float) -> np.ndarray:
    """Um passo da média móvel exponencial (funciona com escalares ou vetores)."""
    alpha = validar_alpha(alpha)
    return alpha * np.asarray(atual, dtype=np.float64) + (1 - alpha) * np.asarray(
        anterior, dtype=np.float64
    )


class SuavizadorEMA:
    """Mantém um estado de EMA independente para cada ID rastreado."""

    def __init__(self, alpha: float) -> None:
        self.alpha = validar_alpha(alpha)
        self._estado: dict[Hashable, np.ndarray] = {}

    def atualizar(self, id_: Hashable, ponto: Sequence[float]) -> tuple[float, float]:
        """Incorpora um novo ponto do ID e devolve a posição suavizada."""
        bruto = np.asarray(ponto, dtype=np.float64)
        anterior = self._estado.get(id_)
        suave = bruto if anterior is None else ema(anterior, bruto, self.alpha)
        self._estado[id_] = suave
        return float(suave[0]), float(suave[1])

    def obter(self, id_: Hashable) -> tuple[float, float] | None:
        """Última posição suavizada do ID, ou ``None`` se ele não existe."""
        valor = self._estado.get(id_)
        return None if valor is None else (float(valor[0]), float(valor[1]))

    def remover(self, id_: Hashable) -> None:
        """Esquece o ID (ex.: veículo saiu da cena)."""
        self._estado.pop(id_, None)

    def manter_apenas(self, ids: Iterable[Hashable]) -> None:
        """Remove todos os IDs que não estão em ``ids``."""
        ativos = set(ids)
        for id_ in list(self._estado):
            if id_ not in ativos:
                del self._estado[id_]

    def __len__(self) -> int:
        return len(self._estado)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._estado
