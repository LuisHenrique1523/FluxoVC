"""Histórico de trajetórias por ID: suavização, classe predominante e velocidade.

O ByteTrack entrega um ID por veículo; aqui guardamos, para cada ID, as
últimas posições suavizadas (EMA), os votos de classe (o detector às vezes
alterna carro/caminhão no mesmo veículo) e a velocidade estimada pela
homografia.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field

import numpy as np

from src.geometry.homografia import estimar_velocidade_kmh, transformar_pontos
from src.geometry.roi import ponto_na_roi
from src.geometry.suavizacao import SuavizadorEMA

# Com menos que isso de histórico no trecho calibrado, a estimativa oscila demais
# (ex.: 19 a 67 km/h para um carro a 50 km/h com 0,3 s de histórico).
TEMPO_MINIMO_VELOCIDADE_S: float = 0.5


@dataclass
class Trajetoria:
    """Estado acumulado de um veículo rastreado."""

    id_rastreio: Hashable
    pontos: deque[tuple[float, float]]                 # posições suavizadas (px do original)
    votos_classe: Counter[int] = field(default_factory=Counter)
    ultimo_frame: int = 0
    velocidade_kmh: float | None = None
    # amostras (tempo, posição bruta em metros) dentro da janela de velocidade
    _amostras_m: deque[tuple[float, np.ndarray]] = field(default_factory=deque)

    @property
    def classe(self) -> int:
        """Classe mais votada até agora."""
        return self.votos_classe.most_common(1)[0][0]

    @property
    def ponto_atual(self) -> tuple[float, float]:
        return self.pontos[-1]


class HistoricoTrajetorias:
    """Mantém as trajetórias ativas e descarta IDs que sumiram."""

    def __init__(
        self,
        alpha: float,
        tamanho_max: int = 40,
        max_frames_ausente: int = 30,
        homografia: np.ndarray | None = None,
        area_calibrada_px: np.ndarray | None = None,
        janela_velocidade_s: float = 1.0,
    ) -> None:
        """
        Args:
            alpha: α da EMA.
            tamanho_max: nº máximo de pontos guardados por trajetória (desenho).
            max_frames_ausente: após quantos frames sem aparecer o ID é descartado.
            homografia: matriz imagem → metros; ``None`` desativa a velocidade.
            area_calibrada_px: quadrilátero da calibração; fora dele a homografia
                extrapola e a velocidade não é medida.
            janela_velocidade_s: intervalo usado na média de velocidade.
        """
        self.suavizador = SuavizadorEMA(alpha)
        self.tamanho_max = max(2, tamanho_max)
        self.max_frames_ausente = max(1, max_frames_ausente)
        self.homografia = homografia
        self.area_calibrada_px = area_calibrada_px
        self.janela_velocidade_s = janela_velocidade_s
        self._trajetorias: dict[Hashable, Trajetoria] = {}

    def atualizar(
        self, id_rastreio: Hashable, classe: int, ponto: tuple[float, float], frame: int, tempo_s: float
    ) -> Trajetoria:
        """Incorpora a nova posição bruta do ID e devolve a trajetória atualizada."""
        traj = self._trajetorias.get(id_rastreio)
        if traj is None:
            traj = Trajetoria(id_rastreio, deque(maxlen=self.tamanho_max))
            self._trajetorias[id_rastreio] = traj

        suave = self.suavizador.atualizar(id_rastreio, ponto)
        traj.pontos.append(suave)
        traj.votos_classe[classe] += 1
        traj.ultimo_frame = frame
        # velocidade com o ponto BRUTO: a EMA atrasa o ponto e subestimaria a velocidade,
        # principalmente nos primeiros frames de um ID (o ajuste de reta já filtra o tremor)
        self._atualizar_velocidade(traj, (float(ponto[0]), float(ponto[1])), tempo_s)
        return traj

    def _atualizar_velocidade(self, traj: Trajetoria, ponto: tuple[float, float], tempo_s: float) -> None:
        if self.homografia is None:
            return
        if self.area_calibrada_px is not None and not ponto_na_roi(ponto, self.area_calibrada_px):
            traj._amostras_m.clear()  # fora do trecho calibrado: mantém a última estimativa
            return
        traj._amostras_m.append((tempo_s, transformar_pontos(self.homografia, [ponto])[0]))
        while traj._amostras_m and tempo_s - traj._amostras_m[0][0] > self.janela_velocidade_s:
            traj._amostras_m.popleft()
        tempos = [t for t, _ in traj._amostras_m]
        estimativa = estimar_velocidade_kmh(
            [p for _, p in traj._amostras_m], tempos, TEMPO_MINIMO_VELOCIDADE_S
        )
        if estimativa is not None:
            traj.velocidade_kmh = estimativa

    def remover_ausentes(self, frame_atual: int) -> list[Hashable]:
        """Descarta IDs não vistos há mais de ``max_frames_ausente`` frames."""
        removidos = [
            id_
            for id_, traj in self._trajetorias.items()
            if frame_atual - traj.ultimo_frame > self.max_frames_ausente
        ]
        for id_ in removidos:
            del self._trajetorias[id_]
            self.suavizador.remover(id_)
        return removidos

    def __getitem__(self, id_rastreio: Hashable) -> Trajetoria:
        return self._trajetorias[id_rastreio]

    def __contains__(self, id_rastreio: object) -> bool:
        return id_rastreio in self._trajetorias

    def __iter__(self) -> Iterator[Trajetoria]:
        return iter(list(self._trajetorias.values()))

    def __len__(self) -> int:
        return len(self._trajetorias)
