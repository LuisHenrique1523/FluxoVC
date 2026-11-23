"""Regressão: velocidade de IDs "jovens" e preenchimento posterior no evento.

Bug encontrado com vídeo real: com muitas trocas de ID, os veículos cruzavam a
linha pouco depois de ganhar um ID. A velocidade era calculada com o ponto
suavizado (EMA), que ainda estava "alcançando" o veículo, e saía subestimada
(ou vazia). Agora ela usa o ponto bruto com ajuste de reta e é preenchida
depois, se o veículo cruzou antes de ter histórico suficiente.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.config import PROJECT_ROOT, GeometriaConfig, calibracao_e_padrao
from src.models.contagem import ContadorLinha
from src.models.rastreamento import TEMPO_MINIMO_VELOCIDADE_S, HistoricoTrajetorias

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from cena_sintetica import H_MUNDO_IMAGEM, mundo_para_imagem  # noqa: E402

FPS, KMH = 24.0, 50.0


def _simular(frames: int, semente: int) -> float | None:
    """Carro a 50 km/h na cena sintética, com calibração perfeita e tremor de 2 px."""
    rng = np.random.default_rng(semente)
    hist = HistoricoTrajetorias(alpha=0.4, homografia=np.linalg.inv(H_MUNDO_IMAGEM))
    traj = None
    for i in range(frames):
        t = i / FPS
        ponto = mundo_para_imagem([[8.75, 10 + KMH / 3.6 * t]])[0] + rng.normal(0, 2, 2)
        traj = hist.atualizar(1, 2, (float(ponto[0]), float(ponto[1])), i, t)
    return traj.velocidade_kmh


def test_id_jovem_nao_subestima_a_velocidade() -> None:
    # 0,5 s de histórico: antes a média ficava em ~43 km/h por causa do atraso da EMA
    estimativas = [_simular(13, s) for s in range(30)]
    assert all(v is not None for v in estimativas)
    assert np.mean(estimativas) == pytest.approx(KMH, rel=0.04)
    assert min(estimativas) > 0.75 * KMH


def test_sem_historico_suficiente_nao_estima() -> None:
    assert TEMPO_MINIMO_VELOCIDADE_S == pytest.approx(0.5)
    assert _simular(10, 0) is None  # 9 intervalos = 0,375 s


def test_evento_sem_velocidade_e_preenchido_depois() -> None:
    contador = ContadorLinha((0, 100), (200, 100))
    contador.atualizar(7, 2, (50, 90), 1, 0.04)
    evento = contador.atualizar(7, 2, (50, 110), 2, 0.08, velocidade_kmh=None)
    assert evento is not None and evento.velocidade_kmh is None
    assert not contador.completar_velocidade(7, None)        # ainda sem estimativa
    assert contador.completar_velocidade(7, 48.2)
    assert contador.eventos[0].velocidade_kmh == 48.2
    assert not contador.completar_velocidade(7, 60.0)        # só preenche uma vez
    assert contador.eventos[0].velocidade_kmh == 48.2


def test_detecta_calibracao_padrao() -> None:
    assert calibracao_e_padrao(GeometriaConfig())
    alterada = GeometriaConfig()
    alterada.homografia_origem = [(0.3, 0.4), (0.65, 0.4), (0.95, 0.95), (0.05, 0.95)]
    assert not calibracao_e_padrao(alterada)
