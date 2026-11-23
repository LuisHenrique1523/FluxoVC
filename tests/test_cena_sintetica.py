"""Teste de integração na cena sintética com perspectiva (scripts/cena_sintetica.py).

Como a homografia da cena e a velocidade de cada veículo são conhecidas, dá
para conferir de ponta a ponta: letterbox → interpretação → EMA → contagem →
homografia → km/h, com perspectiva real e detecções com ruído.
"""

from __future__ import annotations

import sys

import pytest

from src.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cena_sintetica import CenaSintetica  # noqa: E402
from gerar_figuras import Simulacao, config_cena  # noqa: E402

from src.models.contagem import SENTIDO_AB, SENTIDO_BA  # noqa: E402


@pytest.fixture(scope="module")
def simulacao() -> Simulacao:
    return Simulacao(CenaSintetica(), config_cena()).rodar(guardar=False)


def test_conta_todos_os_veiculos_em_movimento_com_classe_e_sentido(simulacao: Simulacao) -> None:
    cena = simulacao.cena
    esperados = {v.id_: v for v in cena.veiculos if v.parado_em_y is None}
    eventos = {e.id_rastreio: e for e in simulacao.contador.eventos}
    assert set(eventos) == set(esperados)  # o carro estacionado (fora da ROI) não conta
    for id_, evento in eventos.items():
        veiculo = esperados[id_]
        assert evento.classe == veiculo.classe
        # faixas 0–1 vêm em direção à câmera: descem na imagem, do lado A para o B
        assert evento.sentido == (SENTIDO_AB if veiculo.vem else SENTIDO_BA)
        assert evento.timestamp_s == pytest.approx(veiculo.cruza_linha_em(), abs=0.5)


def test_velocidade_pela_homografia_fica_perto_da_real(simulacao: Simulacao) -> None:
    reais = {v.id_: v.velocidade_kmh for v in simulacao.cena.veiculos}
    for evento in simulacao.contador.eventos:
        assert evento.velocidade_kmh is not None
        erro = abs(evento.velocidade_kmh - reais[evento.id_rastreio]) / reais[evento.id_rastreio]
        assert erro < 0.06, f"ID {evento.id_rastreio}: {evento.velocidade_kmh:.1f} km/h (real {reais[evento.id_rastreio]})"
