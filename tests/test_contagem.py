"""Testes da Fase 3 — cruzamento de linha (casos sintéticos) e histórico de trajetórias."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.erros import ErroGeometria
from src.geometry.homografia import calcular_homografia
from src.models.contagem import (
    SENTIDO_AB,
    SENTIDO_BA,
    ContadorLinha,
    lado_da_linha,
    sentido_cruzamento,
)
from src.models.rastreamento import HistoricoTrajetorias

# Linha horizontal y = 100, de x = 0 (A) até x = 200 (B)
A, B = (0.0, 100.0), (200.0, 100.0)


# ---------------------------------------------------------------------------
# Produto vetorial e sentido
# ---------------------------------------------------------------------------
def test_lado_da_linha_pelo_sinal_do_produto_vetorial() -> None:
    assert lado_da_linha((50, 150), A, B) > 0   # abaixo (y cresce para baixo)
    assert lado_da_linha((50, 50), A, B) < 0    # acima
    assert lado_da_linha((50, 100), A, B) == 0  # sobre a linha


@pytest.mark.parametrize(
    ("anterior", "atual", "esperado"),
    [
        ((100, 90), (100, 110), SENTIDO_AB),    # desce: A→B
        ((100, 110), (100, 90), SENTIDO_BA),    # sobe: B→A
        ((10, 90), (190, 95), None),            # anda paralelo, sem cruzar
        ((100, 110), (150, 130), None),         # fica do mesmo lado
        ((300, 90), (300, 110), None),          # cruza o prolongamento, fora do segmento
        ((-50, 90), (-50, 110), None),          # idem, do outro lado
        ((190, 90), (210, 110), SENTIDO_AB),    # diagonal que passa pelo trecho final
        ((0, 90), (0, 110), SENTIDO_AB),        # passa exatamente pela ponta A
        ((100, 100), (100, 110), None),         # sai de cima da linha: não decide sozinho
    ],
)
def test_sentido_cruzamento(anterior, atual, esperado) -> None:
    assert sentido_cruzamento(anterior, atual, A, B) == esperado


def test_linha_diagonal() -> None:
    a, b = (0, 0), (100, 100)
    assert sentido_cruzamento((80, 20), (20, 80), a, b) is not None
    assert sentido_cruzamento((80, 20), (20, 80), a, b) != sentido_cruzamento((20, 80), (80, 20), a, b)


# ---------------------------------------------------------------------------
# Contador
# ---------------------------------------------------------------------------
def _percorrer(contador: ContadorLinha, id_: int, classe: int, ys: list[float], x: float = 100.0):
    return [contador.atualizar(id_, classe, (x, y), i, i / 10) for i, y in enumerate(ys)]


def test_conta_uma_vez_com_sentido_e_classe() -> None:
    c = ContadorLinha(A, B)
    eventos = [e for e in _percorrer(c, 1, 2, [60, 80, 95, 105, 120]) if e]
    assert len(eventos) == 1
    evento = eventos[0]
    assert (evento.id_rastreio, evento.classe, evento.sentido, evento.frame) == (1, 2, SENTIDO_AB, 3)
    assert evento.nome_classe == "carro"
    assert evento.timestamp_s == pytest.approx(0.3)
    assert c.frame_ultimo_cruzamento == 3
    assert c.total == 1


def test_tremor_sobre_a_linha_nao_duplica() -> None:
    c = ContadorLinha(A, B)
    _percorrer(c, 7, 2, [95, 105, 97, 103, 96, 104, 120])
    assert c.total == 1


def test_parar_sobre_a_linha_e_depois_cruzar() -> None:
    c = ContadorLinha(A, B)
    eventos = [e for e in _percorrer(c, 3, 3, [90, 100, 100, 110]) if e]
    assert len(eventos) == 1 and eventos[0].sentido == SENTIDO_AB and eventos[0].frame == 3


def test_tocar_a_linha_e_voltar_nao_conta() -> None:
    c = ContadorLinha(A, B)
    _percorrer(c, 4, 2, [90, 100, 90])
    assert c.total == 0


def test_dois_sentidos_e_varias_classes() -> None:
    c = ContadorLinha(A, B)
    _percorrer(c, 1, 2, [80, 120])                 # carro A→B
    _percorrer(c, 2, 2, [120, 80])                 # carro B→A
    _percorrer(c, 3, 5, [80, 120])                 # ônibus A→B
    _percorrer(c, 4, 7, [120, 80])                 # caminhão B→A
    _percorrer(c, 5, 3, [80, 90])                  # moto não cruza
    _percorrer(c, 6, 2, [80, 120], x=500)          # carro fora do segmento
    tabela = c.por_classe()
    assert tabela["carro"] == {SENTIDO_AB: 1, SENTIDO_BA: 1, "total": 2}
    assert tabela["ônibus"][SENTIDO_AB] == 1
    assert tabela["caminhão"][SENTIDO_BA] == 1
    assert tabela["moto"]["total"] == 0
    assert c.total == 4
    assert c.total_por_sentido(SENTIDO_AB) == 2 and c.total_por_sentido(SENTIDO_BA) == 2


def test_esquecer_id_nao_permite_recontar() -> None:
    c = ContadorLinha(A, B)
    _percorrer(c, 1, 2, [80, 120])
    c.esquecer(1)
    _percorrer(c, 1, 2, [80, 120])
    assert c.total == 1


def test_reiniciar() -> None:
    c = ContadorLinha(A, B)
    _percorrer(c, 1, 2, [80, 120])
    c.reiniciar()
    assert c.total == 0 and c.eventos == [] and c.frame_ultimo_cruzamento is None


@pytest.mark.parametrize(("p1", "p2"), [((10, 10), (10, 10)), ((0, 0), (0.5, 0)), ((0, 0), (np.nan, 1))])
def test_linha_degenerada(p1, p2) -> None:
    with pytest.raises(ErroGeometria):
        ContadorLinha(p1, p2)


# ---------------------------------------------------------------------------
# Histórico de trajetórias
# ---------------------------------------------------------------------------
def test_historico_suaviza_limita_tamanho_e_vota_classe() -> None:
    h = HistoricoTrajetorias(alpha=0.5, tamanho_max=5)
    h.atualizar(1, 2, (0, 0), 0, 0.0)
    traj = h.atualizar(1, 7, (10, 0), 1, 0.1)
    assert traj.ponto_atual == (5, 0)                   # EMA com α = 0,5
    for i in range(2, 10):
        h.atualizar(1, 2, (10, 0), i, i / 10)
    assert len(h[1].pontos) == 5                        # só os 5 últimos
    assert h[1].classe == 2                             # carro venceu a votação
    assert h[1].velocidade_kmh is None                  # sem homografia


def test_historico_remove_ausentes() -> None:
    h = HistoricoTrajetorias(alpha=1.0, max_frames_ausente=3)
    h.atualizar(1, 2, (0, 0), 0, 0.0)
    h.atualizar(2, 2, (0, 0), 2, 0.2)
    assert h.remover_ausentes(4) == [1]
    assert 1 not in h and 2 in h and len(h) == 1
    assert 1 not in h.suavizador


def test_historico_velocidade_pela_homografia() -> None:
    # 100 px = 10 m nos dois eixos; veículo anda 5 px (0,5 m) a cada 0,05 s → 36 km/h
    homografia = calcular_homografia([(0, 0), (100, 0), (100, 100), (0, 100)], 10.0, 10.0)
    h = HistoricoTrajetorias(alpha=1.0, homografia=homografia, janela_velocidade_s=0.5)
    for i in range(15):
        traj = h.atualizar(1, 2, (50, 5 * i), i, 0.05 * i)
    assert traj.velocidade_kmh == pytest.approx(36.0)


def test_historico_velocidade_so_dentro_da_area_calibrada() -> None:
    area = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=float)
    homografia = calcular_homografia(area, 10.0, 10.0)
    h = HistoricoTrajetorias(alpha=1.0, homografia=homografia, area_calibrada_px=area)
    for i in range(10):
        traj = h.atualizar(1, 2, (300, 10 * i), i, 0.1 * i)   # sempre fora do trecho
    assert traj.velocidade_kmh is None
