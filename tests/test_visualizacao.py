"""Testes da Fase 4 — camadas de desenho, mapa de calor, bird's-eye e compositor."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.acquisition.fontes import Frame
from src.config import AppConfig, CORES_CLASSE
from src.metrics.desempenho import MedidorFPS
from src.models.analisador import AnalisadorFluxo, ResultadoFrame, VeiculoRastreado
from src.models.contagem import ContadorLinha
from src.models.rastreamento import Trajetoria
from src.visualization.birdseye import VistaBirdseye, colar_miniatura
from src.visualization.camadas import (
    COR_LINHA,
    COR_LINHA_ATIVA,
    desenhar_caixas,
    desenhar_hud,
    desenhar_linha_contagem,
    desenhar_roi,
    desenhar_trajetorias,
    linha_piscando,
    rotulo_veiculo,
)
from src.visualization.compositor import CompositorCamadas, contagem_estatica, painel_comparacao
from src.visualization.desenho import cor_id, misturar, texto_ascii
from src.visualization.mapa_calor import MapaCalor
from tests.conftest import desenhar_frame_sintetico
from tests.falsos import DetectorPorCor


def _preta(largura: int = 640, altura: int = 360) -> np.ndarray:
    return np.zeros((altura, largura, 3), dtype=np.uint8)


def _perto(pixel: np.ndarray, cor: tuple[int, int, int], tolerancia: int = 40) -> bool:
    """Compara cores com folga: o antialiasing (LINE_AA) mistura a borda com o fundo."""
    return bool((np.abs(pixel.astype(int) - np.array(cor)) <= tolerancia).all())


def _veiculo(id_: int | None = 1, classe: int = 2, caixa=(100, 100, 200, 180), vel: float | None = None):
    x1, y1, x2, y2 = caixa
    return VeiculoRastreado(id_, classe, 0.87, caixa, ((x1 + x2) / 2, y2), vel)


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
def test_texto_ascii_remove_acentos_e_setas() -> None:
    assert texto_ascii("ônibus caminhão A→B latência") == "onibus caminhao A->B latencia"


def test_cor_por_id_estavel_e_distinta() -> None:
    assert cor_id(5) == cor_id(5)
    cores = {cor_id(i) for i in range(1, 21)}
    assert len(cores) == 20


def test_misturar_com_e_sem_mascara() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    camada = np.full((4, 4, 3), 200, dtype=np.uint8)
    mascara = np.zeros((4, 4), dtype=bool)
    mascara[0, 0] = True
    misturar(base, camada, 0.5, mascara)
    assert base[0, 0, 0] == 100 and base[3, 3, 0] == 0
    misturar(base, camada, 1.0)
    assert (base == 200).all()


# ---------------------------------------------------------------------------
# Camadas
# ---------------------------------------------------------------------------
def test_rotulo_do_veiculo() -> None:
    assert rotulo_veiculo(_veiculo(12, 5, vel=54.4)) == "ônibus #12 0.87 54 km/h"
    assert rotulo_veiculo(_veiculo(None, 2)) == "carro 0.87"


def test_caixa_na_cor_da_classe_e_escala_de_exibicao() -> None:
    imagem = _preta(320, 180)
    desenhar_caixas(imagem, [_veiculo(1, 7, caixa=(100, 100, 200, 180))], escala=0.5)
    # escala 0,5: a borda esquerda vai de x=100 para x=50, a direita para x=100
    assert _perto(imagem[70, 50], CORES_CLASSE[7])
    assert _perto(imagem[70, 100], CORES_CLASSE[7])
    assert not imagem[70, 75].any()  # interior vazio


def test_trajetoria_desenhada_na_cor_do_id() -> None:
    imagem = _preta()
    traj = Trajetoria(3, deque([(50.0, 300.0), (150.0, 300.0), (250.0, 300.0)]))
    desenhar_trajetorias(imagem, [traj], escala=1.0)
    assert _perto(imagem[300, 100], cor_id(3))
    curta = Trajetoria(4, deque([(10.0, 10.0)]))
    desenhar_trajetorias(imagem, [curta], escala=1.0)  # 1 ponto só: nada a desenhar
    assert not imagem[10, 10].any()


@pytest.mark.parametrize(
    ("frame", "ultimo", "esperado"),
    [(10, None, False), (10, 10, True), (11, 10, True), (12, 10, False), (14, 10, True), (30, 10, False), (9, 10, False)],
)
def test_linha_piscando(frame: int, ultimo: int | None, esperado: bool) -> None:
    assert linha_piscando(frame, ultimo, duracao=12) is esperado


def test_linha_muda_de_cor_quando_ativa() -> None:
    normal, ativa = _preta(), _preta()
    desenhar_linha_contagem(normal, (0, 180), (640, 180), 1.0, ativa=False)
    desenhar_linha_contagem(ativa, (0, 180), (640, 180), 1.0, ativa=True)
    assert _perto(normal[180, 100], COR_LINHA)
    assert _perto(ativa[180, 100], COR_LINHA_ATIVA)
    assert (ativa[:, 100].any(axis=1)).sum() > (normal[:, 100].any(axis=1)).sum()  # mais grossa


def test_roi_semitransparente_so_dentro_do_poligono() -> None:
    imagem = np.full((200, 200, 3), 100, dtype=np.uint8)
    desenhar_roi(imagem, np.array([[50, 50], [150, 50], [150, 150], [50, 150]]), 1.0, opacidade=0.5)
    dentro, fora = imagem[100, 100].astype(int), imagem[10, 10]
    assert (fora == 100).all()
    assert not (dentro == 100).all() and (np.abs(dentro - 100) < 100).all()  # misturado, não opaco


def test_hud_escurece_canto_e_aceita_valores_vazios() -> None:
    imagem = np.full((360, 640, 3), 255, dtype=np.uint8)
    desenhar_hud(imagem, ContadorLinha((0, 0), (10, 0)).por_classe(), None, None)
    assert imagem[5, 5].max() < 255           # painel escuro no canto
    assert (imagem[350, 630] == 255).all()    # resto intacto


# ---------------------------------------------------------------------------
# Mapa de calor
# ---------------------------------------------------------------------------
def test_mapa_de_calor_acumula_pegada() -> None:
    mapa = MapaCalor(400, 300, reducao=1)
    assert mapa.vazio
    for _ in range(5):
        mapa.acumular([(100, 100, 160, 190)])   # pegada = y 160–190
    mapa.acumular([(300, 0, 350, 30)])
    assert mapa.acumulado[180, 130] == 5
    assert mapa.acumulado[120, 130] == 0       # parte de cima da caixa não conta
    assert mapa.acumulado[25, 320] == 1
    normal = mapa.normalizado()
    assert normal[180, 130] == 255 and 0 < normal[25, 320] < 255


def test_mapa_de_calor_sobrepoe_so_onde_houve_ocupacao() -> None:
    mapa = MapaCalor(400, 300)
    mapa.acumular([(100, 100, 160, 190)])
    imagem = np.full((300, 400, 3), 50, dtype=np.uint8)
    mapa.sobrepor(imagem, 0.5, "JET")
    assert not (imagem[180, 130] == 50).all()
    assert (imagem[10, 390] == 50).all()


def test_mapa_de_calor_decaimento_e_reinicio() -> None:
    mapa = MapaCalor(100, 100, reducao=1, decaimento=0.5)
    mapa.acumular([(0, 0, 10, 30)])
    mapa.acumular([])
    assert mapa.acumulado[25, 5] == pytest.approx(0.5)
    mapa.reiniciar()
    assert mapa.vazio


# ---------------------------------------------------------------------------
# Bird's-eye
# ---------------------------------------------------------------------------
def test_birdseye_desenha_veiculo_na_posicao_em_metros() -> None:
    vista = VistaBirdseye([(0, 0), (100, 0), (100, 200), (0, 200)], 10.0, 20.0, altura_px=200)
    assert vista.tamanho == (100, 200)  # 10 px por metro
    imagem = _preta(100, 200)
    veiculo = _veiculo(1, 5, caixa=(40, 80, 60, 100))  # base em (50, 100) = (5 m, 10 m)
    miniatura = vista.renderizar(imagem, [veiculo])
    assert _perto(miniatura[100, 50], CORES_CLASSE[5])


def test_colar_miniatura_no_canto_superior_direito() -> None:
    imagem = _preta(300, 200)
    colar_miniatura(imagem, np.full((50, 40, 3), 77, dtype=np.uint8), margem=8)
    assert (imagem[8 + 25, 300 - 8 - 20] == 77).all()
    gigante = np.full((500, 500, 3), 77, dtype=np.uint8)
    colar_miniatura(_preta(300, 200), gigante)  # maior que a imagem: é reduzida, sem erro


# ---------------------------------------------------------------------------
# Compositor e comparação
# ---------------------------------------------------------------------------
def _cfg_tudo(ligado: bool) -> AppConfig:
    cfg = AppConfig()
    vis = cfg.visualizacao
    for campo in ("caixas", "trajetorias", "linha", "roi", "heatmap", "birdseye", "hud"):
        setattr(vis, f"mostrar_{campo}", ligado)
    return cfg


def test_compositor_com_camadas_desligadas_devolve_frame_reduzido() -> None:
    imagem = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    comp = CompositorCamadas(_cfg_tudo(False), 1920, 1080)
    saida = comp.compor(imagem, ResultadoFrame(0, 0.0, True, [_veiculo()]))
    assert saida.shape == (540, 960, 3)                      # largura de exibição padrão
    assert comp.escala == pytest.approx(0.5)
    imagem_pequena = imagem[:360, :640].copy()
    comp = CompositorCamadas(_cfg_tudo(False), 640, 360)
    np.testing.assert_array_equal(comp.compor(imagem_pequena, ResultadoFrame(0, 0.0, True)), imagem_pequena)


def test_compositor_nao_altera_original_e_acumula_calor_so_em_frames_processados() -> None:
    imagem = _preta()
    comp = CompositorCamadas(_cfg_tudo(True), 640, 360)
    comp.compor(imagem, ResultadoFrame(0, 0.0, False, [_veiculo()]))
    assert comp.mapa_calor.vazio                              # frame pulado não acumula
    saida = comp.compor(imagem, ResultadoFrame(1, 0.1, True, [_veiculo()]))
    assert not comp.mapa_calor.vazio
    assert not imagem.any()                                   # original intacto
    assert saida.any()


def test_compositor_ponta_a_ponta_com_video_sintetico() -> None:
    cfg = _cfg_tudo(True)
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.5), (1.0, 0.5)
    analisador = AnalisadorFluxo(cfg, DetectorPorCor(), 640, 360)
    comp = CompositorCamadas(cfg, 640, 360)
    piscou = False
    for i in range(40):
        imagem = desenhar_frame_sintetico(i, 640, 360)
        resultado = analisador.processar(Frame(i, i / 15, imagem))
        saida = comp.compor(imagem, resultado, analisador.contador, analisador.historico)
        assert saida.shape == (360, 640, 3)
        piscou |= bool((saida[180] == COR_LINHA_ATIVA).all(axis=1).any())
    assert piscou and analisador.contador.total == 2
    assert comp.tempo_desenho_ms > 0 and comp.medidor_fps.fps is not None


def test_contagem_estatica_para_imagem_unica() -> None:
    tabela = contagem_estatica([_veiculo(None, 2), _veiculo(None, 2), _veiculo(None, 7)])
    assert tabela["carro"]["total"] == 2 and tabela["caminhão"]["total"] == 1
    assert tabela["moto"]["total"] == 0


def test_painel_comparacao_mesma_altura() -> None:
    painel = painel_comparacao(_preta(1280, 720), _preta(640, 640), _preta(960, 540), altura=200)
    # 3 painéis (356 + 200 + 356 px de largura) + 2 separadores de 6 px; altura + título de 28 px
    assert painel.shape == (228, 356 + 200 + 356 + 12, 3)


def test_medidor_fps() -> None:
    medidor = MedidorFPS(alpha=0.5)
    assert medidor.registrar(0.0) is None
    assert medidor.registrar(0.1) == pytest.approx(10.0)
    assert medidor.registrar(0.15) == pytest.approx(15.0)  # 0,5·20 + 0,5·10
    medidor.reiniciar()
    assert medidor.fps is None
