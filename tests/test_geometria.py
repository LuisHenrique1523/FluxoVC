"""Testes da Fase 2 — geometria: letterbox, coordenadas, ROI, homografia e EMA."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.geometry.coordenadas import (
    caixas_modelo_para_original,
    calcular_letterbox,
    escala_exibicao,
    exibicao_para_original,
    modelo_para_original,
    normalizados_para_pixels,
    original_para_exibicao,
    original_para_modelo,
    pixels_para_normalizados,
)
from src.geometry.erros import HomografiaInvalidaError, RoiInvalidaError
from src.geometry.homografia import (
    calcular_homografia,
    estimar_velocidade_kmh,
    gerar_birdseye,
    matriz_birdseye,
    transformar_pontos,
    validar_quadrilatero,
)
from src.geometry.roi import ponto_na_roi, roi_em_pixels, validar_roi
from src.geometry.suavizacao import SuavizadorEMA, ema, validar_alpha
from src.preprocessing.operacoes import aplicar_letterbox

RESOLUCOES = [(1920, 1080), (1280, 720), (640, 480), (720, 1280), (333, 777), (640, 640)]


# ---------------------------------------------------------------------------
# Letterbox e seu inverso
# ---------------------------------------------------------------------------
def test_parametros_letterbox_paisagem() -> None:
    p = calcular_letterbox(1280, 720, 640)
    assert (p.nova_largura, p.nova_altura) == (640, 360)
    assert (p.pad_x, p.pad_y) == (0, 140)
    assert p.escala_x == pytest.approx(0.5) and p.escala_y == pytest.approx(0.5)


def test_parametros_letterbox_retrato() -> None:
    p = calcular_letterbox(720, 1280, 640)
    assert (p.nova_largura, p.nova_altura) == (360, 640)
    assert (p.pad_x, p.pad_y) == (140, 0)


def test_letterbox_sem_manter_proporcao_estica() -> None:
    p = calcular_letterbox(1280, 720, 640, manter_proporcao=False)
    assert (p.pad_x, p.pad_y) == (0, 0)
    assert p.escala_x == pytest.approx(0.5) and p.escala_y == pytest.approx(640 / 720)


@pytest.mark.parametrize("dimensoes", [(0, 480, 640), (640, -1, 640), (640, 480, 0)])
def test_letterbox_dimensoes_invalidas(dimensoes: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        calcular_letterbox(*dimensoes)


@pytest.mark.parametrize(("largura", "altura"), RESOLUCOES)
@pytest.mark.parametrize("manter", [True, False])
def test_ida_e_volta_devolve_mesma_coordenada(largura: int, altura: int, manter: bool) -> None:
    p = calcular_letterbox(largura, altura, 640, manter)
    rng = np.random.default_rng(42)
    pontos = rng.uniform([0, 0], [largura, altura], size=(200, 2))
    volta = modelo_para_original(original_para_modelo(pontos, p), p)
    assert np.abs(volta - pontos).max() <= 1.0


@pytest.mark.parametrize(("largura", "altura"), RESOLUCOES)
def test_cantos_do_frame_caem_nas_bordas_do_conteudo(largura: int, altura: int) -> None:
    p = calcular_letterbox(largura, altura, 640)
    cantos = original_para_modelo([[0, 0], [largura, altura]], p)
    np.testing.assert_allclose(cantos[0], [p.pad_x, p.pad_y])
    np.testing.assert_allclose(cantos[1], [p.pad_x + p.nova_largura, p.pad_y + p.nova_altura])


def test_ponto_desenhado_e_reencontrado_apos_letterbox_real() -> None:
    """Desenha um quadrado, aplica o letterbox de verdade e volta: erro ≤ 1 px."""
    largura, altura = 1280, 720
    imagem = np.zeros((altura, largura, 3), dtype=np.uint8)
    x0, y0, lado = 900, 200, 20
    imagem[y0 : y0 + lado, x0 : x0 + lado] = 255
    centro_original = np.array([x0 + lado / 2, y0 + lado / 2])

    saida, p = aplicar_letterbox(imagem, 640)
    ys, xs = np.nonzero(saida[:, :, 0] > 127)
    centro_modelo = np.array([xs.mean() + 0.5, ys.mean() + 0.5])  # centro contínuo do pixel

    reencontrado = modelo_para_original(centro_modelo, p)[0]
    assert np.abs(reencontrado - centro_original).max() <= 1.0


def test_caixas_modelo_para_original_e_limitadas_as_faixas() -> None:
    p = calcular_letterbox(1280, 720, 640)  # faixa de 140 px no topo e embaixo
    caixas = np.array([[100, 240, 200, 340], [0, 0, 50, 150]])  # 2ª começa dentro da faixa
    convertidas = caixas_modelo_para_original(caixas, p)
    np.testing.assert_allclose(convertidas[0], [200, 200, 400, 400])
    np.testing.assert_allclose(convertidas[1], [0, 0, 100, 20])


def test_caixas_vazias() -> None:
    p = calcular_letterbox(1280, 720, 640)
    assert caixas_modelo_para_original(np.empty((0, 4)), p).shape == (0, 4)


def test_cadeia_modelo_original_exibicao() -> None:
    p = calcular_letterbox(1920, 1080, 640)
    escala = escala_exibicao(1920, 960)
    assert escala == pytest.approx(0.5)
    ponto_modelo = [320, 320]  # centro do quadrado = centro do frame
    original = modelo_para_original(ponto_modelo, p)
    np.testing.assert_allclose(original, [[960, 540]])
    exibido = original_para_exibicao(original, escala)
    np.testing.assert_allclose(exibido, [[480, 270]])
    np.testing.assert_allclose(exibicao_para_original(exibido, escala), original)


def test_escala_exibicao_nao_amplia() -> None:
    assert escala_exibicao(640, 960) == 1.0


def test_normalizados_e_pixels() -> None:
    px = normalizados_para_pixels([(0.5, 0.25), (1.0, 1.0)], 1280, 720)
    np.testing.assert_allclose(px, [[640, 180], [1280, 720]])
    np.testing.assert_allclose(pixels_para_normalizados(px, 1280, 720), [[0.5, 0.25], [1, 1]])


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def test_roi_valida() -> None:
    roi = validar_roi([(0, 0.4), (1, 0.4), (1, 1), (0, 1)])
    assert roi.shape == (4, 2)
    px = roi_em_pixels(roi, 1000, 500)
    assert px.dtype == np.int32
    np.testing.assert_array_equal(px[1], [1000, 200])


@pytest.mark.parametrize(
    "pontos",
    [
        [],
        [(0.1, 0.1), (0.9, 0.9)],                       # menos de 3 pontos
        [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],           # alinhados (sem área)
        [(0.1, 0.1), (1.5, 0.1), (0.5, 0.9)],           # fora do frame
        [(0.1, 0.1), (math.nan, 0.1), (0.5, 0.9)],      # valor inválido
        "abc",
    ],
)
def test_roi_invalida(pontos: object) -> None:
    with pytest.raises(RoiInvalidaError):
        validar_roi(pontos)  # type: ignore[arg-type]


def test_ponto_na_roi() -> None:
    poligono = np.array([[0, 100], [200, 100], [200, 200], [0, 200]])
    assert ponto_na_roi((100, 150), poligono)
    assert ponto_na_roi((200, 150), poligono)  # na borda conta como dentro
    assert not ponto_na_roi((100, 50), poligono)


# ---------------------------------------------------------------------------
# Homografia
# ---------------------------------------------------------------------------
# Trapézio típico de câmera olhando a pista em perspectiva
TRAPEZIO = [(400, 200), (600, 200), (900, 600), (100, 600)]


def test_homografia_leva_cantos_ao_retangulo_em_metros() -> None:
    h = calcular_homografia(TRAPEZIO, largura_m=7.0, comprimento_m=20.0)
    np.testing.assert_allclose(
        transformar_pontos(h, TRAPEZIO), [[0, 0], [7, 0], [7, 20], [0, 20]], atol=1e-6
    )


def test_homografia_interseccao_das_diagonais_vai_ao_centro() -> None:
    """A homografia preserva retas: o cruzamento das diagonais vira o centro do retângulo."""
    h = calcular_homografia(TRAPEZIO, 7.0, 20.0)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = TRAPEZIO
    # interseção das retas p1–p3 e p2–p4
    denominador = (x1 - x3) * (y2 - y4) - (y1 - y3) * (x2 - x4)
    a, b = x1 * y3 - y1 * x3, x2 * y4 - y2 * x4
    centro = ((a * (x2 - x4) - (x1 - x3) * b) / denominador, (a * (y2 - y4) - (y1 - y3) * b) / denominador)
    np.testing.assert_allclose(transformar_pontos(h, [centro]), [[3.5, 10.0]], atol=1e-6)


def test_homografia_identidade_em_escala() -> None:
    """Retângulo de 100×200 px mapeado para 10×20 m: 1 m = 10 px em todo o plano."""
    h = calcular_homografia([(0, 0), (100, 0), (100, 200), (0, 200)], 10.0, 20.0)
    np.testing.assert_allclose(transformar_pontos(h, [(50, 50), (30, 170)]), [[5, 5], [3, 17]], atol=1e-9)


@pytest.mark.parametrize(
    "pontos",
    [
        [(0, 0), (100, 0), (200, 0), (0, 100)],            # três alinhados
        [(0, 0), (100, 0), (0, 100), (100, 100)],          # ordem cruzada ("gravata")
        [(0, 0), (100, 0), (100, 100)],                    # só 3 pontos
        [(0, 0), (0, 0), (100, 100), (0, 100)],            # ponto repetido
        [(0, 0), (5, 0), (5, 5), (0, 5)],                  # área minúscula
        [(0, 0), (100, 0), (100, math.inf), (0, 100)],     # valor inválido
        [(0, 0), (100, 0), (40, 40), (0, 100)],            # côncavo
    ],
)
def test_homografia_degenerada(pontos: list[tuple[float, float]]) -> None:
    with pytest.raises(HomografiaInvalidaError):
        calcular_homografia(pontos, 7.0, 20.0)


@pytest.mark.parametrize(("largura", "comprimento"), [(0, 20), (7, -1), (math.nan, 20)])
def test_homografia_medidas_invalidas(largura: float, comprimento: float) -> None:
    with pytest.raises(HomografiaInvalidaError, match="positivo"):
        calcular_homografia(TRAPEZIO, largura, comprimento)


def test_quadrilatero_aceita_sentido_anti_horario() -> None:
    assert validar_quadrilatero(TRAPEZIO[::-1]).shape == (4, 2)


def test_transformar_lista_vazia() -> None:
    h = calcular_homografia(TRAPEZIO, 7.0, 20.0)
    assert transformar_pontos(h, []).shape == (0, 2)


def test_birdseye_tamanho_e_conteudo() -> None:
    imagem = np.zeros((720, 1000, 3), dtype=np.uint8)
    cv2.fillPoly(imagem, [np.array(TRAPEZIO, dtype=np.int32)], (0, 255, 0))
    matriz, tamanho = matriz_birdseye(TRAPEZIO, 7.0, 20.0, px_por_metro=10)
    assert tamanho == (70, 200)
    vista = gerar_birdseye(imagem, matriz, tamanho)
    assert vista.shape == (200, 70, 3)
    # o trapézio verde vira um retângulo que ocupa praticamente a vista inteira
    assert (vista[5:-5, 5:-5, 1] > 200).mean() > 0.99


def test_velocidade() -> None:
    # 10 m em 1 s = 36 km/h
    assert estimar_velocidade_kmh([(0, 0), (0, 5), (0, 10)], [0.0, 0.5, 1.0]) == pytest.approx(36.0)
    assert estimar_velocidade_kmh([(0, 0)], [0.0]) is None                # 1 ponto só
    assert estimar_velocidade_kmh([(0, 0), (0, 1)], [0.0, 0.1]) is None   # Δt curto demais


def test_velocidade_ponta_a_ponta_com_homografia() -> None:
    """Veículo percorre o trecho calibrado inteiro (20 m) em 2 s → 36 km/h."""
    h = calcular_homografia(TRAPEZIO, 7.0, 20.0)
    inicio, fim = (500, 200), (500, 600)  # centro da borda de cima → centro da borda de baixo
    posicoes = transformar_pontos(h, [inicio, fim])
    assert estimar_velocidade_kmh(posicoes, [0.0, 2.0]) == pytest.approx(36.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Suavização EMA
# ---------------------------------------------------------------------------
def test_ema_passo_unico() -> None:
    np.testing.assert_allclose(ema([0, 0], [10, 20], 0.25), [2.5, 5.0])


@pytest.mark.parametrize("alpha", [0, -0.1, 1.01, math.nan])
def test_alpha_invalido(alpha: float) -> None:
    with pytest.raises(ValueError):
        validar_alpha(alpha)


def test_suavizador_primeiro_ponto_e_constante() -> None:
    s = SuavizadorEMA(0.3)
    assert s.atualizar(1, (100, 50)) == (100, 50)
    for _ in range(5):
        assert s.atualizar(1, (100, 50)) == pytest.approx((100, 50))


def test_suavizador_degrau_converge() -> None:
    s = SuavizadorEMA(0.5)
    s.atualizar("a", (0, 0))
    valores = [s.atualizar("a", (100, 0))[0] for _ in range(10)]
    assert valores[:3] == pytest.approx([50, 75, 87.5])
    assert all(a < b for a, b in zip(valores, valores[1:]))  # monotônico
    assert valores[-1] == pytest.approx(100, abs=0.2)


def test_suavizador_alpha_um_nao_suaviza() -> None:
    s = SuavizadorEMA(1.0)
    s.atualizar(1, (0, 0))
    assert s.atualizar(1, (7, 9)) == (7, 9)


def test_suavizador_reduz_tremor() -> None:
    rng = np.random.default_rng(0)
    verdade = np.column_stack((np.linspace(0, 300, 100), np.full(100, 200.0)))
    ruidoso = verdade + rng.normal(0, 4, verdade.shape)
    s = SuavizadorEMA(0.3)
    suave = np.array([s.atualizar(7, p) for p in ruidoso])
    tremor = lambda pts: np.abs(np.diff(pts[:, 1])).mean()  # noqa: E731
    assert tremor(suave) < 0.5 * tremor(ruidoso)


def test_suavizador_ids_independentes_e_limpeza() -> None:
    s = SuavizadorEMA(0.5)
    s.atualizar(1, (0, 0))
    s.atualizar(2, (100, 100))
    assert s.atualizar(1, (10, 10)) == (5, 5)
    assert s.obter(2) == (100, 100)
    s.manter_apenas([1])
    assert 2 not in s and 1 in s and len(s) == 1
    s.remover(1)
    s.remover(99)  # remover ID inexistente não dá erro
    assert len(s) == 0 and s.obter(1) is None
