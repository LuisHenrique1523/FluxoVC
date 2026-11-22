"""Testes da Fase 2 — operações de pré-processamento e seu encadeamento."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import GeometriaConfig, PreprocConfig
from src.geometry.erros import RoiInvalidaError
from src.preprocessing.operacoes import (
    COR_PREENCHIMENTO,
    aplicar_clahe,
    aplicar_letterbox,
    aplicar_mascara,
    criar_mascara_roi,
    deve_processar_frame,
    kernel_impar,
    reduzir_ruido,
)
from src.preprocessing.pipeline import EXPLICACOES, preprocessar


def _imagem(largura: int = 1280, altura: int = 720, valor: int = 200) -> np.ndarray:
    return np.full((altura, largura, 3), valor, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------
def test_letterbox_forma_e_faixas() -> None:
    saida, p = aplicar_letterbox(_imagem(1280, 720), 640)
    assert saida.shape == (640, 640, 3) and saida.dtype == np.uint8
    assert (p.pad_y, p.nova_altura) == (140, 360)
    np.testing.assert_array_equal(saida[:140], 114)           # faixa de cima
    np.testing.assert_array_equal(saida[500:], 114)           # faixa de baixo
    np.testing.assert_array_equal(saida[140:500], 200)        # conteúdo


def test_letterbox_amplia_imagem_pequena() -> None:
    saida, p = aplicar_letterbox(_imagem(320, 240), 640)
    assert saida.shape == (640, 640, 3)
    assert p.escala_x == pytest.approx(2.0) and (p.pad_x, p.pad_y) == (0, 80)


def test_letterbox_esticado_nao_tem_faixas() -> None:
    saida, _ = aplicar_letterbox(_imagem(1280, 720), 640, manter_proporcao=False)
    np.testing.assert_array_equal(saida, 200)


def test_letterbox_nao_altera_entrada() -> None:
    imagem = _imagem()
    copia = imagem.copy()
    aplicar_letterbox(imagem, 640)
    np.testing.assert_array_equal(imagem, copia)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def test_mascara_e_aplicacao_da_roi() -> None:
    imagem = _imagem(200, 100, valor=10)
    poligono = np.array([[50, 0], [150, 0], [150, 100], [50, 100]])
    mascara = criar_mascara_roi(imagem.shape[:2], poligono)
    assert mascara.dtype == np.uint8 and set(np.unique(mascara)) == {0, 255}
    saida = aplicar_mascara(imagem, mascara)
    np.testing.assert_array_equal(saida[50, 100], [10, 10, 10])           # dentro: intacto
    np.testing.assert_array_equal(saida[50, 10], COR_PREENCHIMENTO)       # fora: cinza
    np.testing.assert_array_equal(imagem[50, 10], [10, 10, 10])           # entrada intacta


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------
def test_clahe_aumenta_contraste_de_cena_escura() -> None:
    rng = np.random.default_rng(1)
    escura = rng.integers(20, 50, (240, 320, 3), dtype=np.uint8)  # baixo contraste
    saida = aplicar_clahe(escura, clip=3.0, grade=8)
    assert saida.shape == escura.shape and saida.dtype == np.uint8
    l_antes = cv2.cvtColor(escura, cv2.COLOR_BGR2LAB)[:, :, 0]
    l_depois = cv2.cvtColor(saida, cv2.COLOR_BGR2LAB)[:, :, 0]
    assert l_depois.std() > 1.5 * l_antes.std()


def test_clahe_preserva_tonalidade() -> None:
    """Equalizar só L mantém os canais de cor (a, b) praticamente iguais."""
    rng = np.random.default_rng(2)
    imagem = np.zeros((240, 320, 3), dtype=np.uint8)
    imagem[:] = (40, 40, 120)  # avermelhado escuro
    imagem = cv2.add(imagem, rng.integers(0, 15, imagem.shape, dtype=np.uint8))
    lab_antes = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(int)
    lab_depois = cv2.cvtColor(aplicar_clahe(imagem), cv2.COLOR_BGR2LAB).astype(int)
    assert np.abs(lab_depois[:, :, 1:] - lab_antes[:, :, 1:]).mean() < 6


@pytest.mark.parametrize(("clip", "grade"), [(0, 8), (2.0, 0)])
def test_clahe_parametros_invalidos(clip: float, grade: int) -> None:
    with pytest.raises(ValueError):
        aplicar_clahe(_imagem(320, 240), clip, grade)


# ---------------------------------------------------------------------------
# Redução de ruído
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tipo", ["bilateral", "gaussiano"])
def test_reducao_de_ruido_diminui_variacao(tipo: str) -> None:
    rng = np.random.default_rng(3)
    limpa = np.full((240, 320, 3), 128, dtype=np.int16)
    ruidosa = np.clip(limpa + rng.normal(0, 15, limpa.shape), 0, 255).astype(np.uint8)
    saida = reduzir_ruido(ruidosa, tipo, 5)
    assert saida.shape == ruidosa.shape and saida.dtype == np.uint8
    assert saida.astype(float).std() < 0.7 * ruidosa.astype(float).std()


def test_bilateral_preserva_borda_melhor_que_gaussiano() -> None:
    imagem = np.zeros((100, 100, 3), dtype=np.uint8)
    imagem[:, 50:] = 255  # borda vertical nítida
    degrau = lambda img: int(img[50, 52, 0]) - int(img[50, 47, 0])  # noqa: E731
    assert degrau(reduzir_ruido(imagem, "bilateral", 9)) > degrau(reduzir_ruido(imagem, "gaussiano", 9))


@pytest.mark.parametrize(("entrada", "esperado"), [(1, 3), (3, 3), (4, 5), (5, 5), (8, 9)])
def test_kernel_sempre_impar(entrada: int, esperado: int) -> None:
    assert kernel_impar(entrada) == esperado


def test_filtro_desconhecido() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        reduzir_ruido(_imagem(320, 240), "mediana", 5)


# ---------------------------------------------------------------------------
# Frame skip
# ---------------------------------------------------------------------------
def test_frame_skip() -> None:
    assert [i for i in range(10) if deve_processar_frame(i, 0)] == list(range(10))
    assert [i for i in range(10) if deve_processar_frame(i, 2)] == [0, 3, 6, 9]
    with pytest.raises(ValueError):
        deve_processar_frame(0, -1)


# ---------------------------------------------------------------------------
# Encadeamento
# ---------------------------------------------------------------------------
def test_pipeline_com_tudo_ligado() -> None:
    cfg = PreprocConfig(usar_roi=True, usar_clahe=True, usar_reducao_ruido=True)
    roi = GeometriaConfig().roi  # metade de baixo do frame (y ≥ 0,35)
    resultado = preprocessar(_imagem(1280, 720), cfg, roi)

    assert resultado.imagem.shape == (640, 640, 3)
    assert set(resultado.tempos_ms) == {"letterbox", "clahe", "ruido", "roi"}
    assert resultado.tempo_total_ms >= 0
    assert resultado.roi_modelo is not None and resultado.roi_modelo.shape == (4, 2)
    # y = 0,35 · 720 = 252 px no original → 252 · 0,5 + 140 = 266 px no modelo
    assert resultado.roi_modelo[:, 1].min() == 266
    np.testing.assert_array_equal(resultado.imagem[200, 320], COR_PREENCHIMENTO)  # acima da ROI


def test_pipeline_so_letterbox() -> None:
    cfg = PreprocConfig(usar_roi=False, usar_clahe=False, usar_reducao_ruido=False)
    resultado = preprocessar(_imagem(), cfg, GeometriaConfig().roi)
    assert set(resultado.tempos_ms) == {"letterbox"}
    assert resultado.roi_modelo is None


def test_pipeline_roi_invalida() -> None:
    with pytest.raises(RoiInvalidaError):
        preprocessar(_imagem(), PreprocConfig(usar_roi=True), [(0.1, 0.1), (0.2, 0.2)])


def test_explicacoes_para_cada_toggle() -> None:
    assert {"letterbox", "roi", "clahe", "ruido", "frame_skip"} <= set(EXPLICACOES)
    assert all(len(texto) > 40 for texto in EXPLICACOES.values())
