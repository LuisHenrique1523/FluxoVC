"""Marcação com o mouse: linha de contagem, ROI e calibração clicando na prévia.

A interface mostra a prévia num componente que devolve a posição do clique.
Este módulo guarda a lógica (sem Streamlit): converte o clique em coordenadas
normalizadas, desenha os pontos já marcados, sabe quantos cliques cada modo
precisa, ordena os 4 cantos da calibração e valida o resultado com as mesmas
regras usadas na análise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import PontoNorm
from src.geometry.coordenadas import normalizados_para_pixels
from src.geometry.erros import ErroGeometria
from src.geometry.homografia import validar_quadrilatero
from src.geometry.roi import validar_roi
from src.models.contagem import ContadorLinha
from src.visualization.desenho import BRANCO, PRETO, escrever

COR_MARCACAO = (0, 140, 255)  # laranja (BGR): diferente da ROI (ciano), linha (amarelo) e calibração (magenta)


@dataclass(frozen=True)
class ModoMarcacao:
    chave: str
    rotulo: str
    minimo: int
    maximo: int | None      # None = quantos o usuário quiser (termina com "Concluir")
    instrucao: str


MODOS: dict[str, ModoMarcacao] = {
    "linha": ModoMarcacao(
        "linha", "Linha de contagem", 2, 2,
        "Clique em **2 pontos**: o início (A) e o fim (B) da linha. Ela deve atravessar a pista onde "
        "os veículos passam.",
    ),
    "roi": ModoMarcacao(
        "roi", "Região de interesse (ROI)", 3, None,
        "Clique nos **cantos da área da pista**, contornando-a (3 pontos ou mais). Depois clique em "
        "**Concluir ROI**. Veículos fora dela são ignorados.",
    ),
    "calibracao": ModoMarcacao(
        "calibracao", "Calibração da velocidade", 4, 4,
        "Clique nos **4 cantos de um trecho retangular da pista** (ex.: entre duas linhas de faixa e "
        "dois traços), em qualquer ordem. Depois informe a largura e o comprimento reais do trecho.",
    ),
}


def ponto_do_clique(valor: Mapping[str, float] | None) -> PontoNorm | None:
    """Clique do componente (``x``, ``y``, ``width``, ``height`` em px da tela) → ponto normalizado."""
    if not valor or not valor.get("width") or not valor.get("height"):
        return None
    x = min(1.0, max(0.0, float(valor["x"]) / float(valor["width"])))
    y = min(1.0, max(0.0, float(valor["y"]) / float(valor["height"])))
    return (round(x, 4), round(y, 4))


def completa(modo: str, pontos: Sequence[PontoNorm]) -> bool:
    """``True`` quando o modo já tem cliques suficientes para ser aplicado automaticamente."""
    m = MODOS[modo]
    return m.maximo is not None and len(pontos) >= m.maximo


def ordenar_quadrilatero(pontos: Sequence[PontoNorm]) -> list[PontoNorm]:
    """Ordena 4 cantos como a calibração espera: sup. esq., sup. dir., inf. dir., inf. esq.

    Os dois pontos mais altos da imagem formam o lado de cima; em cada lado, o
    de menor x é o esquerdo. Funciona para o trapézio típico de uma pista vista
    em perspectiva.
    """
    if len(pontos) != 4:
        raise ValueError("A calibração precisa de exatamente 4 pontos.")
    por_altura = sorted(pontos, key=lambda p: p[1])
    cima = sorted(por_altura[:2], key=lambda p: p[0])
    baixo = sorted(por_altura[2:], key=lambda p: p[0])
    return [cima[0], cima[1], baixo[1], baixo[0]]


def finalizar(modo: str, pontos: Sequence[PontoNorm], largura: int, altura: int) -> list[PontoNorm]:
    """Prepara os pontos para aplicar e valida com as regras da análise.

    Raises:
        ErroGeometria: linha degenerada, ROI sem área ou quadrilátero inválido.
    """
    pts = [tuple(p) for p in pontos]
    if modo == "linha":
        a, b = normalizados_para_pixels(pts[:2], largura, altura)
        ContadorLinha(tuple(a), tuple(b))
        return pts[:2]  # type: ignore[return-value]
    if modo == "roi":
        validar_roi(pts)
        return pts  # type: ignore[return-value]
    if modo == "calibracao":
        ordenados = ordenar_quadrilatero(pts)  # type: ignore[arg-type]
        validar_quadrilatero(normalizados_para_pixels(ordenados, largura, altura))
        return ordenados
    raise ValueError(f"Modo de marcação desconhecido: {modo}")


def desenhar_marcacao(imagem: np.ndarray, modo: str, pontos: Sequence[PontoNorm]) -> np.ndarray:
    """Cópia da prévia com os pontos já clicados, numerados e ligados."""
    saida = imagem.copy()
    if not pontos:
        return saida
    altura, largura = saida.shape[:2]
    px = np.round(normalizados_para_pixels(pontos, largura, altura)).astype(np.int32)
    raio = max(5, largura // 140)
    fechar = modo == "roi" and len(px) >= 3
    if len(px) >= 2:
        cv2.polylines(saida, [px.reshape(-1, 1, 2)], fechar, COR_MARCACAO, max(2, raio // 2), cv2.LINE_AA)
    for i, (x, y) in enumerate(px, start=1):
        cv2.circle(saida, (int(x), int(y)), raio + 2, BRANCO, -1, cv2.LINE_AA)
        cv2.circle(saida, (int(x), int(y)), raio, COR_MARCACAO, -1, cv2.LINE_AA)
        rotulo = ("A", "B")[i - 1] if modo == "linha" else str(i)
        escrever(saida, rotulo, (int(x) + raio + 4, int(y) - raio - 2), 0.6, PRETO, fundo=BRANCO)
    return saida


__all__ = [
    "MODOS",
    "ErroGeometria",
    "ModoMarcacao",
    "completa",
    "desenhar_marcacao",
    "finalizar",
    "ordenar_quadrilatero",
    "ponto_do_clique",
]
