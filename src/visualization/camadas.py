"""Camadas de desenho sobre o frame de exibição.

Todas as funções recebem coordenadas no **espaço do frame original** e o fator
``escala`` (original → exibição); a conversão é feita aqui, com
:func:`original_para_exibicao`. Elas desenham **na própria imagem recebida**
(in-place), para não copiar o frame a cada camada.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.geometry.coordenadas import original_para_exibicao
from src.models.analisador import VeiculoRastreado
from src.models.rastreamento import Trajetoria
from src.visualization.desenho import (
    BRANCO,
    PRETO,
    Cor,
    cor_classe,
    cor_id,
    escrever,
    espessura,
    misturar,
    texto_ascii,
)

COR_LINHA: Cor = (0, 255, 255)          # amarelo
COR_LINHA_ATIVA: Cor = (0, 255, 0)      # verde ao piscar
COR_ROI: Cor = (255, 200, 0)            # ciano


def _inteiros(pontos: Sequence[Sequence[float]] | np.ndarray, escala: float) -> np.ndarray:
    return np.round(original_para_exibicao(pontos, escala)).astype(np.int32)


# ---------------------------------------------------------------------------
# Caixas
# ---------------------------------------------------------------------------
def rotulo_veiculo(veiculo: VeiculoRastreado) -> str:
    """Texto da caixa: classe, ID, confiança e velocidade (quando houver)."""
    partes = [veiculo.nome_classe]
    if veiculo.id_rastreio is not None:
        partes.append(f"#{veiculo.id_rastreio}")
    partes.append(f"{veiculo.confianca:.2f}")
    if veiculo.velocidade_kmh is not None:
        partes.append(f"{veiculo.velocidade_kmh:.0f} km/h")
    return " ".join(partes)


def desenhar_caixas(imagem: np.ndarray, veiculos: Iterable[VeiculoRastreado], escala: float) -> None:
    """Caixa com a cor da classe e rótulo com fundo da mesma cor."""
    traco = espessura(imagem.shape[1])
    tamanho_fonte = 0.4 * imagem.shape[1] / 960 + 0.1
    for v in veiculos:
        (x1, y1), (x2, y2) = _inteiros([v.caixa[:2], v.caixa[2:]], escala)
        cor = cor_classe(v.classe)
        cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, traco, cv2.LINE_AA)
        # rótulo acima da caixa; se não couber (topo da imagem), vai para dentro
        y_texto = y1 - 5 if y1 > 18 else y1 + 15
        escrever(imagem, rotulo_veiculo(v), (x1 + 2, y_texto), tamanho_fonte, PRETO, fundo=cor)


# ---------------------------------------------------------------------------
# Trajetórias
# ---------------------------------------------------------------------------
def desenhar_trajetorias(imagem: np.ndarray, trajetorias: Iterable[Trajetoria], escala: float) -> None:
    """Polilinha suavizada por ID; o trecho mais recente é mais grosso."""
    traco_max = espessura(imagem.shape[1], 1.5)
    for traj in trajetorias:
        if len(traj.pontos) < 2:
            continue
        pontos = _inteiros(list(traj.pontos), escala)
        cor = cor_id(int(traj.id_rastreio)) if isinstance(traj.id_rastreio, (int, np.integer)) else BRANCO
        n = len(pontos)
        for i in range(1, n):
            traco = max(1, round(traco_max * i / n))  # "cauda" afinando no passado
            cv2.line(imagem, tuple(pontos[i - 1]), tuple(pontos[i]), cor, traco, cv2.LINE_AA)
        cv2.circle(imagem, tuple(pontos[-1]), traco_max + 1, cor, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Linha de contagem
# ---------------------------------------------------------------------------
def linha_piscando(frame_atual: int, frame_ultimo_cruzamento: int | None, duracao: int) -> bool:
    """Liga/desliga a cada 2 frames durante ``duracao`` frames após um cruzamento."""
    if frame_ultimo_cruzamento is None:
        return False
    decorrido = frame_atual - frame_ultimo_cruzamento
    return 0 <= decorrido < duracao and (decorrido // 2) % 2 == 0


def desenhar_linha_contagem(
    imagem: np.ndarray,
    p1: Sequence[float],
    p2: Sequence[float],
    escala: float,
    ativa: bool = False,
) -> None:
    """Linha virtual com marcação dos lados A e B; fica verde e grossa quando ``ativa``."""
    a, b = _inteiros([p1, p2], escala)
    base = espessura(imagem.shape[1])
    cor, traco = (COR_LINHA_ATIVA, base * 3) if ativa else (COR_LINHA, base)
    cv2.line(imagem, tuple(a), tuple(b), cor, traco, cv2.LINE_AA)
    for ponta in (a, b):
        cv2.circle(imagem, tuple(ponta), base + 2, cor, -1, cv2.LINE_AA)

    # Lado A = produto vetorial negativo; B = positivo. A normal (−dy, dx) aponta para B.
    direcao = (b - a).astype(float)
    comprimento = float(np.linalg.norm(direcao))
    if comprimento < 1:
        return
    normal = np.array([-direcao[1], direcao[0]]) / comprimento
    meio = (a + b) / 2
    distancia = 14 * base
    tamanho_fonte = 0.35 * imagem.shape[1] / 960 + 0.15
    for rotulo, sinal in (("A", -1), ("B", 1)):
        x, y = meio + sinal * distancia * normal
        escrever(imagem, rotulo, (int(x) - 5, int(y) + 5), tamanho_fonte, PRETO, fundo=cor)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def desenhar_roi(imagem: np.ndarray, poligono_px: np.ndarray, escala: float, opacidade: float) -> None:
    """Preenchimento semitransparente (``addWeighted``) só dentro do polígono + contorno."""
    pontos = _inteiros(poligono_px, escala).reshape(-1, 1, 2)
    camada = np.empty_like(imagem)
    camada[:] = COR_ROI
    mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [pontos], 255)
    misturar(imagem, camada, opacidade, mascara)
    cv2.polylines(imagem, [pontos], True, COR_ROI, espessura(imagem.shape[1]), cv2.LINE_AA)


def desenhar_calibracao(imagem: np.ndarray, pontos_px: np.ndarray, escala: float) -> None:
    """Quadrilátero da calibração (velocidade) com os pontos numerados 1–4 (prévia)."""
    cor = (255, 0, 255)  # magenta
    pontos = _inteiros(pontos_px, escala)
    cv2.polylines(imagem, [pontos.reshape(-1, 1, 2)], True, cor, espessura(imagem.shape[1]), cv2.LINE_AA)
    for i, (x, y) in enumerate(pontos, start=1):
        cv2.circle(imagem, (int(x), int(y)), espessura(imagem.shape[1]) + 3, cor, -1, cv2.LINE_AA)
        escrever(imagem, str(i), (int(x) + 8, int(y) - 6), 0.5, BRANCO, fundo=cor)


# ---------------------------------------------------------------------------
# Painel HUD
# ---------------------------------------------------------------------------
def desenhar_hud(
    imagem: np.ndarray,
    contagem_por_classe: Mapping[str, Mapping[str, int]],
    fps: float | None,
    latencia_ms: float | None,
    opacidade: float = 0.6,
) -> None:
    """Painel no canto superior esquerdo: contagem por classe/sentido, FPS e latência."""
    tamanho_fonte = 0.45 * imagem.shape[1] / 960 + 0.1
    altura_linha = int(26 * imagem.shape[1] / 960) + 6
    margem = altura_linha // 2

    # Tabela: cada célula é escrita numa coluna de x fixo (as fontes Hershey
    # não são monoespaçadas, então alinhar com espaços não funciona).
    amarelo, verde = (0, 255, 255), (180, 255, 180)
    tabela: list[tuple[list[str], tuple[int, int, int]]] = [(["CLASSE", "A->B", "B->A", "TOTAL"], BRANCO)]
    total = {"A→B": 0, "B→A": 0, "total": 0}
    for nome, valores in contagem_por_classe.items():
        tabela.append(([nome] + [str(valores.get(c, 0)) for c in total], BRANCO))
        for chave in total:
            total[chave] += valores.get(chave, 0)
    tabela.append((["TOTAL"] + [str(v) for v in total.values()], amarelo))
    fps_txt = "--" if fps is None else f"{fps:.1f}"
    lat_txt = "--" if latencia_ms is None else f"{latencia_ms:.0f} ms"
    rodape = f"FPS {fps_txt}   latencia {lat_txt}"

    def largura_texto(texto: str) -> int:
        return cv2.getTextSize(texto_ascii(texto), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, 1)[0][0]

    # x de cada coluna: largura máxima da coluna anterior + espaçamento
    colunas_x = [margem]
    for coluna in range(3):
        maior = max(largura_texto(linha[coluna]) for linha, _ in tabela)
        colunas_x.append(colunas_x[-1] + maior + margem * 2)
    largura_tabela = colunas_x[-1] + max(largura_texto(linha[3]) for linha, _ in tabela)
    x2 = min(imagem.shape[1] - 1, max(largura_tabela, largura_texto(rodape) + margem) + margem)
    y2 = min(imagem.shape[0] - 1, altura_linha * (len(tabela) + 1) + margem)

    # escurece só a região do painel (mistura num recorte, sem copiar o frame todo)
    regiao = imagem[: y2 + 1, : x2 + 1]
    regiao[:] = cv2.addWeighted(regiao, 1.0 - opacidade, np.zeros_like(regiao), opacidade, 0.0)

    for i, (celulas, cor) in enumerate(tabela):
        y = altura_linha * (i + 1)
        for coluna, (texto, x) in enumerate(zip(celulas, colunas_x)):
            if coluna:  # números alinhados à direita do cabeçalho
                x += largura_texto(tabela[0][0][coluna]) - largura_texto(texto)
            escrever(imagem, texto, (x, y), tamanho_fonte, cor, fundo=None)
    escrever(imagem, rodape, (margem, altura_linha * (len(tabela) + 1)), tamanho_fonte, verde, fundo=None)
