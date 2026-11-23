"""Cena sintética de uma avenida vista por uma câmera elevada (para as figuras da documentação).

O "mundo" é o plano da rua em metros (x = atravessando a pista, y = ao longo
dela). Uma homografia conhecida leva o mundo para a imagem da câmera, então a
cena tem **perspectiva real** e a velocidade de cada veículo é conhecida —
o que permite conferir a homografia e o cálculo de velocidade do FluxoVC.

Os veículos são caixas 3D simples. O :class:`DetectorCena` faz o papel do
YOLO + ByteTrack: devolve a caixa de cada veículo visível (com um pequeno
ruído, como um detector real) no espaço do modelo, com ID fixo por veículo.
Nenhum peso de modelo é necessário.

As figuras **não** mostram o desempenho do YOLO: para isso, veja
``docs/resultados.md`` (gerado com vídeo real por ``scripts/evaluate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.geometry.coordenadas import ParametrosLetterbox, original_para_modelo
from src.models.tipos import DeteccaoBruta

LARGURA, ALTURA = 1280, 720
LARGURA_PISTA_M = 14.0              # 4 faixas de 3,5 m
FAIXAS_X = (1.75, 5.25, 8.75, 12.25)
# Retângulo do mundo (x 0–14 m, y 0–60 m) → quadrilátero na imagem
MUNDO_REF = np.float32([[0, 0], [LARGURA_PISTA_M, 0], [LARGURA_PISTA_M, 60], [0, 60]])
IMAGEM_REF = np.float32([[500, 215], [780, 215], [1230, 720], [50, 720]])
H_MUNDO_IMAGEM = cv2.getPerspectiveTransform(MUNDO_REF, IMAGEM_REF)

# Calibração "medida em campo": trecho de 14 m × 30 m (y de 20 a 50 m)
CALIBRACAO_MUNDO = [(0.0, 20.0), (LARGURA_PISTA_M, 20.0), (LARGURA_PISTA_M, 50.0), (0.0, 50.0)]
COMPRIMENTO_CALIBRACAO_M = 30.0
Y_LINHA_M = 38.0                    # linha de contagem atravessando a pista
Y_MAXIMO_M = 64.0                   # além disso o chão já está abaixo da borda da imagem


def mundo_para_imagem(pontos_m: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(pontos_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H_MUNDO_IMAGEM).reshape(-1, 2)


def normalizar(pontos_px: np.ndarray) -> list[tuple[float, float]]:
    return [(round(float(x) / LARGURA, 4), round(float(y) / ALTURA, 4)) for x, y in pontos_px]


def pixels_por_metro(x_m: float, y_m: float) -> float:
    """Tamanho aproximado, em pixels, de 1 m na horizontal naquele ponto da rua."""
    a, b = mundo_para_imagem([[x_m, y_m], [x_m + 1.0, y_m]])
    return float(np.linalg.norm(b - a))


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Veiculo:
    id_: int
    classe: int                     # id COCO (2 carro, 3 moto, 5 ônibus, 7 caminhão)
    faixa: int                      # 0–3 (0 e 1 vêm na direção da câmera)
    velocidade_kmh: float
    inicio_s: float
    cor: tuple[int, int, int]       # BGR
    largura_m: float
    comprimento_m: float
    altura_m: float
    parado_em_y: float | None = None  # veículo estacionado (fora da pista)
    x_fixo: float | None = None

    @property
    def vem(self) -> bool:
        return self.faixa <= 1

    def posicao(self, t: float) -> tuple[float, float] | None:
        """Centro da pegada no chão (m) no instante ``t``; ``None`` se ainda não entrou."""
        if self.parado_em_y is not None:
            return (self.x_fixo or 0.0, self.parado_em_y)
        if t < self.inicio_s:
            return None
        andou = (t - self.inicio_s) * self.velocidade_kmh / 3.6
        y = -30.0 + andou if self.vem else 75.0 - andou
        if not -30.0 <= y <= 75.0:
            return None
        return (FAIXAS_X[self.faixa], y)

    def cruza_linha_em(self) -> float | None:
        """Instante em que o ponto de referência cruza a linha de contagem.

        O ponto de referência do FluxoVC é o centro da **base** da caixa, que fica
        na borda do veículo mais próxima da câmera (frente de quem vem, traseira de
        quem vai) — não no centro do veículo.
        """
        if self.parado_em_y is None:
            meio_c = self.comprimento_m / 2
            distancia = (Y_LINHA_M - meio_c + 30.0) if self.vem else (75.0 - Y_LINHA_M + meio_c)
            return self.inicio_s + distancia / (self.velocidade_kmh / 3.6)
        return None


DIMENSOES = {2: (1.8, 4.5, 1.5), 3: (0.8, 2.0, 1.4), 5: (2.6, 12.0, 3.2), 7: (2.5, 8.0, 3.0)}


def _v(id_, classe, faixa, kmh, inicio, cor, **kw) -> Veiculo:
    largura, comprimento, altura = DIMENSOES[classe]
    return Veiculo(id_, classe, faixa, kmh, inicio, cor, largura, comprimento, altura, **kw)


VEICULOS_PADRAO: tuple[Veiculo, ...] = (
    _v(1, 2, 2, 50, 0.0, (40, 40, 200)),
    _v(2, 2, 1, 45, 0.3, (200, 200, 200)),
    _v(3, 5, 3, 35, 0.8, (60, 150, 60)),
    _v(4, 2, 0, 55, 1.6, (180, 90, 30)),
    _v(5, 3, 2, 40, 2.6, (30, 30, 30)),
    _v(6, 7, 0, 40, 3.4, (40, 120, 230)),
    _v(7, 2, 3, 60, 4.2, (230, 230, 230)),
    _v(8, 2, 1, 50, 5.0, (90, 40, 120)),
    _v(9, 2, 0, 0, 0.0, (110, 110, 150), parado_em_y=30.0, x_fixo=-2.2),  # estacionado
)


# ---------------------------------------------------------------------------
@dataclass
class CenaSintetica:
    fps: float = 15.0
    duracao_s: float = 11.0
    veiculos: tuple[Veiculo, ...] = VEICULOS_PADRAO
    noite: bool = False
    _fundo: np.ndarray | None = field(default=None, repr=False)

    @property
    def total_frames(self) -> int:
        return int(round(self.fps * self.duracao_s))

    def fundo(self) -> np.ndarray:
        if self._fundo is None:
            self._fundo = _desenhar_fundo()
        return self._fundo

    def caixas(self, t: float) -> list[tuple[Veiculo, tuple[float, float, float, float]]]:
        """Caixa (x1, y1, x2, y2) na imagem de cada veículo visível, do mais longe ao mais perto."""
        saida = []
        for v in self.veiculos:
            pos = v.posicao(t)
            if pos is None:
                continue
            pontos = _pontos_caixa_3d(v, pos)
            x1, y1 = pontos.min(axis=0)
            x2, y2 = pontos.max(axis=0)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(LARGURA, x2), min(ALTURA, y2)
            if x2 - x1 >= 6 and y2 - y1 >= 6 and y1 < ALTURA - 4:
                saida.append((v, (float(x1), float(y1), float(x2), float(y2))))
        return sorted(saida, key=lambda item: item[1][3])

    def frame(self, indice: int) -> np.ndarray:
        t = indice / self.fps
        imagem = self.fundo().copy()
        for v, _ in self.caixas(t):
            _desenhar_veiculo(imagem, v, v.posicao(t))  # type: ignore[arg-type]
        if self.noite:
            imagem = _escurecer(imagem, indice)
        return imagem


def _pontos_caixa_3d(v: Veiculo, pos: tuple[float, float]) -> np.ndarray:
    x, y = pos
    meia_l, meio_c = v.largura_m / 2, v.comprimento_m / 2
    # a parte do veículo já "embaixo" da câmera (fora da imagem) é achatada em Y_MAXIMO_M,
    # antes da linha singular da homografia (y ≈ 78,7 m), onde a projeção se inverteria
    longe, perto = min(y - meio_c, Y_MAXIMO_M), min(y + meio_c, Y_MAXIMO_M)
    base = mundo_para_imagem(
        [[x - meia_l, longe], [x + meia_l, longe], [x + meia_l, perto], [x - meia_l, perto]]
    )
    escalas = np.array([pixels_por_metro(x, longe), pixels_por_metro(x, longe),
                        pixels_por_metro(x, perto), pixels_por_metro(x, perto)])
    topo = base - np.column_stack((np.zeros(4), escalas * v.altura_m))
    return np.vstack((base, topo))


def _desenhar_veiculo(imagem: np.ndarray, v: Veiculo, pos: tuple[float, float]) -> None:
    pontos = _pontos_caixa_3d(v, pos)
    base, topo = pontos[:4], pontos[4:]
    casco = cv2.convexHull(np.round(pontos).astype(np.int32))
    sombra = np.round(base + np.array([4.0, 2.0])).astype(np.int32)
    cv2.fillConvexPoly(imagem, sombra, (35, 35, 35), cv2.LINE_AA)
    cor = np.array(v.cor, dtype=np.float64)
    cv2.fillConvexPoly(imagem, casco, tuple(int(c) for c in cor * 0.7), cv2.LINE_AA)
    cv2.fillConvexPoly(imagem, np.round(topo).astype(np.int32), tuple(int(min(255, c * 1.05 + 10)) for c in cor), cv2.LINE_AA)
    # vidro (para-brisa ou vidro traseiro) no lado voltado para a câmera
    perto, longe = topo[2:4], topo[1::-1]
    vidro = np.vstack((perto, perto[::-1] + 0.3 * (longe[::-1] - perto[::-1])))
    cv2.fillConvexPoly(imagem, np.round(vidro).astype(np.int32), (70, 60, 50), cv2.LINE_AA)
    cv2.polylines(imagem, [casco], True, (20, 20, 20), 1, cv2.LINE_AA)


def _desenhar_fundo() -> np.ndarray:
    """Céu, prédios ao fundo, calçadas, gramado e a avenida com faixas."""
    escala = 12  # px por metro na textura vista de cima
    x0, x1, y0, y1 = -60.0, 74.0, -400.0, 62.0
    larg_tex, alt_tex = int((x1 - x0) * escala), int((y1 - y0) * escala)
    textura = np.full((alt_tex, larg_tex, 3), (70, 120, 80), dtype=np.uint8)  # gramado

    def px(xm: float, ym: float) -> tuple[int, int]:
        return int(round((xm - x0) * escala)), int(round((ym - y0) * escala))

    cv2.rectangle(textura, px(-4.5, y0), px(LARGURA_PISTA_M + 4.5, y1), (175, 175, 170), -1)  # calçadas
    cv2.rectangle(textura, px(0, y0), px(LARGURA_PISTA_M, y1), (85, 85, 88), -1)              # asfalto
    ruido = np.random.default_rng(7).normal(0, 6, textura.shape[:2])
    textura = np.clip(textura.astype(np.float64) + ruido[..., None], 0, 255).astype(np.uint8)
    cv2.line(textura, px(7.0 - 0.1, y0), px(7.0 - 0.1, y1), (0, 200, 230), 3)                # divisória
    cv2.line(textura, px(7.0 + 0.1, y0), px(7.0 + 0.1, y1), (0, 200, 230), 3)
    for xm in (3.5, 10.5):                                                                   # tracejadas
        y = y0
        while y < y1:
            cv2.rectangle(textura, px(xm - 0.08, y), px(xm + 0.08, y + 3.0), (235, 235, 235), -1)
            y += 9.0
    for xm in (0.15, LARGURA_PISTA_M - 0.15):                                                # bordas
        cv2.line(textura, px(xm, y0), px(xm, y1), (235, 235, 235), 2)

    # textura (px) → mundo (m) → imagem
    t_mundo = np.array([[1 / escala, 0, x0], [0, 1 / escala, y0], [0, 0, 1]])
    chao = cv2.warpPerspective(textura, H_MUNDO_IMAGEM @ t_mundo, (LARGURA, ALTURA), flags=cv2.INTER_AREA)
    mascara = cv2.warpPerspective(np.full(textura.shape[:2], 255, np.uint8), H_MUNDO_IMAGEM @ t_mundo,
                                  (LARGURA, ALTURA), flags=cv2.INTER_NEAREST)

    imagem = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
    horizonte = int(mundo_para_imagem([[7.0, y0]])[0, 1]) + 2
    for y in range(ALTURA):  # céu em degradê
        f = min(1.0, y / max(1, horizonte))
        imagem[y] = (235 - 60 * f, 205 - 30 * f, 170 - 10 * f)
    rng = np.random.default_rng(3)
    x = 0
    while x < LARGURA:  # prédios no horizonte
        largura_p = int(rng.integers(40, 110))
        altura_p = int(rng.integers(30, 130))
        tom = int(rng.integers(110, 170))
        cv2.rectangle(imagem, (x, horizonte - altura_p), (x + largura_p - 4, horizonte), (tom, tom - 5, tom - 15), -1)
        x += largura_p
    imagem[mascara > 0] = chao[mascara > 0]
    return imagem


def _escurecer(imagem: np.ndarray, indice: int) -> np.ndarray:
    """Versão "anoitecer": pouco contraste e ruído de sensor (para mostrar CLAHE e filtro)."""
    escura = (imagem.astype(np.float64) * 0.28 + 8)
    ruido = np.random.default_rng(indice).normal(0, 7, imagem.shape)
    return np.clip(escura + ruido, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
class DetectorCena:
    """Imita YOLO + ByteTrack na cena sintética (cumpre o contrato ``Detector``).

    Antes de cada frame, defina :attr:`tempo_s`. O detector devolve as caixas no
    espaço do modelo (aplica o mesmo letterbox da pipeline) com ruído de ±``ruido_px``.
    """

    def __init__(self, cena: CenaSintetica, letterbox: ParametrosLetterbox, ruido_px: float = 2.0, semente: int = 0) -> None:
        self.cena = cena
        self.letterbox = letterbox
        self.ruido_px = ruido_px
        self.rng = np.random.default_rng(semente)
        self.tempo_s = 0.0
        self.confianca = {v.id_: float(c) for v, c in zip(cena.veiculos, np.linspace(0.93, 0.62, len(cena.veiculos)))}

    def _deteccoes(self, com_id: bool) -> list[DeteccaoBruta]:
        saida = []
        for v, caixa in self.cena.caixas(self.tempo_s):
            ruido = self.rng.normal(0, self.ruido_px, 4)
            cantos = np.array(caixa).reshape(2, 2) + ruido.reshape(2, 2)
            m = original_para_modelo(cantos, self.letterbox).ravel()
            saida.append(DeteccaoBruta((m[0], m[1], m[2], m[3]), v.classe, self.confianca[v.id_],
                                       v.id_ if com_id else None))
        return saida

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        return self._deteccoes(True)

    def detectar(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        return self._deteccoes(False)

    def reiniciar(self) -> None:
        pass
