"""Gera as figuras da documentação (docs/img/) a partir da cena sintética.

Uso::

    python scripts/gerar_figuras.py            # todas as figuras + GIF
    python scripts/gerar_figuras.py --sem-gif  # mais rápido

Cada figura ilustra um requisito de Computação Gráfica de
``docs/computacao_grafica.md``. Todo o desenho usa o próprio código do FluxoVC
(``src/``); só o vídeo e as detecções são simulados (ver
``scripts/cena_sintetica.py``). Os resultados são determinísticos.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "scripts"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from cena_sintetica import (  # noqa: E402
    ALTURA,
    CALIBRACAO_MUNDO,
    COMPRIMENTO_CALIBRACAO_M,
    LARGURA,
    LARGURA_PISTA_M,
    Y_LINHA_M,
    CenaSintetica,
    DetectorCena,
    mundo_para_imagem,
    normalizar,
)
from src.acquisition.fontes import Frame  # noqa: E402
from src.config import DOCS_DIR, AppConfig  # noqa: E402
from src.geometry.coordenadas import (  # noqa: E402
    calcular_letterbox,
    caixas_modelo_para_original,
    original_para_exibicao,
)
from src.geometry.homografia import gerar_birdseye, matriz_birdseye, transformar_pontos  # noqa: E402
from src.geometry.suavizacao import SuavizadorEMA  # noqa: E402
from src.logs import configurar_logging  # noqa: E402
from src.models.analisador import AnalisadorFluxo, ResultadoFrame  # noqa: E402
from src.models.contagem import ContadorLinha  # noqa: E402
from src.models.rastreamento import HistoricoTrajetorias  # noqa: E402
from src.preprocessing.pipeline import preprocessar  # noqa: E402
from src.visualization.camadas import desenhar_calibracao, linha_piscando  # noqa: E402
from src.visualization.compositor import CompositorCamadas, painel_comparacao  # noqa: E402

logger = logging.getLogger("fluxovc.figuras")

PASTA_CG = DOCS_DIR / "img" / "cg"
SUPERFICIE, TEXTO, TEXTO_2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DPI = 110


# ---------------------------------------------------------------------------
# Configuração da cena
# ---------------------------------------------------------------------------
def config_cena() -> AppConfig:
    cfg = AppConfig()
    linha = mundo_para_imagem([[-1.0, Y_LINHA_M], [LARGURA_PISTA_M + 1.0, Y_LINHA_M]])
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = normalizar(linha)
    roi = mundo_para_imagem([[-0.6, -8.0], [LARGURA_PISTA_M + 0.6, -8.0],
                             [LARGURA_PISTA_M + 0.6, 60.0], [-0.6, 60.0]])
    cfg.geometria.roi = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in normalizar(roi)]
    cfg.geometria.homografia_origem = normalizar(mundo_para_imagem(CALIBRACAO_MUNDO))
    cfg.geometria.largura_pista_m = LARGURA_PISTA_M
    cfg.geometria.comprimento_pista_m = COMPRIMENTO_CALIBRACAO_M
    cfg.visualizacao.mostrar_heatmap = True
    cfg.visualizacao.opacidade_heatmap = 0.40
    return cfg


@dataclass
class Simulacao:
    """Roda a pipeline real (analisador + compositor) na cena sintética."""

    cena: CenaSintetica
    cfg: AppConfig
    ruido_px: float = 2.0
    finais: list[np.ndarray] = field(default_factory=list)
    resultados: list[ResultadoFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.letterbox = calcular_letterbox(LARGURA, ALTURA, self.cfg.preproc.tamanho_entrada,
                                            self.cfg.preproc.manter_proporcao)
        self.detector = DetectorCena(self.cena, self.letterbox, self.ruido_px)
        self.analisador = AnalisadorFluxo(self.cfg, self.detector, LARGURA, ALTURA)
        self.compositor = CompositorCamadas(self.cfg, LARGURA, ALTURA)

    def rodar(self, ate_frame: int | None = None, guardar: bool = True) -> Simulacao:
        fim = self.cena.total_frames if ate_frame is None else ate_frame
        for i in range(len(self.resultados), fim):
            t = i / self.cena.fps
            self.detector.tempo_s = t
            imagem = self.cena.frame(i)
            resultado = self.analisador.processar(Frame(i, t, imagem))
            final = self.compositor.compor(imagem, resultado, self.analisador.contador, self.analisador.historico)
            self.resultados.append(resultado)
            self.finais.append(final if guardar else np.empty(0))
        return self

    @property
    def contador(self) -> ContadorLinha:
        return self.analisador.contador

    @property
    def historico(self) -> HistoricoTrajetorias:
        return self.analisador.historico


# ---------------------------------------------------------------------------
# Utilidades de figura
# ---------------------------------------------------------------------------
def rgb(imagem: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "text.color": TEXTO, "axes.labelcolor": TEXTO_2, "axes.edgecolor": GRADE,
        "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "font.size": 10, "axes.grid": False, "legend.frameon": False,
    })


def painel_imagens(imagens: Sequence[np.ndarray], titulos: Sequence[str], colunas: int,
                   largura_pol: float = 13.0, legenda: str | None = None) -> plt.Figure:
    linhas = int(np.ceil(len(imagens) / colunas))
    proporcao = imagens[0].shape[0] / imagens[0].shape[1]
    altura = largura_pol / colunas * proporcao * linhas + 0.45 * linhas + (0.4 if legenda else 0)
    fig, eixos = plt.subplots(linhas, colunas, figsize=(largura_pol, altura), squeeze=False)
    for eixo in eixos.ravel():
        eixo.axis("off")
    for eixo, imagem, titulo in zip(eixos.ravel(), imagens, titulos):
        eixo.imshow(rgb(imagem))
        eixo.set_title(titulo, loc="left")
    if legenda:
        fig.text(0.01, 0.01, legenda, color=TEXTO_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03 if legenda else 0, 1, 1))
    return fig


def salvar(fig: plt.Figure, nome: str) -> Path:
    PASTA_CG.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_CG / nome
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)
    logger.info("Figura salva: %s", caminho.relative_to(RAIZ))
    return caminho


def desenhar_caixa_rotulada(imagem: np.ndarray, caixa: Sequence[float], texto: str, cor=(0, 0, 255),
                            escala: float = 0.9) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in caixa)
    esp = max(2, imagem.shape[1] // 400)
    cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, esp, cv2.LINE_AA)
    cv2.circle(imagem, (x1, y1), esp + 3, cor, -1, cv2.LINE_AA)
    cv2.putText(imagem, texto, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), esp + 3, cv2.LINE_AA)
    cv2.putText(imagem, texto, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), esp, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 1. Coordenadas: modelo → original → exibição
# ---------------------------------------------------------------------------
def figura_coordenadas(cena: CenaSintetica, cfg: AppConfig) -> None:
    indice = 50
    imagem = cena.frame(indice)
    cfg_sem_roi = copy.deepcopy(cfg)
    cfg_sem_roi.preproc.usar_roi = False
    preproc = preprocessar(imagem, cfg_sem_roi.preproc)
    detector = DetectorCena(cena, preproc.letterbox, ruido_px=0.0)
    detector.tempo_s = indice / cena.fps
    deteccao = max(detector.rastrear(preproc.imagem), key=lambda d: (d.caixa[2] - d.caixa[0]) * (d.caixa[3] - d.caixa[1]))

    no_modelo = preproc.imagem.copy()
    cm = deteccao.caixa
    desenhar_caixa_rotulada(no_modelo, cm, f"({cm[0]:.0f}, {cm[1]:.0f})", escala=0.9)

    co = caixas_modelo_para_original([cm], preproc.letterbox)[0]
    no_original = imagem.copy()
    desenhar_caixa_rotulada(no_original, co, f"({co[0]:.0f}, {co[1]:.0f})", escala=1.6)

    escala = cfg.visualizacao.largura_exibicao / LARGURA
    exibicao = cv2.resize(imagem, (round(LARGURA * escala), round(ALTURA * escala)), interpolation=cv2.INTER_AREA)
    ce = original_para_exibicao(co.reshape(2, 2), escala).ravel()
    desenhar_caixa_rotulada(exibicao, ce, f"({ce[0]:.0f}, {ce[1]:.0f})", escala=1.2)

    lb = preproc.letterbox
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.78, 1.78]})
    for eixo, img, titulo in zip(
        eixos, (no_modelo, no_original, exibicao),
        (f"1 · Espaço do modelo ({lb.tamanho}×{lb.tamanho})",
         f"2 · Frame original ({LARGURA}×{ALTURA})",
         f"3 · Exibição na interface ({exibicao.shape[1]}×{exibicao.shape[0]})"),
    ):
        eixo.imshow(rgb(img))
        eixo.set_title(titulo, loc="left")
        eixo.set_xticks([0, img.shape[1]])
        eixo.set_yticks([0, img.shape[0]])
        for lado in eixo.spines.values():
            lado.set_visible(False)
    fig.text(
        0.01, 0.02,
        f"Letterbox: escala {lb.escala_x:.4f}, faixas pad_x = {lb.pad_x}, pad_y = {lb.pad_y} px.   "
        f"Inverso: x = (x_m − {lb.pad_x}) / {lb.escala_x:.4f},  y = (y_m − {lb.pad_y}) / {lb.escala_y:.4f}.   "
        f"Exibição: × {escala:.2f}.",
        color=TEXTO_2, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "01_coordenadas.png")


# ---------------------------------------------------------------------------
# 2. Redimensionamento: esticar × letterbox
# ---------------------------------------------------------------------------
def figura_letterbox(cena: CenaSintetica, cfg: AppConfig) -> None:
    imagem = cena.frame(40)
    pre = copy.deepcopy(cfg.preproc)
    pre.usar_roi = False
    pre.manter_proporcao = False
    esticada = preprocessar(imagem, pre).imagem
    pre.manter_proporcao = True
    letterbox = preprocessar(imagem, pre).imagem
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.78, 1, 1]})
    for eixo, img, titulo in zip(eixos, (imagem, esticada, letterbox),
                                 (f"Original {LARGURA}×{ALTURA} (16:9)", "Esticada para 640×640 (distorce)",
                                  "Letterbox 640×640 (mantém a proporção)")):
        eixo.imshow(rgb(img))
        eixo.set_title(titulo, loc="left")
        eixo.axis("off")
    fig.tight_layout()
    salvar(fig, "02_letterbox.png")


# ---------------------------------------------------------------------------
# 3. Homografia: bird's-eye e velocidade
# ---------------------------------------------------------------------------
def figura_homografia(sim: Simulacao, cena: CenaSintetica) -> list[tuple[int, str, float, float | None]]:
    indice = 75
    imagem = cena.frame(indice)
    origem = np.array(mundo_para_imagem(CALIBRACAO_MUNDO))
    com_quadro = imagem.copy()
    desenhar_calibracao(com_quadro, origem, 1.0)

    px_m = 14
    matriz, tamanho = matriz_birdseye(origem, LARGURA_PISTA_M, COMPRIMENTO_CALIBRACAO_M, px_m)
    vista = gerar_birdseye(imagem, matriz, tamanho)
    vista = cv2.resize(vista, (tamanho[0] * 2, tamanho[1] * 2), interpolation=cv2.INTER_NEAREST)
    for v in sim.resultados[indice].veiculos:
        x, y = transformar_pontos(matriz, [v.ponto])[0] * 2
        if 0 <= x < vista.shape[1] and 0 <= y < vista.shape[0]:
            cv2.circle(vista, (int(x), int(y)), 9, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vista, (int(x), int(y)), 9, (255, 255, 255), 2, cv2.LINE_AA)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2.6, 1]})
    eixos[0].imshow(rgb(com_quadro))
    eixos[0].set_title("Câmera: 4 pontos de um trecho de 14 m × 30 m (magenta)", loc="left")
    eixos[1].imshow(rgb(vista))
    eixos[1].set_title("warpPerspective → vista de cima", loc="left")
    passo = 5 * px_m * 2
    eixos[1].set_yticks(range(0, vista.shape[0] + 1, passo), [f"{m} m" for m in range(0, 31, 5)][: vista.shape[0] // passo + 1])
    eixos[1].set_xticks([0, vista.shape[1]], ["0", "14 m"])
    for eixo in eixos:
        for lado in eixo.spines.values():
            lado.set_visible(False)
    eixos[0].axis("off")
    fig.tight_layout()
    salvar(fig, "03_homografia.png")

    tabela = []
    for evento in sim.contador.eventos:
        real = next(v.velocidade_kmh for v in cena.veiculos if v.id_ == evento.id_rastreio)
        tabela.append((evento.id_rastreio, evento.nome_classe, real, evento.velocidade_kmh))
    return tabela


# ---------------------------------------------------------------------------
# 4. Composição por camadas
# ---------------------------------------------------------------------------
def figura_camadas(cena: CenaSintetica, cfg: AppConfig) -> None:
    sim = Simulacao(cena, cfg).rodar(ate_frame=90, guardar=False)
    imagem = cena.frame(89)
    resultado = sim.resultados[89]
    etapas = [
        ("0 · Frame original", set()),
        ("1 · + mapa de calor", {"mostrar_heatmap"}),
        ("2 · + ROI", {"mostrar_roi"}),
        ("3 · + linha de contagem", {"mostrar_linha"}),
        ("4 · + caixas e rótulos", {"mostrar_caixas"}),
        ("5 · + trajetórias", {"mostrar_trajetorias"}),
        ("6 · + vista de cima", {"mostrar_birdseye"}),
        ("7 · + painel (HUD)", {"mostrar_hud"}),
    ]
    todas = ["mostrar_heatmap", "mostrar_roi", "mostrar_linha", "mostrar_caixas",
             "mostrar_trajetorias", "mostrar_birdseye", "mostrar_hud"]
    ligadas: set[str] = set()
    imagens, titulos = [], []
    for titulo, novas in etapas:
        ligadas |= novas
        for nome in todas:
            setattr(sim.compositor.cfg.visualizacao, nome, nome in ligadas)
        ultimo = titulo.startswith("7")  # só o HUD precisa da latência do frame processado
        imagens.append(sim.compositor.compor(imagem, resultado if ultimo else _sem_acumular(resultado),
                                             sim.contador, sim.historico))
        titulos.append(titulo)
    salvar(painel_imagens(imagens, titulos, colunas=4, largura_pol=16), "04_camadas.png")


def _sem_acumular(resultado: ResultadoFrame) -> ResultadoFrame:
    """Mesmo resultado, mas marcado como não processado (não soma de novo no mapa de calor)."""
    return ResultadoFrame(resultado.frame, resultado.timestamp_s, False, resultado.veiculos,
                          resultado.eventos, resultado.preproc, resultado.tempos_ms)


# ---------------------------------------------------------------------------
# 5. Mapa de calor
# ---------------------------------------------------------------------------
def figura_mapa_calor(sim: Simulacao, cena: CenaSintetica) -> None:
    imagem = cena.frame(cena.total_frames - 1)
    mapa = sim.compositor.mapa_calor
    norm = mapa.normalizado()
    imagens, titulos = [], []
    cinza = cv2.cvtColor(cv2.resize(norm, (LARGURA, ALTURA), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
    imagens.append(cinza)
    titulos.append("Ocupação acumulada (normalizada, √)")
    for colormap, opacidade in (("TURBO", 0.45), ("JET", 0.45), ("TURBO", 0.8)):
        saida = imagem.copy()
        mapa.sobrepor(saida, opacidade, colormap)
        imagens.append(saida)
        titulos.append(f"{colormap}, opacidade {opacidade:.2f}".replace(".", ","))
    salvar(painel_imagens(imagens, titulos, colunas=2, largura_pol=13), "05_mapa_calor.png")


# ---------------------------------------------------------------------------
# 6. Primitivas de desenho (zoom)
# ---------------------------------------------------------------------------
def figura_desenho(sim: Simulacao) -> None:
    vis = sim.cfg.visualizacao
    evento = next(e for e in sim.contador.eventos if e.frame + vis.duracao_piscar_frames < len(sim.finais))
    indice = next(
        f for f in range(evento.frame + 2, evento.frame + vis.duracao_piscar_frames)
        if linha_piscando(f, evento.frame, vis.duracao_piscar_frames)
    )
    final = sim.finais[indice]
    escala = final.shape[1] / LARGURA
    y_linha = int(mundo_para_imagem([[7.0, Y_LINHA_M]])[0, 1] * escala)
    y1, y2 = max(0, y_linha - int(150 * escala)), min(final.shape[0], y_linha + int(170 * escala))
    x1, x2 = int(470 * escala), int(1150 * escala)
    recorte = cv2.resize(final[y1:y2, x1:x2], None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    fig = painel_imagens(
        [recorte], [f"Zoom no frame {indice}: a linha fica verde (pisca) logo após o cruzamento do ID {evento.id_rastreio}"],
        colunas=1, largura_pol=11,
        legenda="Caixas (cv2.rectangle), rótulos (cv2.putText), trajetórias (cv2.polylines), "
                "ROI (cv2.fillPoly + addWeighted), linha com rótulos A/B.",
    )
    salvar(fig, "06_desenho.png")


# ---------------------------------------------------------------------------
# 7. Atualização em tempo real
# ---------------------------------------------------------------------------
def figura_tempo_real(sim: Simulacao, cena: CenaSintetica) -> None:
    instantes = [3.0, 5.0, 7.0, 10.0]
    fig = plt.figure(figsize=(15, 7.4))
    grade = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    for i, t in enumerate(instantes):
        eixo = fig.add_subplot(grade[0, i])
        eixo.imshow(rgb(sim.finais[int(t * cena.fps)]))
        eixo.set_title(f"t = {t:.0f} s".replace(".", ","), loc="left")
        eixo.axis("off")
    eixo = fig.add_subplot(grade[1, :])
    tempos = np.arange(cena.total_frames) / cena.fps
    for cor, (sentido, rotulo) in zip(SERIES, (("A→B", "sentido A→B"), ("B→A", "sentido B→A"))):
        cruzamentos = sorted(e.timestamp_s for e in sim.contador.eventos if e.sentido == sentido)
        acumulado = np.searchsorted(cruzamentos, tempos, side="right")
        eixo.step(tempos, acumulado, where="post", color=cor, linewidth=2, label=rotulo)
    for t in instantes:
        eixo.axvline(t, color=GRADE, linewidth=1, zorder=0)
    eixo.set_xlabel("tempo do vídeo (s)")
    eixo.set_ylabel("veículos contados (acumulado)")
    eixo.set_title("A cada frame: detecção → contagem → desenho → tela (o gráfico da interface é atualizado junto)", loc="left")
    eixo.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    eixo.grid(axis="y", color=GRADE, linewidth=0.8)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    eixo.legend(loc="upper left")
    eixo.set_xlim(0, tempos[-1] + 0.8)
    fig.tight_layout()
    salvar(fig, "07_tempo_real.png")


# ---------------------------------------------------------------------------
# 8. Suavização EMA
# ---------------------------------------------------------------------------
def figura_ema(cena: CenaSintetica, cfg: AppConfig) -> None:
    lb = calcular_letterbox(LARGURA, ALTURA, cfg.preproc.tamanho_entrada)
    detector = DetectorCena(cena, lb, ruido_px=4.0, semente=11)
    alvo = 4  # carro vindo em direção à câmera
    brutos = []
    for i in range(cena.total_frames):
        detector.tempo_s = i / cena.fps
        for d in detector.rastrear(np.empty(0)):
            if d.id_rastreio == alvo:
                c = caixas_modelo_para_original([d.caixa], lb)[0]
                brutos.append((i, (c[0] + c[2]) / 2, c[3]))
    b = np.array(brutos)
    alphas = (0.8, 0.4, 0.15)
    suavizados = {}
    for alpha in alphas:
        suav = SuavizadorEMA(alpha)
        suavizados[alpha] = np.array([suav.atualizar(alvo, (x, y)) for _, x, y in brutos])
    y_linha = float(mundo_para_imagem([[7.0, Y_LINHA_M]])[0, 1])

    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(14, 5.0))
    esq.plot(b[:, 0], b[:, 1], color="#9a9994", linewidth=1, marker="o", markersize=3, label="ponto bruto")
    dir_.plot(b[:, 0], b[:, 2], color="#9a9994", linewidth=1, marker="o", markersize=3, label="ponto bruto")
    for cor, alpha in zip(SERIES, alphas):
        rotulo = f"EMA α = {alpha}".replace(".", ",")
        esq.plot(b[:, 0], suavizados[alpha][:, 0], color=cor, linewidth=2, label=rotulo)
        dir_.plot(b[:, 0], suavizados[alpha][:, 1], color=cor, linewidth=2, label=rotulo)
    meio = b[np.argmin(np.abs(b[:, 2] - y_linha)), 0]
    dir_.set_xlim(meio - 8, meio + 8)
    janela = (b[:, 0] >= meio - 8) & (b[:, 0] <= meio + 8)
    dir_.set_ylim(b[janela, 2].min() - 8, b[janela, 2].max() + 8)
    dir_.invert_yaxis()
    dir_.axhline(y_linha, color=TEXTO_2, linewidth=1, linestyle="--")
    dir_.annotate("linha de contagem", (meio - 8, y_linha), xytext=(4, -4), textcoords="offset points",
                  color=TEXTO_2, fontsize=9, va="bottom")
    janela_x = (b[:, 0] >= 30) & (b[:, 0] <= 70)
    esq.set_xlim(30, 70)
    esq.set_ylim(min(b[janela_x, 1].min(), suavizados[0.15][janela_x, 0].min()) - 4,
                 max(b[janela_x, 1].max(), suavizados[0.15][janela_x, 0].max()) + 4)
    esq.set_title("Tremor: x do ponto (frames 30–70)", loc="left")
    dir_.set_title("Atraso: y perto da linha (α pequeno cruza depois)", loc="left")
    esq.set_ylabel("x na imagem (px)")
    dir_.set_ylabel("y na imagem (px)")
    for eixo in (esq, dir_):
        eixo.set_xlabel("frame")
        eixo.grid(color=GRADE, linewidth=0.8)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
    esq.legend(loc="lower left")
    fig.suptitle(f"Suavização exponencial do ponto de referência do ID {alvo} (ruído de detecção simulado: σ = 4 px)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "08_ema.png")


# ---------------------------------------------------------------------------
# 9. ROI
# ---------------------------------------------------------------------------
def figura_roi(cena: CenaSintetica, cfg: AppConfig) -> None:
    indice = 45
    imagem = cena.frame(indice)
    preproc = preprocessar(imagem, cfg.preproc, cfg.geometria.roi)
    mascara = np.zeros((preproc.imagem.shape[0], preproc.imagem.shape[1], 3), np.uint8)
    cv2.fillPoly(mascara, [preproc.roi_modelo], (255, 255, 255))

    resultados = []
    for usar_roi in (False, True):
        c = copy.deepcopy(cfg)
        c.preproc.usar_roi = usar_roi
        c.visualizacao.mostrar_heatmap = c.visualizacao.mostrar_birdseye = c.visualizacao.mostrar_hud = False
        resultados.append(Simulacao(cena, c).rodar(ate_frame=indice + 1).finais[indice])
    fig = painel_imagens(
        [resultados[0], mascara, preproc.imagem, resultados[1]],
        ["Sem ROI: o carro estacionado na calçada é detectado", "Máscara do polígono (espaço do modelo)",
         "Entrada do modelo: fora da ROI vira cinza", "Com ROI: só veículos na pista"],
        colunas=2, largura_pol=13,
    )
    salvar(fig, "09_roi.png")


# ---------------------------------------------------------------------------
# 10. Comparação original × pré-processado × final (cena ao anoitecer)
# ---------------------------------------------------------------------------
def figura_comparacao(cfg: AppConfig) -> None:
    noite = CenaSintetica(noite=True)
    c = copy.deepcopy(cfg)
    c.preproc.usar_clahe = True
    c.preproc.clahe_clip = 3.0
    c.preproc.usar_reducao_ruido = True
    c.preproc.kernel_ruido = 7
    indice = 75
    sim = Simulacao(noite, c).rodar(ate_frame=indice + 1, guardar=False)
    imagem = noite.frame(indice)
    final = sim.compositor.compor(imagem, sim.resultados[indice], sim.contador, sim.historico)
    painel = painel_comparacao(imagem, sim.resultados[indice].preproc.imagem, final, altura=360)
    PASTA_CG.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PASTA_CG / "10_comparacao.png"), painel)
    logger.info("Figura salva: docs/img/cg/10_comparacao.png")

    # etapas do pré-processamento, com zoom para ver o ruído
    etapas, titulos = [], []
    cfg_etapa = copy.deepcopy(c.preproc)
    for titulo, clahe, ruido, roi in (("Original (letterbox)", False, False, False), ("+ CLAHE", True, False, False),
                                      ("+ filtro bilateral", True, True, False), ("+ máscara de ROI", True, True, True)):
        cfg_etapa.usar_clahe, cfg_etapa.usar_reducao_ruido, cfg_etapa.usar_roi = clahe, ruido, roi
        saida = preprocessar(imagem, cfg_etapa, c.geometria.roi).imagem
        etapas.append(cv2.resize(saida[180:460, 120:520], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
        titulos.append(titulo)
    salvar(painel_imagens(etapas, titulos, colunas=4, largura_pol=16,
                          legenda="Recorte ampliado da entrada do modelo (640×640) na cena ao anoitecer."),
           "11_preprocessamento.png")


# ---------------------------------------------------------------------------
# GIF de demonstração
# ---------------------------------------------------------------------------
def gerar_gif(sim: Simulacao, caminho: Path, largura: int = 560, passo: int = 2) -> None:
    from PIL import Image

    quadros = []
    for final in sim.finais[::passo]:
        pequeno = cv2.resize(final, (largura, round(final.shape[0] * largura / final.shape[1])), interpolation=cv2.INTER_AREA)
        quadros.append(Image.fromarray(rgb(pequeno)).quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    caminho.parent.mkdir(parents=True, exist_ok=True)
    duracao_ms = int(1000 * passo / sim.cena.fps)
    quadros[0].save(caminho, save_all=True, append_images=quadros[1:], duration=duracao_ms, loop=0, optimize=True)
    logger.info("GIF salvo: %s (%.1f MB)", caminho.relative_to(RAIZ), caminho.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    configurar_logging()
    parser = argparse.ArgumentParser(description="Gera as figuras da documentação.")
    parser.add_argument("--sem-gif", action="store_true")
    args = parser.parse_args(argv)
    estilo()

    cena = CenaSintetica()
    cfg = config_cena()
    sim = Simulacao(cena, copy.deepcopy(cfg)).rodar()
    logger.info("Simulação: %d veículos contados (%s).", sim.contador.total, sim.contador.por_classe())

    figura_coordenadas(cena, cfg)
    figura_letterbox(cena, cfg)
    velocidades = figura_homografia(sim, cena)
    figura_camadas(cena, copy.deepcopy(cfg))
    figura_mapa_calor(sim, cena)
    figura_desenho(sim)
    figura_tempo_real(sim, cena)
    figura_ema(cena, cfg)
    figura_roi(cena, cfg)
    figura_comparacao(cfg)
    if not args.sem_gif:
        gerar_gif(sim, DOCS_DIR / "img" / "demo.gif")

    logger.info("Velocidade real × estimada (km/h):")
    for id_, classe, real, estimada in velocidades:
        logger.info("  ID %d %-9s real %5.1f  estimada %s", id_, classe, real,
                    "—" if estimada is None else f"{estimada:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
