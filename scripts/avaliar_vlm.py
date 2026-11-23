"""Avalia o VLM nos 10 casos de teste → ``docs/vlm_resultados.md``.

Para cada frame, a verdade vem da pipeline (detecções do YOLO no frame
inteiro) e cada uma das 10 perguntas de ``src/vlm/casos.py`` é feita ao VLM,
**sempre com o frame anonimizado**. As respostas são classificadas
automaticamente (correta / incompleta / incorreta / alucinada) e a latência é
medida. A coluna "revisão" fica para a equipe conferir a classificação.

Uso::

    python scripts/avaliar_vlm.py --video data_samples/avenida.mp4 --frames 3
    python scripts/avaliar_vlm.py --imagens data_samples/ground_truth/para_anotar --frames 5
    python scripts/avaliar_vlm.py --cena-sintetica          # sem vídeo: verdade da cena sintética
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
for _caminho in (RAIZ, RAIZ / "scripts"):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

from src.acquisition.erros import ErroEntrada  # noqa: E402
from src.acquisition.fontes import abrir_arquivo, carregar_imagem  # noqa: E402
from src.config import DOCS_DIR, AppConfig  # noqa: E402
from src.logs import configurar_logging  # noqa: E402
from src.models.analisador import analisar_imagem  # noqa: E402
from src.models.tipos import ErroModelo  # noqa: E402
from src.vlm.backends import VLM, ConfigVLM, ErroVLM, criar_vlm  # noqa: E402
from src.vlm.casos import CASOS, CATEGORIAS, Verdade, classificar  # noqa: E402
from src.vlm.servico import FrameAnonimizado, pergunta_para, preparar_frame  # noqa: E402

logger = logging.getLogger("fluxovc.avaliar_vlm")

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Amostra:
    nome: str
    frame: FrameAnonimizado
    verdade: Verdade


@dataclass(frozen=True)
class Linha:
    amostra: str
    caso: str
    pergunta: str
    esperado: str
    resposta: str
    categoria: str
    latencia_s: float


# ---------------------------------------------------------------------------
# Amostras (frame anonimizado + verdade da pipeline)
# ---------------------------------------------------------------------------
def frames_do_video(caminho: Path, quantidade: int) -> list[tuple[str, np.ndarray]]:
    with abrir_arquivo(caminho) as fonte:
        total = fonte.info.total_frames or 0
        alvos = set(np.linspace(0, max(0, total - 1), quantidade).round().astype(int).tolist())
        frames = [(f"{caminho.stem} f{f.indice}", f.imagem) for f in fonte if f.indice in alvos]
    return frames


def frames_de_imagens(pasta: Path, quantidade: int) -> list[tuple[str, np.ndarray]]:
    caminhos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM)
    if quantidade and len(caminhos) > quantidade:
        caminhos = [caminhos[i] for i in np.linspace(0, len(caminhos) - 1, quantidade).round().astype(int)]
    return [(p.name, carregar_imagem(p)) for p in caminhos]


def amostra_da_pipeline(nome: str, imagem: np.ndarray, detector: object) -> Amostra:
    cfg = AppConfig()
    cfg.preproc.usar_roi = False  # a pergunta é sobre a imagem inteira
    veiculos, _ = analisar_imagem(imagem, cfg, detector)  # type: ignore[arg-type]
    verdade = Verdade.de_deteccoes([(v.classe, v.caixa) for v in veiculos], imagem.shape[1])
    return Amostra(nome, preparar_frame(imagem, [v.caixa for v in veiculos], detector), verdade)


def amostras_cena_sintetica(quantidade: int) -> list[Amostra]:
    from cena_sintetica import LARGURA, CenaSintetica

    cena = CenaSintetica()
    amostras = []
    for indice in np.linspace(30, cena.total_frames - 30, quantidade).round().astype(int):
        t = indice / cena.fps
        caixas = [(v.classe, c) for v, c in cena.caixas(t)]
        imagem = cena.frame(int(indice))
        frame = preparar_frame(imagem, [c for _, c in caixas], detector=None)
        amostras.append(Amostra(f"cena sintética f{indice}", frame, Verdade.de_deteccoes(caixas, LARGURA)))
    return amostras


# ---------------------------------------------------------------------------
def avaliar(vlm: VLM, amostras: Sequence[Amostra]) -> list[Linha]:
    linhas = []
    for amostra in amostras:
        for caso in CASOS:
            pergunta = pergunta_para(vlm.backend, caso.pergunta_pt, caso.pergunta_en)
            resposta = vlm.responder(amostra.frame.imagem, pergunta)
            categoria = classificar(caso, resposta.texto, amostra.verdade)
            logger.info("%s %s → %r (%s, %.1f s)", amostra.nome, caso.id_, resposta.texto[:60], categoria,
                        resposta.latencia_s)
            linhas.append(Linha(amostra.nome, caso.id_, pergunta, caso.esperado(amostra.verdade),
                                resposta.texto, categoria, resposta.latencia_s))
    return linhas


def _decimal(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _celula(texto: str, limite: int = 140) -> str:
    texto = " ".join(str(texto).split()).replace("|", "\\|")
    return texto if len(texto) <= limite else texto[: limite - 1] + "…"


def gerar_relatorio(linhas: Sequence[Linha], vlm: VLM, origem: str, caminho: Path) -> Path:
    contagem = Counter(linha.categoria for linha in linhas)
    latencias = [linha.latencia_s for linha in linhas]
    n = len(linhas)
    partes = [
        "# Resultados do VLM",
        "",
        "> Gerado por `python scripts/avaliar_vlm.py`. **Não edite à mão**, exceto a coluna "
        "*revisão*. A classificação automática é uma primeira passada: confira cada linha.",
        "",
        f"- **Modelo:** `{vlm.nome}` (backend `{vlm.backend}`)",
        f"- **Frames:** {origem}",
        f"- **Data:** {datetime.now():%d/%m/%Y %H:%M}",
        "- **Verdade:** detecções da pipeline no frame inteiro (sem ROI).",
        "- **Privacidade:** todas as perguntas usaram o frame anonimizado (reduzido, pessoas e placas pixelizadas).",
        "",
        "## Resumo",
        "",
        "| Categoria | Respostas | % |",
        "|---|---|---|",
    ]
    for categoria in CATEGORIAS:
        partes.append(f"| {categoria} | {contagem.get(categoria, 0)} | {100 * contagem.get(categoria, 0) / n:.0f} % |")
    partes += [
        f"| **total** | **{n}** | |",
        "",
        f"**Latência por resposta:** média {_decimal(statistics.mean(latencias), 1)} s, "
        f"máxima {_decimal(max(latencias), 1)} s.",
        "",
        "## Acerto por caso",
        "",
        "| Caso | Pergunta | Corretas |",
        "|---|---|---|",
    ]
    for caso in CASOS:
        do_caso = [linha for linha in linhas if linha.caso == caso.id_]
        corretas = sum(linha.categoria == "correta" for linha in do_caso)
        partes.append(f"| {caso.id_} | {caso.pergunta_pt} | {corretas}/{len(do_caso)} |")
    partes += [
        "",
        "## Respostas",
        "",
        "| Frame | Caso | Esperado (pipeline) | Resposta do VLM | Automática | Revisão | Latência (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for linha in linhas:
        partes.append(
            f"| {_celula(linha.amostra)} | {linha.caso} | {_celula(linha.esperado)} | {_celula(linha.resposta)} "
            f"| {linha.categoria} | | {_decimal(linha.latencia_s, 1)} |"
        )
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(partes) + "\n", encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
def fabrica_detector() -> object:
    from src.models.detector_yolo import DetectorYOLO

    cfg = AppConfig()
    return DetectorYOLO(cfg.modelo, cfg.preproc.tamanho_entrada)


def main(
    argv: Sequence[str] | None = None,
    fabrica_vlm: Callable[[ConfigVLM], VLM] = criar_vlm,
    fabrica_det: Callable[[], object] = fabrica_detector,
) -> int:
    configurar_logging()
    parser = argparse.ArgumentParser(description="Avalia o VLM nos 10 casos de teste.")
    origem = parser.add_mutually_exclusive_group(required=True)
    origem.add_argument("--video", type=Path)
    origem.add_argument("--imagens", type=Path, help="pasta com .jpg/.png")
    origem.add_argument("--cena-sintetica", action="store_true")
    parser.add_argument("--frames", type=int, default=3, help="quantos frames avaliar")
    parser.add_argument("--backend", choices=("local", "claude"), default=None, help="sobrepõe o .env")
    parser.add_argument("--saida", type=Path, default=DOCS_DIR / "vlm_resultados.md")
    args = parser.parse_args(argv)

    cfg_vlm = ConfigVLM.do_ambiente()
    if args.backend:
        cfg_vlm = ConfigVLM(args.backend, cfg_vlm.modelo_local, cfg_vlm.modelo_claude, cfg_vlm.max_tokens)
    try:
        if args.cena_sintetica:
            amostras, descricao = amostras_cena_sintetica(args.frames), "cena sintética (verdade = geometria da cena)"
        else:
            detector = fabrica_det()
            brutos = frames_do_video(args.video, args.frames) if args.video else frames_de_imagens(args.imagens, args.frames)
            amostras = [amostra_da_pipeline(nome, img, detector) for nome, img in brutos]
            descricao = f"{len(amostras)} de `{(args.video or args.imagens).name}`"
        if not amostras:
            logger.error("Nenhum frame para avaliar.")
            return 1
        vlm = fabrica_vlm(cfg_vlm)
        linhas = avaliar(vlm, amostras)
    except (ErroEntrada, ErroModelo, ErroVLM) as erro:
        logger.error("%s", erro)
        return 2
    caminho = gerar_relatorio(linhas, vlm, descricao, args.saida)
    logger.info("Pronto: %s", caminho)
    return 0


if __name__ == "__main__":
    sys.exit(main())
