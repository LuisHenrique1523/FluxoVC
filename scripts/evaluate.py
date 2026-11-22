"""Avaliação em lote do FluxoVC → ``docs/resultados.md``.

Para cada vídeo de ``data_samples/`` (ou os informados em ``--videos``), roda
a pipeline com cada modelo e cada valor de frame skip, e compara com:

* ``data_samples/ground_truth/<video>.json`` — configuração do vídeo (linha,
  ROI, calibração), salva pela interface em "Salvar configuração";
* ``data_samples/ground_truth/<video>_contagem.csv`` — contagem manual;
* ``data_samples/ground_truth/deteccao/`` — frames anotados no formato YOLO.

Uso::

    python scripts/evaluate.py                         # completo (n e s, skip 0 e 2)
    python scripts/evaluate.py --max-frames 300        # rápido, só o início de cada vídeo
    python scripts/evaluate.py --modelos yolo11n.pt --frame-skip 0
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:  # permite rodar "python scripts/evaluate.py" da raiz
    sys.path.insert(0, str(RAIZ))

from src.acquisition.erros import ErroEntrada  # noqa: E402
from src.acquisition.fontes import listar_videos_demo  # noqa: E402
from src.config import DOCS_DIR, GROUND_TRUTH_DIR, AppConfig, carregar_config  # noqa: E402
from src.geometry.erros import ErroGeometria  # noqa: E402
from src.logs import configurar_logging  # noqa: E402
from src.metrics.anotacoes import (  # noqa: E402
    carregar_contagem_manual,
    ler_nomes_classes,
    listar_frames_anotados,
)
from src.metrics.avaliacao import ResultadoVideo, avaliar_deteccao, avaliar_video  # noqa: E402
from src.metrics.relatorio import DadosRelatorio, descrever_ambiente, gerar_relatorio  # noqa: E402
from src.models.tipos import Detector, ErroModelo  # noqa: E402

logger = logging.getLogger("fluxovc.evaluate")

FabricaDetector = Callable[[AppConfig], Detector]


def fabrica_yolo(cfg: AppConfig) -> Detector:
    from src.models.detector_yolo import DetectorYOLO

    return DetectorYOLO(cfg.modelo, cfg.preproc.tamanho_entrada)


def ler_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação em lote do FluxoVC.")
    parser.add_argument("--videos", nargs="*", type=Path, help="vídeos (padrão: todos de data_samples/)")
    parser.add_argument("--modelos", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    parser.add_argument("--frame-skip", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--max-frames", type=int, default=None, help="limita os frames por vídeo")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_DIR)
    parser.add_argument("--deteccao", type=Path, default=None, help="pasta dos frames anotados")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para casar detecções")
    parser.add_argument("--dispositivo", default="auto", help="auto, cpu ou 0 (GPU)")
    parser.add_argument("--saida", type=Path, default=DOCS_DIR / "resultados.md")
    return parser.parse_args(argv)


def config_do_video(video: Path, pasta_gt: Path, avisos: list[str]) -> AppConfig:
    caminho = pasta_gt / f"{video.stem}.json"
    if caminho.exists():
        return carregar_config(caminho)
    avisos.append(
        f"`{video.name}`: sem `{caminho.name}` — usada a configuração padrão "
        "(linha horizontal em y = 0,60). Salve a configuração pela interface."
    )
    return AppConfig()


def main(argv: Sequence[str] | None = None, fabrica: FabricaDetector = fabrica_yolo) -> int:
    configurar_logging()
    args = ler_argumentos(argv)
    pasta_deteccao = args.deteccao or args.ground_truth / "deteccao"
    videos = list(args.videos) if args.videos is not None else listar_videos_demo()

    detectores: dict[str, Detector] = {}

    def detector_para(cfg: AppConfig) -> Detector:
        if cfg.modelo.pesos not in detectores:
            detectores[cfg.modelo.pesos] = fabrica(cfg)
        detector = detectores[cfg.modelo.pesos]
        if hasattr(detector, "cfg"):
            detector.cfg = cfg.modelo  # type: ignore[attr-defined]
        return detector

    resultados: list[ResultadoVideo] = []
    manuais: dict[str, dict] = {}
    avisos: list[str] = []
    try:
        for video in videos:
            base = config_do_video(video, args.ground_truth, avisos)
            csv = args.ground_truth / f"{video.stem}_contagem.csv"
            if csv.exists():
                try:
                    manuais[video.name] = carregar_contagem_manual(csv)
                except ValueError as erro:
                    avisos.append(f"Contagem manual ignorada: {erro}")
            for modelo in args.modelos:
                for skip in args.frame_skip:
                    cfg = copy.deepcopy(base)
                    cfg.modelo.pesos, cfg.modelo.dispositivo = modelo, args.dispositivo
                    cfg.preproc.frame_skip = skip
                    logger.info("Avaliando %s com %s, frame skip %d…", video.name, modelo, skip)
                    try:
                        resultados.append(avaliar_video(video, cfg, detector_para(cfg), args.max_frames))
                    except (ErroEntrada, ErroGeometria) as erro:
                        avisos.append(f"`{video.name}` ({modelo}, skip {skip}) não avaliado: {erro}")

        deteccao = {}
        frames = listar_frames_anotados(pasta_deteccao) if pasta_deteccao.is_dir() else []
        if frames:
            try:
                nomes = ler_nomes_classes(pasta_deteccao)
                for modelo in args.modelos:
                    cfg = AppConfig()
                    cfg.modelo.pesos, cfg.modelo.dispositivo = modelo, args.dispositivo
                    logger.info("Avaliando detecção em %d frames com %s…", len(frames), modelo)
                    deteccao[modelo] = avaliar_deteccao(frames, nomes, cfg, detector_para(cfg), args.iou)
            except (FileNotFoundError, ValueError) as erro:
                avisos.append(f"Avaliação de detecção ignorada: {erro}")
    except ErroModelo as erro:
        logger.error("%s", erro)
        return 2

    if not resultados and not deteccao:
        logger.error(
            "Nada para avaliar: coloque vídeos em data_samples/ (ou use --videos) "
            "e/ou frames anotados em %s.", pasta_deteccao,
        )
        return 1

    dados = DadosRelatorio(resultados, manuais, deteccao, len(frames), descrever_ambiente(), avisos)
    caminho = gerar_relatorio(dados, args.saida)
    logger.info("Pronto: %s", caminho)
    return 0


if __name__ == "__main__":
    sys.exit(main())
