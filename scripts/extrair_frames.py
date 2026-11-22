"""Extrai frames igualmente espaçados de um vídeo para anotação (CVAT / Label Studio).

Uso::

    python scripts/extrair_frames.py data_samples/avenida.mp4 --quantidade 30

Os JPEGs vão para ``data_samples/ground_truth/para_anotar/``. Depois de anotar,
exporte no formato YOLO **com as imagens** para ``ground_truth/deteccao/`` (ver
docs/metricas.md). Não coloque os frames sem rótulo dentro de ``deteccao/``:
eles seriam lidos como "frames sem veículos".
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.acquisition.erros import ErroEntrada  # noqa: E402
from src.acquisition.fontes import extrair_frames  # noqa: E402
from src.config import GROUND_TRUTH_DIR  # noqa: E402
from src.logs import configurar_logging  # noqa: E402

logger = logging.getLogger("fluxovc.extrair_frames")


def main(argv: Sequence[str] | None = None) -> int:
    configurar_logging()
    parser = argparse.ArgumentParser(description="Extrai frames para anotação.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--quantidade", type=int, default=30)
    parser.add_argument("--saida", type=Path, default=GROUND_TRUTH_DIR / "para_anotar")
    args = parser.parse_args(argv)
    try:
        salvos = extrair_frames(args.video, args.quantidade, args.saida)
    except (ErroEntrada, ValueError) as erro:
        logger.error("%s", erro)
        return 1
    logger.info("%d frames salvos em %s", len(salvos), args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
