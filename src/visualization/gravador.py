"""Gravação do vídeo processado (para download na interface).

Tenta primeiro H.264 (``avc1``), que os navegadores reproduzem; se o OpenCV
da máquina não tiver esse codificador, usa MPEG-4 Part 2 (``mp4v``), que abre
em qualquer player de vídeo, mas não no navegador.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CODECS_PREFERIDOS: tuple[str, ...] = ("avc1", "mp4v")


class GravadorVideo:
    """Grava frames BGR num ``.mp4``. O tamanho é definido pelo primeiro frame."""

    def __init__(self, caminho: Path, fps: float, codecs: tuple[str, ...] = CODECS_PREFERIDOS) -> None:
        self.caminho = Path(caminho)
        self.fps = fps
        self.codecs = codecs
        self.codec: str | None = None
        self.frames_gravados = 0
        self._escritor: cv2.VideoWriter | None = None
        self._tamanho: tuple[int, int] | None = None

    def _abrir(self, largura: int, altura: int) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        for codec in self.codecs:
            escritor = cv2.VideoWriter(
                str(self.caminho), cv2.VideoWriter.fourcc(*codec), self.fps, (largura, altura)
            )
            if escritor.isOpened():
                self._escritor, self.codec, self._tamanho = escritor, codec, (largura, altura)
                logger.info("Gravando %s com codec %s (%dx%d).", self.caminho.name, codec, largura, altura)
                return
            escritor.release()
        raise RuntimeError("Nenhum codec de vídeo disponível para gravar o resultado.")

    def escrever(self, frame: np.ndarray) -> None:
        """Acrescenta um frame (redimensiona se o tamanho mudar, ex.: webcam)."""
        altura, largura = frame.shape[:2]
        if self._escritor is None:
            self._abrir(largura, altura)
        assert self._escritor is not None and self._tamanho is not None
        if (largura, altura) != self._tamanho:
            frame = cv2.resize(frame, self._tamanho)
        self._escritor.write(frame)
        self.frames_gravados += 1

    @property
    def reproduz_no_navegador(self) -> bool:
        return self.codec == "avc1"

    def fechar(self) -> Path | None:
        """Finaliza o arquivo. Devolve o caminho, ou ``None`` se nada foi gravado."""
        if self._escritor is not None:
            self._escritor.release()
            self._escritor = None
        return self.caminho if self.frames_gravados > 0 else None
