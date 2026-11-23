"""Wrapper do YOLO (Ultralytics) com rastreador ByteTrack embutido.

Recebe a imagem **já pré-processada** (quadrado do letterbox) e devolve as
detecções no espaço do modelo; o letterbox inverso é feito pelo analisador.
Como a imagem já tem o tamanho ``imgsz``, o letterbox interno do Ultralytics
não altera nada e as coordenadas voltam no mesmo espaço que enviamos.

Licença: Ultralytics é AGPL-3.0; os pesos pré-treinados vêm do dataset COCO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PESOS_DIR, ModeloConfig
from src.models.tipos import DeteccaoBruta, ErroModelo, PesosIndisponiveisError

logger = logging.getLogger(__name__)

URL_PESOS = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{nome}"
CLASSE_PESSOA = 0  # COCO; usada só para anonimizar o frame enviado ao VLM


def escolher_dispositivo(preferencia: str = "auto") -> str:
    """Resolve ``"auto"`` para GPU (``"0"``) se houver CUDA, senão ``"cpu"``."""
    if preferencia != "auto":
        return preferencia
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:  # pragma: no cover - torch vem com o ultralytics
        pass
    return "cpu"


def _para_numpy(valor: Any) -> np.ndarray:
    """Converte tensores do PyTorch (ou arrays) em ``np.ndarray``."""
    if hasattr(valor, "cpu"):
        valor = valor.cpu()
    if hasattr(valor, "numpy"):
        valor = valor.numpy()
    return np.asarray(valor)


def converter_resultado(resultado: Any) -> list[DeteccaoBruta]:
    """Converte um ``ultralytics.engine.results.Results`` em :class:`DeteccaoBruta`."""
    caixas = getattr(resultado, "boxes", None)
    if caixas is None or len(caixas) == 0:
        return []
    xyxy = _para_numpy(caixas.xyxy).reshape(-1, 4)
    classes = _para_numpy(caixas.cls).astype(int).ravel()
    confs = _para_numpy(caixas.conf).astype(float).ravel()
    ids = None if caixas.id is None else _para_numpy(caixas.id).astype(int).ravel()
    return [
        DeteccaoBruta(
            caixa=tuple(float(v) for v in xyxy[i]),  # type: ignore[arg-type]
            classe=int(classes[i]),
            confianca=float(confs[i]),
            id_rastreio=None if ids is None else int(ids[i]),
        )
        for i in range(len(xyxy))
    ]


class DetectorYOLO:
    """YOLO11/YOLOv8 + ByteTrack. Os parâmetros são lidos de ``cfg`` a cada chamada,
    então a interface pode mudar confiança e IoU sem recarregar o modelo."""

    def __init__(
        self, cfg: ModeloConfig, tamanho_entrada: int = 640, pasta_pesos: Path = PESOS_DIR
    ) -> None:
        self.cfg = cfg
        self.tamanho_entrada = tamanho_entrada
        self.dispositivo = escolher_dispositivo(cfg.dispositivo)
        self.caminho_pesos = pasta_pesos / cfg.pesos
        self._modelo = self._carregar()

    def _carregar(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as erro:
            raise ErroModelo(
                "A biblioteca 'ultralytics' não está instalada. Rode: pip install -r requirements.txt"
            ) from erro

        baixar = not self.caminho_pesos.exists()
        if baixar:
            logger.info("Baixando pesos '%s' para %s…", self.cfg.pesos, self.caminho_pesos.parent)
            self.caminho_pesos.parent.mkdir(parents=True, exist_ok=True)
        try:
            modelo = YOLO(str(self.caminho_pesos))
        except Exception as erro:  # rede, arquivo corrompido, nome inválido…
            logger.exception("Falha ao carregar os pesos %s", self.caminho_pesos)
            if baixar:
                raise PesosIndisponiveisError(
                    f"Não foi possível baixar os pesos '{self.cfg.pesos}'. Na primeira execução é "
                    "preciso estar conectado à internet. Alternativa: baixe manualmente em "
                    f"{URL_PESOS.format(nome=self.cfg.pesos)} e salve em 'pesos/{self.cfg.pesos}'."
                ) from erro
            raise ErroModelo(
                f"Os pesos em 'pesos/{self.cfg.pesos}' parecem corrompidos. "
                "Apague o arquivo para que ele seja baixado novamente."
            ) from erro
        logger.info("Modelo %s carregado (dispositivo: %s).", self.cfg.pesos, self.dispositivo)
        return modelo

    def _argumentos(self) -> dict[str, Any]:
        return {
            "conf": self.cfg.confianca,
            "iou": self.cfg.iou,
            "classes": list(self.cfg.classes),
            "imgsz": self.tamanho_entrada,
            "device": self.dispositivo,
            "verbose": False,
        }

    def rastrear(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        """Detecta e rastreia (ByteTrack mantém os IDs entre chamadas)."""
        resultados = self._modelo.track(
            imagem, persist=True, tracker=self.cfg.rastreador, **self._argumentos()
        )
        return converter_resultado(resultados[0])

    def detectar(self, imagem: np.ndarray) -> list[DeteccaoBruta]:
        """Só detecta (sem IDs) — usado na análise de imagem única."""
        return converter_resultado(self._modelo.predict(imagem, **self._argumentos())[0])

    def detectar_pessoas(self, imagem_original: np.ndarray, confianca: float = 0.25) -> list[tuple[float, float, float, float]]:
        """Caixas de **pessoas** no frame original — usadas só para anonimizar antes do VLM.

        A pipeline de contagem nunca usa a classe pessoa. Aqui o próprio
        Ultralytics faz o letterbox e devolve as caixas no espaço do original.
        """
        resultado = self._modelo.predict(
            imagem_original, conf=confianca, classes=[CLASSE_PESSOA], imgsz=self.tamanho_entrada,
            device=self.dispositivo, verbose=False,
        )[0]
        return [d.caixa for d in converter_resultado(resultado)]

    def reiniciar(self) -> None:
        """Zera o ByteTrack para começar um vídeo novo com IDs a partir de 1."""
        preditor = getattr(self._modelo, "predictor", None)
        for rastreador in getattr(preditor, "trackers", None) or []:
            rastreador.reset()
