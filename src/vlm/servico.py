"""Liga a pipeline ao VLM: monta o frame anonimizado e consulta o backend."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.vlm.anonimizacao import anonimizar
from src.vlm.backends import VLM, RespostaVLM

logger = logging.getLogger(__name__)

PERGUNTA_DESCRICAO_PT = (
    "Descreva esta cena de trânsito em até 3 frases: tipo de via, quantidade aproximada e tipos "
    "de veículos, e se o trânsito parece livre ou congestionado."
)
PERGUNTA_DESCRICAO_EN = (
    "Describe this traffic scene in up to 3 sentences: the type of road, the approximate number "
    "and types of vehicles, and whether traffic looks free-flowing or congested."
)


def pergunta_para(backend: str, pt: str, en: str) -> str:
    """O modelo local (SmolVLM) entende bem só inglês; a API recebe em português."""
    return en if backend == "local" else pt


@dataclass(frozen=True)
class FrameAnonimizado:
    imagem: np.ndarray           # BGR reduzido, com pessoas e placas pixelizadas
    pessoas: int                 # quantas pessoas foram pixelizadas
    veiculos: int                # quantas regiões de placa foram pixelizadas
    pessoas_verificadas: bool    # False se o detector não sabe achar pessoas


def preparar_frame(
    imagem: np.ndarray, caixas_veiculos: Iterable[Sequence[float]], detector: object
) -> FrameAnonimizado:
    """Anonimiza o frame original usando as caixas de veículo da pipeline e as pessoas do YOLO."""
    veiculos = [tuple(c) for c in caixas_veiculos]
    detectar_pessoas = getattr(detector, "detectar_pessoas", None)
    pessoas: list = []
    if callable(detectar_pessoas):
        pessoas = list(detectar_pessoas(imagem))
    else:
        logger.warning("Detector sem detectar_pessoas(): só as placas serão pixelizadas.")
    return FrameAnonimizado(anonimizar(imagem, pessoas, veiculos), len(pessoas), len(veiculos),
                            callable(detectar_pessoas))


def descrever(vlm: VLM, frame: FrameAnonimizado, pergunta: str | None = None) -> RespostaVLM:
    """Descreve a cena (ou responde ``pergunta``) a partir do frame **já anonimizado**."""
    texto = pergunta.strip() if pergunta and pergunta.strip() else pergunta_para(
        vlm.backend, PERGUNTA_DESCRICAO_PT, PERGUNTA_DESCRICAO_EN
    )
    return vlm.responder(frame.imagem, texto)
