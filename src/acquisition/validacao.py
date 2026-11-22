"""Regras de validação de entradas (funções puras, sem abrir vídeo).

Cada função verifica **uma** condição e lança :class:`ArquivoInvalidoError`
com mensagem amigável quando ela não é atendida. A abertura do vídeo em si
(``cv2.VideoCapture``) fica em :mod:`src.acquisition.fontes`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from src.acquisition.erros import ArquivoInvalidoError

BYTES_POR_MB: int = 1024 * 1024

# FPS acima disso é considerado leitura inválida do contêiner (ex.: 90000).
FPS_MAXIMO_PLAUSIVEL: float = 240.0


def validar_existencia(caminho: Path) -> None:
    """Garante que ``caminho`` existe e é um arquivo comum.

    Raises:
        ArquivoInvalidoError: se o arquivo não existe ou é uma pasta.
    """
    if not caminho.exists():
        raise ArquivoInvalidoError(f"Arquivo não encontrado: '{caminho.name}'.")
    if not caminho.is_file():
        raise ArquivoInvalidoError(f"'{caminho.name}' não é um arquivo.")


def validar_extensao(nome: str | Path, permitidas: Iterable[str]) -> str:
    """Confere a extensão do arquivo (sem diferenciar maiúsculas/minúsculas).

    Args:
        nome: nome ou caminho do arquivo.
        permitidas: extensões aceitas, com ponto (ex.: ``(".mp4", ".avi")``).

    Returns:
        A extensão normalizada em minúsculas (ex.: ``".mp4"``).

    Raises:
        ArquivoInvalidoError: se a extensão não está na lista.
    """
    permitidas = tuple(e.lower() for e in permitidas)
    extensao = Path(nome).suffix.lower()
    if extensao not in permitidas:
        descricao = f"'{extensao}'" if extensao else "sem extensão"
        raise ArquivoInvalidoError(
            f"Formato {descricao} não suportado. Use: {', '.join(permitidas)}."
        )
    return extensao


def validar_tamanho(n_bytes: int, max_mb: float) -> None:
    """Rejeita arquivos vazios ou maiores que ``max_mb`` megabytes.

    Raises:
        ArquivoInvalidoError: se o arquivo está vazio ou excede o limite.
    """
    if n_bytes <= 0:
        raise ArquivoInvalidoError("O arquivo está vazio.")
    if n_bytes > max_mb * BYTES_POR_MB:
        raise ArquivoInvalidoError(
            f"Arquivo muito grande ({n_bytes / BYTES_POR_MB:.1f} MB). "
            f"Limite: {max_mb:g} MB."
        )


def validar_resolucao(largura: int, altura: int, minimo: tuple[int, int]) -> None:
    """Exige resolução mínima ``minimo = (largura, altura)``.

    Raises:
        ArquivoInvalidoError: se o frame é menor que o mínimo em qualquer eixo.
    """
    min_l, min_a = minimo
    if largura < min_l or altura < min_a:
        raise ArquivoInvalidoError(
            f"Resolução {largura}×{altura} é baixa demais para detectar veículos. "
            f"Mínimo: {min_l}×{min_a}."
        )


def fps_valido(fps: float | None, fallback: float) -> tuple[float, bool]:
    """Devolve um FPS utilizável.

    Alguns contêineres e webcams informam FPS 0, negativo, NaN ou absurdo
    (ex.: 90000). Nesses casos usamos ``fallback``.

    Returns:
        Tupla ``(fps, usou_fallback)``.
    """
    if fps is None or not math.isfinite(fps) or fps <= 0 or fps > FPS_MAXIMO_PLAUSIVEL:
        return float(fallback), True
    return float(fps), False


def normalizar_total_frames(valor: float | None) -> int | None:
    """Converte ``CAP_PROP_FRAME_COUNT`` em inteiro, ou ``None`` se desconhecido.

    Webcams e streams retornam 0 ou -1; alguns contêineres retornam valores
    negativos enormes. Tudo isso vira ``None`` ("duração desconhecida").
    """
    if valor is None or not math.isfinite(valor) or valor <= 0:
        return None
    return int(valor)
