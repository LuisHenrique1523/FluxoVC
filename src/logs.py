"""Configuração central de logging (o projeto não usa ``print`` solto)."""

from __future__ import annotations

import logging

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura o logger raiz uma única vez.

    Args:
        nivel: nível mínimo das mensagens (ex.: ``logging.DEBUG``).
    """
    raiz = logging.getLogger()
    if raiz.handlers:
        raiz.setLevel(nivel)
        return
    logging.basicConfig(level=nivel, format=_FORMATO, datefmt="%H:%M:%S")
