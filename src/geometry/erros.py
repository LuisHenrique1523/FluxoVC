"""Exceções da etapa de geometria, com mensagens amigáveis para a interface."""

from __future__ import annotations


class ErroGeometria(ValueError):
    """Parâmetro geométrico inválido informado pelo usuário."""


class RoiInvalidaError(ErroGeometria):
    """ROI com menos de 3 pontos, fora do frame ou sem área."""


class HomografiaInvalidaError(ErroGeometria):
    """Pontos de calibração degenerados ou medidas reais inválidas."""
