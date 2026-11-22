"""Exceções da etapa de aquisição.

Todas carregam uma mensagem em português pronta para ser exibida na interface
(``st.error(str(erro))``), sem detalhes técnicos que confundam o usuário.
"""

from __future__ import annotations


class ErroEntrada(Exception):
    """Erro de entrada com mensagem amigável para o usuário."""


class ArquivoInvalidoError(ErroEntrada):
    """Arquivo inexistente, vazio, grande demais, com formato errado ou corrompido."""


class WebcamIndisponivelError(ErroEntrada):
    """Nenhuma câmera respondeu no índice informado."""


class FonteInvalidaError(ErroEntrada):
    """Fonte de vídeo (URL/stream) inválida ou inacessível."""
