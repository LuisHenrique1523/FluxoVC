"""Backends do VLM: modelo local (Hugging Face) ou API do Claude.

O VLM **complementa** a pipeline: descreve a cena e responde perguntas em
linguagem natural. As contagens oficiais continuam vindo do YOLO + ByteTrack.

Configuração (arquivo ``.env`` na raiz, modelo em ``.env.example``):

* ``FLUXOVC_VLM_BACKEND``: ``local`` (padrão) ou ``claude``;
* ``FLUXOVC_VLM_MODELO_LOCAL``: modelo do Hugging Face (padrão: SmolVLM-256M);
* ``FLUXOVC_VLM_MODELO_CLAUDE``: modelo da API (padrão: ``claude-opus-5-5``);
* ``ANTHROPIC_API_KEY``: chave da API, **só** no ``.env`` (nunca no código).

O backend local roda no computador: o frame não sai da máquina. Com o backend
``claude``, o frame **já anonimizado** é enviado à API da Anthropic.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from src.config import PESOS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

BACKENDS: tuple[str, ...] = ("local", "claude")
MODELO_LOCAL_PADRAO = "HuggingFaceTB/SmolVLM-256M-Instruct"
MODELO_CLAUDE_PADRAO = "claude-opus-5-5"
PASTA_MODELOS_HF = PESOS_DIR / "hf"
AVISO_IA = "Resposta gerada por IA, pode conter erros. As contagens oficiais são as da pipeline."

SISTEMA_CLAUDE = (
    "Você analisa frames de câmeras de trânsito para um sistema de contagem de veículos. "
    "Responda em português do Brasil, de forma curta e objetiva, apenas com o que é visível "
    "na imagem. Se não for possível responder com segurança, diga isso. Regiões pixelizadas "
    "foram anonimizadas de propósito (pessoas e placas): não tente descrevê-las."
)


class ErroVLM(RuntimeError):
    """Falha ao configurar ou consultar o VLM (mensagem pronta para a interface)."""


@dataclass(frozen=True)
class RespostaVLM:
    texto: str
    latencia_s: float
    modelo: str
    backend: str


class VLM(Protocol):
    nome: str
    backend: str

    def responder(self, imagem_bgr: np.ndarray, pergunta: str) -> RespostaVLM:
        """Responde ``pergunta`` sobre a imagem (já anonimizada)."""
        ...


@dataclass(frozen=True)
class ConfigVLM:
    backend: str = "local"
    modelo_local: str = MODELO_LOCAL_PADRAO
    modelo_claude: str = MODELO_CLAUDE_PADRAO
    max_tokens: int = 200

    @classmethod
    def do_ambiente(cls, carregar_env: bool = True) -> ConfigVLM:
        """Lê o ``.env`` da raiz do projeto (se existir) e as variáveis de ambiente."""
        if carregar_env:
            try:
                from dotenv import load_dotenv

                load_dotenv(PROJECT_ROOT / ".env", override=False)
            except ImportError:  # pragma: no cover - python-dotenv está no requirements
                pass
        backend = os.getenv("FLUXOVC_VLM_BACKEND", "local").strip().lower() or "local"
        return cls(
            backend=backend,
            modelo_local=os.getenv("FLUXOVC_VLM_MODELO_LOCAL", MODELO_LOCAL_PADRAO).strip() or MODELO_LOCAL_PADRAO,
            modelo_claude=os.getenv("FLUXOVC_VLM_MODELO_CLAUDE", MODELO_CLAUDE_PADRAO).strip() or MODELO_CLAUDE_PADRAO,
        )


def bgr_para_png(imagem_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", imagem_bgr)
    if not ok:
        raise ErroVLM("Não foi possível codificar a imagem para o VLM.")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Local: SmolVLM (Hugging Face transformers)
# ---------------------------------------------------------------------------
class VLMLocal:
    """VLM pequeno rodando localmente (CPU ou GPU). Baixa o modelo na primeira vez (~500 MB).

    As perguntas vão em inglês, porque o SmolVLM foi treinado quase só em inglês.
    """

    backend = "local"

    def __init__(self, modelo: str = MODELO_LOCAL_PADRAO, max_tokens: int = 200) -> None:
        self.nome = modelo
        self.max_tokens = max_tokens
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as erro:
            raise ErroVLM("Instale as dependências: pip install -r requirements.txt") from erro
        try:
            # sem dividir a imagem em blocos: ~10× mais rápido em CPU, com pouca perda
            self._processador = AutoProcessor.from_pretrained(
                modelo, cache_dir=PASTA_MODELOS_HF, do_image_splitting=False
            )
            self._modelo = AutoModelForImageTextToText.from_pretrained(
                modelo, cache_dir=PASTA_MODELOS_HF, dtype=torch.float32
            )
        except Exception as erro:  # rede, nome inválido, disco…
            logger.exception("Falha ao carregar o VLM %s", modelo)
            raise ErroVLM(
                f"Não foi possível carregar o VLM '{modelo}'. Na primeira vez é preciso internet "
                "para baixar o modelo (~500 MB, salvo em pesos/hf/)."
            ) from erro
        self._torch = torch
        logger.info("VLM local %s carregado.", modelo)

    def responder(self, imagem_bgr: np.ndarray, pergunta: str) -> RespostaVLM:
        from PIL import Image

        imagem = Image.fromarray(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB))
        mensagens = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pergunta}]}]
        prompt = self._processador.apply_chat_template(mensagens, add_generation_prompt=True)
        entradas = self._processador(text=prompt, images=[imagem], return_tensors="pt")
        inicio = time.perf_counter()
        with self._torch.no_grad():
            saida = self._modelo.generate(**entradas, max_new_tokens=self.max_tokens, do_sample=False)
        texto = self._processador.batch_decode(
            saida[:, entradas["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        return RespostaVLM(texto, time.perf_counter() - inicio, self.nome, self.backend)


# ---------------------------------------------------------------------------
# API: Claude (Anthropic)
# ---------------------------------------------------------------------------
class VLMClaude:
    """VLM pela API do Claude. A chave vem de ``ANTHROPIC_API_KEY`` (arquivo ``.env``)."""

    backend = "claude"

    def __init__(self, modelo: str = MODELO_CLAUDE_PADRAO, max_tokens: int = 1024, cliente: Any = None) -> None:
        self.nome = modelo
        self.max_tokens = max_tokens
        if cliente is not None:  # testes injetam um cliente falso
            self._cliente = cliente
            return
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ErroVLM(
                "Backend 'claude' sem chave: defina ANTHROPIC_API_KEY no arquivo .env "
                "(veja .env.example) ou use FLUXOVC_VLM_BACKEND=local."
            )
        try:
            import anthropic
        except ImportError as erro:
            raise ErroVLM("Instale as dependências: pip install -r requirements.txt") from erro
        self._cliente = anthropic.Anthropic()

    def requisicao(self, imagem_bgr: np.ndarray, pergunta: str) -> dict[str, Any]:
        """Parâmetros enviados à API (separado para os testes conferirem)."""
        dados = base64.standard_b64encode(bgr_para_png(imagem_bgr)).decode("ascii")
        return {
            "model": self.nome,
            "max_tokens": self.max_tokens,
            "system": SISTEMA_CLAUDE,
            # tarefa simples de descrição: pouco raciocínio, resposta mais rápida e barata
            "output_config": {"effort": "low"},
            # se o modelo recusar, a API tenta outro modelo automaticamente
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": dados}},
                    {"type": "text", "text": pergunta},
                ],
            }],
        }

    def responder(self, imagem_bgr: np.ndarray, pergunta: str) -> RespostaVLM:
        try:
            import anthropic
        except ImportError as erro:  # pragma: no cover
            raise ErroVLM("Instale as dependências: pip install -r requirements.txt") from erro
        inicio = time.perf_counter()
        try:
            resposta = self._cliente.beta.messages.create(**self.requisicao(imagem_bgr, pergunta))
        except anthropic.AuthenticationError as erro:
            raise ErroVLM("Chave da API inválida (ANTHROPIC_API_KEY no .env).") from erro
        except anthropic.RateLimitError as erro:
            raise ErroVLM("Limite de uso da API atingido. Tente de novo em instantes.") from erro
        except anthropic.APIStatusError as erro:
            raise ErroVLM(f"A API recusou a requisição (HTTP {erro.status_code}).") from erro
        except anthropic.APIConnectionError as erro:
            raise ErroVLM("Sem conexão com a API. Verifique a internet.") from erro
        latencia = time.perf_counter() - inicio
        if resposta.stop_reason == "refusal":
            return RespostaVLM("(o modelo se recusou a responder)", latencia, resposta.model, self.backend)
        texto = "".join(b.text for b in resposta.content if getattr(b, "type", "") == "text").strip()
        return RespostaVLM(texto, latencia, resposta.model, self.backend)


# ---------------------------------------------------------------------------
def criar_vlm(cfg: ConfigVLM | None = None) -> VLM:
    """Cria o backend configurado no ``.env``.

    Raises:
        ErroVLM: backend desconhecido, sem chave, sem internet na primeira vez etc.
    """
    cfg = cfg or ConfigVLM.do_ambiente()
    if cfg.backend == "local":
        return VLMLocal(cfg.modelo_local, cfg.max_tokens)
    if cfg.backend == "claude":
        return VLMClaude(cfg.modelo_claude)
    raise ErroVLM(f"FLUXOVC_VLM_BACKEND='{cfg.backend}' inválido. Use: {', '.join(BACKENDS)}.")
