"""Fontes de entrada: arquivo de vídeo, vídeo de demonstração, webcam, URL e imagem.

Fluxo típico (a interface só chama estas funções)::

    with abrir_arquivo(caminho, cfg.aquisicao) as fonte:
        st.write(fonte.info)
        for frame in fonte:
            processar(frame.imagem)

Toda abertura **valida** a fonte antes de devolvê-la: o arquivo abre no
``cv2.VideoCapture``, o primeiro frame é lido de fato, a resolução atende ao
mínimo e o FPS é válido (senão usa o fallback). Problemas viram exceções de
:mod:`src.acquisition.erros` com mensagem amigável; nada trava a aplicação.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from urllib.parse import urlparse

import cv2
import numpy as np

from src.acquisition.erros import (
    ArquivoInvalidoError,
    ErroEntrada,
    FonteInvalidaError,
    WebcamIndisponivelError,
)
from src.acquisition.validacao import (
    fps_valido,
    normalizar_total_frames,
    validar_existencia,
    validar_extensao,
    validar_resolucao,
    validar_tamanho,
)
from src.config import DATA_DIR, AquisicaoConfig

logger = logging.getLogger(__name__)

ESQUEMAS_URL: tuple[str, ...] = ("http", "https", "rtsp", "rtmp")
TIMEOUT_URL_MS: int = 8000
TENTATIVAS_WEBCAM: int = 5  # algumas câmeras devolvem frames vazios logo ao abrir
PASTA_UPLOADS: Path = Path(tempfile.gettempdir()) / "fluxovc_uploads"


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InfoVideo:
    """Metadados validados de uma fonte de vídeo."""

    origem: str                 # descrição legível: nome do arquivo, "webcam 0" ou URL
    tipo: str                   # "arquivo", "webcam" ou "url"
    largura: int
    altura: int
    fps: float
    fps_estimado: bool          # True se o FPS real era inválido e usamos o fallback
    total_frames: int | None    # None quando desconhecido (webcam/stream)

    @property
    def ao_vivo(self) -> bool:
        """Fonte em tempo real (o timestamp vem do relógio, não do índice do frame)."""
        return self.tipo == "webcam"

    @property
    def duracao_s(self) -> float | None:
        """Duração estimada em segundos, se o total de frames for conhecido."""
        if self.total_frames is None:
            return None
        return self.total_frames / self.fps


@dataclass(frozen=True)
class Frame:
    """Um frame lido da fonte."""

    indice: int                 # 0, 1, 2, ...
    timestamp_s: float          # segundos desde o início do vídeo/captura
    imagem: np.ndarray          # BGR, uint8, formato (altura, largura, 3)


# ---------------------------------------------------------------------------
# Fonte de vídeo
# ---------------------------------------------------------------------------
class FonteVideo:
    """Envolve um ``cv2.VideoCapture`` já validado.

    Use como gerenciador de contexto (``with``) para garantir que a câmera ou
    o arquivo sejam liberados. Pode ser percorrida com ``for`` ou frame a frame
    com :meth:`ler_frame` (útil no laço da interface com botão "Parar").
    """

    def __init__(self, captura: cv2.VideoCapture, info: InfoVideo, primeiro: np.ndarray) -> None:
        self._captura: cv2.VideoCapture | None = captura
        self.info = info
        self._pendente: np.ndarray | None = primeiro  # frame lido durante a validação
        self._proximo_indice = 0
        self._inicio: float | None = None

    # -- leitura ------------------------------------------------------------
    def ler_frame(self) -> Frame | None:
        """Lê o próximo frame; devolve ``None`` no fim do vídeo ou se a fonte foi liberada."""
        if self._captura is None:
            return None
        if self._inicio is None:
            self._inicio = time.monotonic()

        if self._pendente is not None:
            imagem, self._pendente = self._pendente, None
        else:
            ok, imagem = self._captura.read()
            if not ok or imagem is None:
                self._registrar_fim()
                return None

        indice = self._proximo_indice
        self._proximo_indice += 1
        if self.info.ao_vivo:
            timestamp = time.monotonic() - self._inicio
        else:
            timestamp = indice / self.info.fps
        return Frame(indice=indice, timestamp_s=timestamp, imagem=imagem)

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.ler_frame()) is not None:
            yield frame

    @property
    def frames_lidos(self) -> int:
        """Quantos frames já foram entregues."""
        return self._proximo_indice

    def _registrar_fim(self) -> None:
        total = self.info.total_frames
        if total is not None and self._proximo_indice < 0.9 * total:
            logger.warning(
                "Leitura de '%s' terminou no frame %d de %d esperados (arquivo truncado?).",
                self.info.origem, self._proximo_indice, total,
            )
        else:
            logger.info("Fim da fonte '%s' após %d frames.", self.info.origem, self._proximo_indice)

    # -- ciclo de vida ------------------------------------------------------
    @property
    def aberta(self) -> bool:
        return self._captura is not None

    def liberar(self) -> None:
        """Libera o arquivo/câmera. Pode ser chamado mais de uma vez."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
            self._pendente = None

    def __enter__(self) -> FonteVideo:
        return self

    def __exit__(
        self,
        tipo: type[BaseException] | None,
        valor: BaseException | None,
        rastro: TracebackType | None,
    ) -> None:
        self.liberar()

    def __del__(self) -> None:  # rede de segurança caso esqueçam o ``with``
        self.liberar()


def _validar_captura(
    captura: cv2.VideoCapture,
    origem: str,
    tipo: str,
    cfg: AquisicaoConfig,
    tentativas: int = 1,
) -> FonteVideo:
    """Lê o primeiro frame, confere resolução/FPS e monta a :class:`FonteVideo`.

    Em caso de falha libera a captura e relança o erro.
    """
    try:
        primeiro: np.ndarray | None = None
        for _ in range(max(1, tentativas)):
            ok, imagem = captura.read()
            if ok and imagem is not None and imagem.size > 0:
                primeiro = imagem
                break
        if primeiro is None:
            if tipo == "webcam":
                raise WebcamIndisponivelError(
                    f"A câmera '{origem}' abriu, mas não enviou imagens. "
                    "Ela pode estar em uso por outro programa."
                )
            raise ArquivoInvalidoError(
                f"'{origem}' não contém frames legíveis (arquivo corrompido ou vazio)."
            )

        altura, largura = primeiro.shape[:2]
        validar_resolucao(largura, altura, cfg.resolucao_min)

        fps, estimado = fps_valido(captura.get(cv2.CAP_PROP_FPS), cfg.fps_fallback)
        if estimado:
            logger.warning("FPS inválido em '%s'; usando %.1f.", origem, fps)

        total = None if tipo == "webcam" else normalizar_total_frames(
            captura.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        info = InfoVideo(origem, tipo, largura, altura, fps, estimado, total)
        logger.info(
            "Fonte aberta: %s (%s) %dx%d @ %.1f fps, frames=%s",
            origem, tipo, largura, altura, fps, total,
        )
        return FonteVideo(captura, info, primeiro)
    except ErroEntrada:
        captura.release()
        raise


# ---------------------------------------------------------------------------
# Aberturas por tipo de fonte
# ---------------------------------------------------------------------------
def abrir_arquivo(caminho: str | Path, cfg: AquisicaoConfig | None = None) -> FonteVideo:
    """Abre e valida um arquivo de vídeo (upload ou demonstração).

    Raises:
        ArquivoInvalidoError: arquivo inexistente, vazio, grande, com extensão
            não suportada, que não abre ou sem frames.
    """
    cfg = cfg or AquisicaoConfig()
    caminho = Path(caminho)
    validar_existencia(caminho)
    validar_extensao(caminho, cfg.extensoes_video)
    validar_tamanho(caminho.stat().st_size, cfg.tamanho_max_mb)

    captura = cv2.VideoCapture(str(caminho))
    if not captura.isOpened():
        captura.release()
        raise ArquivoInvalidoError(
            f"Não foi possível abrir '{caminho.name}'. O arquivo pode estar "
            "corrompido ou usar um codec não suportado."
        )
    return _validar_captura(captura, caminho.name, "arquivo", cfg)


def abrir_webcam(indice: int | None = None, cfg: AquisicaoConfig | None = None) -> FonteVideo:
    """Abre a webcam no ``indice`` informado (padrão: ``cfg.indice_webcam``).

    Raises:
        WebcamIndisponivelError: nenhuma câmera respondeu no índice.
    """
    cfg = cfg or AquisicaoConfig()
    indice = cfg.indice_webcam if indice is None else indice
    if not isinstance(indice, int) or indice < 0:
        raise WebcamIndisponivelError(f"Índice de câmera inválido: {indice!r}. Use 0, 1, 2…")

    # No Windows o DirectShow abre bem mais rápido que o backend padrão (MSMF).
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    captura = cv2.VideoCapture(indice, backend)
    if not captura.isOpened():
        captura.release()
        raise WebcamIndisponivelError(
            f"Nenhuma câmera encontrada no índice {indice}. Verifique se ela está "
            "conectada e não está sendo usada por outro programa, ou tente outro índice."
        )
    return _validar_captura(captura, f"webcam {indice}", "webcam", cfg, TENTATIVAS_WEBCAM)


def validar_url(url: str) -> str:
    """Confere se ``url`` tem esquema suportado (http, https, rtsp, rtmp).

    Returns:
        A URL sem espaços nas pontas.

    Raises:
        FonteInvalidaError: URL vazia ou com esquema não suportado.
    """
    url = (url or "").strip()
    partes = urlparse(url)
    if partes.scheme.lower() not in ESQUEMAS_URL or not partes.netloc:
        raise FonteInvalidaError(
            f"URL inválida. Use um endereço começando com {', '.join(e + '://' for e in ESQUEMAS_URL)}."
        )
    return url


def abrir_url(url: str, cfg: AquisicaoConfig | None = None) -> FonteVideo:
    """Abre um vídeo remoto ou stream (ex.: câmera IP via RTSP).

    Raises:
        FonteInvalidaError: URL inválida ou inacessível dentro do tempo limite.
    """
    cfg = cfg or AquisicaoConfig()
    url = validar_url(url)
    parametros = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_URL_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_URL_MS,
    ]
    captura = cv2.VideoCapture(url, cv2.CAP_FFMPEG, parametros)
    if not captura.isOpened():
        captura.release()
        raise FonteInvalidaError(
            "Não foi possível acessar o vídeo nessa URL. Confira o endereço e a conexão."
        )
    try:
        return _validar_captura(captura, url, "url", cfg)
    except ArquivoInvalidoError as erro:
        raise FonteInvalidaError(str(erro)) from erro


# ---------------------------------------------------------------------------
# Vídeos de demonstração e upload
# ---------------------------------------------------------------------------
def listar_videos_demo(
    pasta: Path = DATA_DIR, extensoes: tuple[str, ...] | None = None
) -> list[Path]:
    """Lista os vídeos de demonstração disponíveis em ``pasta`` (ordem alfabética).

    Devolve lista vazia se a pasta não existir ou não houver vídeos.
    """
    extensoes = tuple(e.lower() for e in (extensoes or AquisicaoConfig().extensoes_video))
    if not pasta.is_dir():
        return []
    return sorted(
        (p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in extensoes),
        key=lambda p: p.name.lower(),
    )


def _nome_seguro(nome: str) -> str:
    """Remove caracteres problemáticos (acentos, espaços, barras) do nome do arquivo."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(nome).stem).strip("_")
    return base[:50] or "video"


def salvar_upload(
    dados: bytes,
    nome_original: str,
    cfg: AquisicaoConfig | None = None,
    destino: Path | None = None,
) -> Path:
    """Valida e grava em disco um vídeo enviado pela interface.

    O ``cv2.VideoCapture`` precisa de um caminho de arquivo, por isso o upload
    é salvo numa pasta temporária do sistema (fora do OneDrive/repositório),
    com nome único e sem caracteres especiais.

    Raises:
        ArquivoInvalidoError: extensão não suportada, arquivo vazio ou grande demais.
    """
    cfg = cfg or AquisicaoConfig()
    extensao = validar_extensao(nome_original, cfg.extensoes_video)
    validar_tamanho(len(dados), cfg.tamanho_max_mb)

    destino = destino or PASTA_UPLOADS
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{uuid.uuid4().hex[:8]}_{_nome_seguro(nome_original)}{extensao}"
    caminho.write_bytes(dados)
    logger.info("Upload '%s' salvo em %s (%d bytes).", nome_original, caminho, len(dados))
    return caminho


# ---------------------------------------------------------------------------
# Imagem única (análise estática)
# ---------------------------------------------------------------------------
def decodificar_imagem(dados: bytes, nome: str, cfg: AquisicaoConfig | None = None) -> np.ndarray:
    """Decodifica uma imagem (jpg/png) a partir dos bytes, sem gravar em disco.

    Returns:
        Imagem BGR ``uint8`` com formato ``(altura, largura, 3)``.

    Raises:
        ArquivoInvalidoError: extensão, tamanho, conteúdo ou resolução inválidos.
    """
    cfg = cfg or AquisicaoConfig()
    validar_extensao(nome, cfg.extensoes_imagem)
    validar_tamanho(len(dados), cfg.tamanho_max_mb)
    imagem = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ArquivoInvalidoError(
            f"Não foi possível ler a imagem '{Path(nome).name}'. "
            "O arquivo pode estar corrompido ou não ser uma imagem."
        )
    altura, largura = imagem.shape[:2]
    validar_resolucao(largura, altura, cfg.resolucao_min)
    return imagem


def carregar_imagem(caminho: str | Path, cfg: AquisicaoConfig | None = None) -> np.ndarray:
    """Lê uma imagem do disco (funciona também com caminhos acentuados no Windows).

    Raises:
        ArquivoInvalidoError: ver :func:`decodificar_imagem`.
    """
    caminho = Path(caminho)
    validar_existencia(caminho)
    return decodificar_imagem(caminho.read_bytes(), caminho.name, cfg)
