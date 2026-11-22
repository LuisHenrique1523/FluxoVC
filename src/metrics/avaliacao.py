"""Execução das avaliações: um vídeo com uma configuração, e os frames anotados.

O detector é recebido pronto (injeção), para que os testes usem um detector
falso e o script de avaliação use o YOLO.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.acquisition.fontes import abrir_arquivo, carregar_imagem
from src.aplicacao.sessao import SessaoAnalise
from src.config import CLASSES_VEICULO, AppConfig
from src.metrics.anotacoes import FrameAnotado, ler_rotulos_yolo
from src.metrics.desempenho import MonitorMemoria, resumo_tempos
from src.metrics.qualidade import Acertos, casar_deteccoes, tabela_prf
from src.models.analisador import analisar_imagem
from src.models.tipos import Detector

logger = logging.getLogger(__name__)


@dataclass
class ResultadoVideo:
    """Tudo o que foi medido numa execução (vídeo × modelo × frame skip)."""

    video: str
    modelo: str
    frame_skip: int
    frames: int
    duracao_video_s: float
    contagem: Counter[tuple[int, str]]          # (classe, sentido) → quantidade
    frames_por_id: Counter[int]
    tempos: pd.DataFrame                        # ms por etapa, um frame processado por linha
    tempo_parede_s: float
    memoria_pico_mb: float
    memoria_inicial_mb: float
    eventos: list = field(default_factory=list)

    @property
    def configuracao(self) -> str:
        nome = self.modelo.removesuffix(".pt")
        return f"{nome} · skip {self.frame_skip}" if self.frame_skip else nome

    @property
    def fps_medio(self) -> float | None:
        """Frames do vídeo por segundo de processamento (inclui os pulados pelo frame skip)."""
        return self.frames / self.tempo_parede_s if self.tempo_parede_s > 0 else None

    @property
    def tempo_real(self) -> float | None:
        """Razão FPS de processamento ÷ FPS do vídeo (≥ 1 = acompanha o tempo real)."""
        if self.duracao_video_s <= 0 or self.tempo_parede_s <= 0:
            return None
        return self.duracao_video_s / self.tempo_parede_s

    def contagem_por_classe(self) -> dict[str, int]:
        """Total por classe (somando os dois sentidos), com todas as classes."""
        return {nome: sum(n for (c, _), n in self.contagem.items() if c == classe)
                for classe, nome in CLASSES_VEICULO.items()}

    def resumo_desempenho(self) -> dict[str, float | None]:
        return {
            "fps": self.fps_medio,
            "tempo_real": self.tempo_real,
            **resumo_tempos(self.tempos),
            "memoria_pico_mb": self.memoria_pico_mb,
        }


def avaliar_video(
    caminho: Path, cfg: AppConfig, detector: Detector, max_frames: int | None = None
) -> ResultadoVideo:
    """Roda a pipeline completa (inclusive o desenho das camadas) sobre o vídeo.

    Raises:
        ErroEntrada / ErroGeometria: vídeo ou configuração inválidos.
    """
    cfg = copy.deepcopy(cfg)
    memoria = MonitorMemoria()
    sessao = SessaoAnalise(abrir_arquivo(caminho, cfg.aquisicao), cfg, detector, pasta_saida=None)
    inicio = time.perf_counter()
    while (max_frames is None or sessao.frames_lidos < max_frames) and sessao.passo() is not None:
        memoria.amostrar()
    decorrido = time.perf_counter() - inicio
    sessao.finalizar()

    resultado = ResultadoVideo(
        video=Path(caminho).name,
        modelo=cfg.modelo.pesos,
        frame_skip=cfg.preproc.frame_skip,
        frames=sessao.frames_lidos,
        duracao_video_s=sessao.frames_lidos / sessao.info.fps,
        contagem=Counter(sessao.contador.contagem),
        frames_por_id=Counter(sessao.analisador.frames_por_id),
        tempos=sessao.tempos_dataframe(),
        tempo_parede_s=decorrido,
        memoria_pico_mb=memoria.pico_mb,
        memoria_inicial_mb=memoria.inicial_mb,
        eventos=list(sessao.eventos),
    )
    logger.info(
        "%s [%s]: %d frames, %d veículos, %.1f fps",
        resultado.video, resultado.configuracao, resultado.frames,
        sum(resultado.contagem.values()), resultado.fps_medio or 0,
    )
    return resultado


def avaliar_deteccao(
    frames: list[FrameAnotado],
    nomes_classes: list[str],
    cfg: AppConfig,
    detector: Detector,
    limiar_iou: float = 0.5,
) -> pd.DataFrame:
    """Precisão, recall e F1 por classe nos frames anotados.

    A ROI é desligada: a anotação cobre o frame inteiro, e o que se mede aqui é
    o detector (com letterbox e demais pré-processamentos configurados).
    """
    cfg = copy.deepcopy(cfg)
    cfg.preproc.usar_roi = False
    total: dict[int, Acertos] = {classe: Acertos() for classe in CLASSES_VEICULO}
    for frame in frames:
        imagem = carregar_imagem(frame.imagem, cfg.aquisicao)
        altura, largura = imagem.shape[:2]
        reais = ler_rotulos_yolo(frame.rotulos, largura, altura, nomes_classes)
        veiculos, _ = analisar_imagem(imagem, cfg, detector)
        previstas = [(v.caixa, v.classe) for v in veiculos]
        for classe, acertos in casar_deteccoes(previstas, reais, limiar_iou).items():
            total[classe] = total.get(classe, Acertos()) + acertos
    return tabela_prf({CLASSES_VEICULO[c]: a for c, a in total.items()})
