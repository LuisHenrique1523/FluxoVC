"""Parâmetros padrão do FluxoVC.

Todos os valores ajustáveis da aplicação ficam aqui, agrupados em dataclasses.
A interface Streamlit parte de uma instância de :class:`AppConfig` e altera os
campos conforme o usuário mexe nos controles, sem precisar editar código.

Coordenadas de linha, ROI e homografia são **normalizadas** (0 a 1 em relação à
largura/altura do frame), para funcionarem com vídeos de qualquer resolução.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (sempre relativos à raiz do projeto)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data_samples"
GROUND_TRUTH_DIR: Path = DATA_DIR / "ground_truth"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

PontoNorm = tuple[float, float]


# ---------------------------------------------------------------------------
# Classes de veículo do COCO usadas pelo detector
# ---------------------------------------------------------------------------
CLASSES_VEICULO: dict[int, str] = {
    2: "carro",
    3: "moto",
    5: "ônibus",
    7: "caminhão",
}

# Cores BGR (OpenCV) por classe
CORES_CLASSE: dict[int, tuple[int, int, int]] = {
    2: (255, 170, 0),    # carro — azul
    3: (0, 200, 255),    # moto — amarelo/laranja
    5: (60, 200, 60),    # ônibus — verde
    7: (60, 60, 230),    # caminhão — vermelho
}


@dataclass
class AquisicaoConfig:
    """Fontes de entrada e regras de validação."""

    extensoes_video: tuple[str, ...] = (".mp4", ".avi", ".mov")
    extensoes_imagem: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    tamanho_max_mb: float = 200.0
    resolucao_min: tuple[int, int] = (320, 240)  # (largura, altura)
    fps_fallback: float = 30.0
    indice_webcam: int = 0


@dataclass
class PreprocConfig:
    """Operações de pré-processamento (cada uma pode ser ligada/desligada)."""

    tamanho_entrada: int = 640          # lado do quadrado do letterbox
    usar_roi: bool = True
    usar_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grade: int = 8
    usar_reducao_ruido: bool = False
    tipo_filtro: str = "bilateral"      # "bilateral" ou "gaussiano"
    kernel_ruido: int = 5               # ímpar
    frame_skip: int = 0                 # 0 = processa todos os frames


@dataclass
class ModeloConfig:
    """Detector YOLO + rastreador ByteTrack."""

    pesos: str = "yolo11n.pt"           # "yolo11n.pt" ou "yolo11s.pt"
    confianca: float = 0.35
    iou: float = 0.5
    rastreador: str = "bytetrack.yaml"
    dispositivo: str = "auto"           # "auto", "cpu" ou "cuda"
    classes: tuple[int, ...] = tuple(CLASSES_VEICULO)


@dataclass
class ContagemConfig:
    """Linha virtual de contagem (dois pontos normalizados)."""

    linha_p1: PontoNorm = (0.05, 0.60)
    linha_p2: PontoNorm = (0.95, 0.60)


@dataclass
class GeometriaConfig:
    """ROI, homografia para velocidade e suavização de trajetórias."""

    roi: list[PontoNorm] = field(
        default_factory=lambda: [(0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)]
    )
    # 4 pontos da pista na imagem (sup. esq., sup. dir., inf. dir., inf. esq.)
    homografia_origem: list[PontoNorm] = field(
        default_factory=lambda: [(0.35, 0.40), (0.65, 0.40), (0.95, 0.95), (0.05, 0.95)]
    )
    largura_pista_m: float = 7.0        # largura real do trecho (metros)
    comprimento_pista_m: float = 20.0   # comprimento real do trecho (metros)
    ema_alpha: float = 0.4              # 0 < α ≤ 1; maior = menos suavização


@dataclass
class VisualizacaoConfig:
    """Camadas de desenho (cada uma pode ser ligada/desligada)."""

    mostrar_caixas: bool = True
    mostrar_trajetorias: bool = True
    mostrar_linha: bool = True
    mostrar_roi: bool = True
    mostrar_heatmap: bool = False
    mostrar_birdseye: bool = True
    mostrar_hud: bool = True
    opacidade_heatmap: float = 0.45
    opacidade_roi: float = 0.20
    tamanho_trajetoria: int = 40        # nº máximo de pontos guardados por ID
    largura_exibicao: int = 960         # largura do vídeo exibido na interface


@dataclass
class AppConfig:
    """Agrupa todos os parâmetros da aplicação."""

    aquisicao: AquisicaoConfig = field(default_factory=AquisicaoConfig)
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    modelo: ModeloConfig = field(default_factory=ModeloConfig)
    contagem: ContagemConfig = field(default_factory=ContagemConfig)
    geometria: GeometriaConfig = field(default_factory=GeometriaConfig)
    visualizacao: VisualizacaoConfig = field(default_factory=VisualizacaoConfig)
