"""Orquestra um frame: pré-processamento → detector → interpretação das saídas.

Interpretação (etapa 5 da pipeline):

1. desfaz o letterbox: caixas do espaço do modelo → frame original;
2. filtra classes de veículo, confiança e ROI (pelo ponto de referência);
3. atualiza o histórico de cada ID (EMA, classe predominante, velocidade);
4. testa o cruzamento da linha virtual e registra eventos.

A interface só chama :meth:`AnalisadorFluxo.processar` frame a frame.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.acquisition.fontes import Frame
from src.config import AVISO_CALIBRACAO_PADRAO, CLASSES_VEICULO, AppConfig, calibracao_e_padrao
from src.geometry.coordenadas import caixas_modelo_para_original, normalizados_para_pixels
from src.geometry.erros import HomografiaInvalidaError, RoiInvalidaError
from src.geometry.homografia import calcular_homografia
from src.geometry.roi import ponto_na_roi, roi_em_pixels
from src.models.contagem import ContadorLinha, EventoCruzamento
from src.models.rastreamento import HistoricoTrajetorias
from src.models.tipos import DeteccaoBruta, Detector
from src.preprocessing.operacoes import deve_processar_frame
from src.preprocessing.pipeline import ResultadoPreproc, preprocessar

logger = logging.getLogger(__name__)

MODOS_PONTO: tuple[str, ...] = ("base", "centro")
MSG_SEM_DETECCOES = "Nenhum veículo detectado — tente reduzir o limiar de confiança."

Caixa = tuple[float, float, float, float]


def ponto_referencia(caixa: Caixa, modo: str = "base") -> tuple[float, float]:
    """Ponto do veículo usado na contagem: centro da base (contato com o chão) ou centro da caixa."""
    x1, y1, x2, y2 = caixa
    if modo == "base":
        return (x1 + x2) / 2, y2
    if modo == "centro":
        return (x1 + x2) / 2, (y1 + y2) / 2
    raise ValueError(f"Ponto de referência '{modo}' inválido. Use: {', '.join(MODOS_PONTO)}.")


@dataclass(frozen=True)
class VeiculoRastreado:
    """Um veículo no frame, já no espaço do frame original."""

    id_rastreio: int | None
    classe: int
    confianca: float
    caixa: Caixa                                # x1, y1, x2, y2 (px do original)
    ponto: tuple[float, float]                  # ponto de referência (suavizado se tiver ID)
    velocidade_kmh: float | None = None

    @property
    def nome_classe(self) -> str:
        return CLASSES_VEICULO.get(self.classe, str(self.classe))


@dataclass
class ResultadoFrame:
    """Tudo o que a visualização e a interface precisam de um frame."""

    frame: int
    timestamp_s: float
    processado: bool                             # False quando pulado pelo frame skip
    veiculos: list[VeiculoRastreado] = field(default_factory=list)
    eventos: list[EventoCruzamento] = field(default_factory=list)   # novos neste frame
    preproc: ResultadoPreproc | None = None
    tempos_ms: dict[str, float] = field(default_factory=dict)

    @property
    def tempo_total_ms(self) -> float:
        return sum(self.tempos_ms.values())


def interpretar_deteccoes(
    brutas: list[DeteccaoBruta],
    preproc: ResultadoPreproc,
    confianca_min: float,
    roi_px: np.ndarray | None,
    modo_ponto: str = "base",
) -> list[tuple[DeteccaoBruta, Caixa, tuple[float, float]]]:
    """Desfaz o letterbox e filtra classe, confiança, caixas degeneradas e ROI.

    Returns:
        Lista de ``(detecção, caixa no original, ponto de referência bruto)``.
    """
    if not brutas:
        return []
    caixas = caixas_modelo_para_original([d.caixa for d in brutas], preproc.letterbox)
    validas = []
    for det, caixa_arr in zip(brutas, caixas):
        if det.classe not in CLASSES_VEICULO or det.confianca < confianca_min:
            continue  # pessoas e outras classes são sempre ignoradas
        caixa: Caixa = tuple(float(v) for v in caixa_arr)  # type: ignore[assignment]
        if caixa[2] - caixa[0] < 1 or caixa[3] - caixa[1] < 1:
            continue  # caixa toda dentro da faixa do letterbox
        ponto = ponto_referencia(caixa, modo_ponto)
        if roi_px is not None and not ponto_na_roi(ponto, roi_px):
            continue
        validas.append((det, caixa, ponto))
    return validas


class AnalisadorFluxo:
    """Processa os frames de **um** vídeo, acumulando trajetórias e contagens."""

    def __init__(self, cfg: AppConfig, detector: Detector, largura: int, altura: int) -> None:
        """
        Raises:
            RoiInvalidaError: ROI ligada com polígono inválido.
            ErroGeometria: linha de contagem degenerada.
        """
        if cfg.contagem.ponto_referencia not in MODOS_PONTO:
            raise ValueError(f"Ponto de referência inválido: {cfg.contagem.ponto_referencia}")
        self.cfg = cfg
        self.detector = detector
        self.largura, self.altura = largura, altura
        self.avisos: list[str] = []

        linha = normalizados_para_pixels([cfg.contagem.linha_p1, cfg.contagem.linha_p2], largura, altura)
        self.contador = ContadorLinha(linha[0], linha[1])
        self.roi_px = self._roi_px()
        self.homografia, self.area_calibrada_px = self._calibrar()
        self.historico = HistoricoTrajetorias(
            alpha=cfg.geometria.ema_alpha,
            tamanho_max=cfg.visualizacao.tamanho_trajetoria,
            max_frames_ausente=cfg.contagem.max_frames_ausente,
            homografia=self.homografia,
            area_calibrada_px=self.area_calibrada_px,
            janela_velocidade_s=cfg.contagem.janela_velocidade_s,
        )
        self.frames_processados = 0
        self.frames_com_deteccao = 0
        # nº de frames processados em que cada ID apareceu (estabilidade do rastreamento)
        self.frames_por_id: Counter[int] = Counter()
        self._ultimo: ResultadoFrame | None = None
        detector.reiniciar()

    def _roi_px(self) -> np.ndarray | None:
        """Polígono da ROI em pixels. Só é obrigatório ser válido se a ROI estiver ligada.

        É calculado mesmo com a ROI desligada para que o toggle possa ser ligado
        e desligado durante a análise (a interface aplica os toggles ao vivo).
        """
        try:
            return roi_em_pixels(self.cfg.geometria.roi, self.largura, self.altura)
        except RoiInvalidaError:
            if self.cfg.preproc.usar_roi:
                raise
            return None

    @property
    def roi_ativa(self) -> np.ndarray | None:
        """ROI em uso neste frame (respeita o toggle atual)."""
        return self.roi_px if self.cfg.preproc.usar_roi else None

    def _calibrar(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Calcula a homografia; se for inválida, desativa só a velocidade e avisa."""
        geo = self.cfg.geometria
        origem_px = normalizados_para_pixels(geo.homografia_origem, self.largura, self.altura)
        try:
            matriz = calcular_homografia(origem_px, geo.largura_pista_m, geo.comprimento_pista_m)
        except HomografiaInvalidaError as erro:
            aviso = f"Velocidade desativada: {erro}"
            logger.warning(aviso)
            self.avisos.append(aviso)
            return None, None
        if calibracao_e_padrao(geo):
            logger.warning(AVISO_CALIBRACAO_PADRAO)
            self.avisos.append(AVISO_CALIBRACAO_PADRAO)
        return matriz, origem_px

    # ------------------------------------------------------------------
    def processar(self, frame: Frame) -> ResultadoFrame:
        """Analisa um frame (ou o pula, conforme o frame skip)."""
        if not deve_processar_frame(frame.indice, self.cfg.preproc.frame_skip):
            anteriores = self._ultimo.veiculos if self._ultimo else []
            return ResultadoFrame(frame.indice, frame.timestamp_s, False, list(anteriores))

        tempos: dict[str, float] = {}
        inicio = time.perf_counter()
        roi_norm = self.cfg.geometria.roi if self.roi_px is not None else None
        preproc = preprocessar(frame.imagem, self.cfg.preproc, roi_norm)
        tempos["preprocessamento"] = (time.perf_counter() - inicio) * 1000

        inicio = time.perf_counter()
        brutas = self.detector.rastrear(preproc.imagem)
        tempos["inferencia"] = (time.perf_counter() - inicio) * 1000

        inicio = time.perf_counter()
        veiculos, eventos = self._interpretar(brutas, preproc, frame)
        for id_ in self.historico.remover_ausentes(frame.indice):
            self.contador.esquecer(id_)
        tempos["interpretacao"] = (time.perf_counter() - inicio) * 1000

        self.frames_processados += 1
        self.frames_com_deteccao += bool(veiculos)
        resultado = ResultadoFrame(frame.indice, frame.timestamp_s, True, veiculos, eventos, preproc, tempos)
        self._ultimo = resultado
        return resultado

    def _interpretar(
        self, brutas: list[DeteccaoBruta], preproc: ResultadoPreproc, frame: Frame
    ) -> tuple[list[VeiculoRastreado], list[EventoCruzamento]]:
        veiculos: list[VeiculoRastreado] = []
        eventos: list[EventoCruzamento] = []
        validas = interpretar_deteccoes(
            brutas, preproc, self.cfg.modelo.confianca, self.roi_ativa, self.cfg.contagem.ponto_referencia
        )
        for det, caixa, ponto in validas:
            if det.id_rastreio is None:  # ainda não confirmado pelo rastreador
                veiculos.append(VeiculoRastreado(None, det.classe, det.confianca, caixa, ponto))
                continue
            traj = self.historico.atualizar(
                det.id_rastreio, det.classe, ponto, frame.indice, frame.timestamp_s
            )
            self.frames_por_id[det.id_rastreio] += 1
            evento = self.contador.atualizar(
                det.id_rastreio, traj.classe, traj.ponto_atual, frame.indice,
                frame.timestamp_s, traj.velocidade_kmh,
            )
            if evento is None:
                self.contador.completar_velocidade(det.id_rastreio, traj.velocidade_kmh)
            else:
                eventos.append(evento)
                logger.info(
                    "Contado: ID %s %s %s em %.2fs", evento.id_rastreio, evento.nome_classe,
                    evento.sentido, evento.timestamp_s,
                )
            veiculos.append(
                VeiculoRastreado(
                    det.id_rastreio, traj.classe, det.confianca, caixa, traj.ponto_atual,
                    traj.velocidade_kmh,
                )
            )
        return veiculos, eventos

    # ------------------------------------------------------------------
    @property
    def sem_deteccoes(self) -> bool:
        """``True`` se já processou frames e não achou nenhum veículo em nenhum deles."""
        return self.frames_processados > 0 and self.frames_com_deteccao == 0

    @property
    def mensagem_sem_deteccoes(self) -> str | None:
        return MSG_SEM_DETECCOES if self.sem_deteccoes else None


def analisar_imagem(
    imagem: np.ndarray, cfg: AppConfig, detector: Detector
) -> tuple[list[VeiculoRastreado], ResultadoPreproc]:
    """Análise estática de uma imagem única (sem rastreamento nem contagem).

    Raises:
        RoiInvalidaError: ROI ligada com polígono inválido.
    """
    altura, largura = imagem.shape[:2]
    preproc = preprocessar(imagem, cfg.preproc, cfg.geometria.roi)
    roi_px = roi_em_pixels(cfg.geometria.roi, largura, altura) if cfg.preproc.usar_roi else None
    validas = interpretar_deteccoes(
        detector.detectar(preproc.imagem), preproc, cfg.modelo.confianca, roi_px,
        cfg.contagem.ponto_referencia,
    )
    veiculos = [VeiculoRastreado(None, d.classe, d.confianca, c, p) for d, c, p in validas]
    return veiculos, preproc

