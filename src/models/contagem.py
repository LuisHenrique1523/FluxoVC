"""Contagem de veículos por cruzamento de uma linha virtual.

Para a reta que passa por ``A`` e ``B``, o **produto vetorial**

    lado(P) = (B − A) × (P − A) = (Bx − Ax)(Py − Ay) − (By − Ay)(Px − Ax)

é positivo de um lado, negativo do outro e zero sobre a reta. Um veículo
cruzou quando o sinal muda entre duas posições consecutivas **e** o
deslocamento corta o segmento AB (não só o prolongamento da reta).

Sentidos: ``A→B`` = do lado negativo para o positivo; ``B→A`` = o contrário.
Numa linha horizontal desenhada da esquerda para a direita (com o eixo y da
imagem para baixo), ``A→B`` é de cima para baixo.

Cada ID é contado **uma única vez**: o tremor da caixa sobre a linha não gera
contagens duplicadas.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from src.config import CLASSES_VEICULO
from src.geometry.erros import ErroGeometria

SENTIDO_AB = "A→B"
SENTIDO_BA = "B→A"
SENTIDOS: tuple[str, str] = (SENTIDO_AB, SENTIDO_BA)
EPS: float = 1e-9

Ponto = Sequence[float]


def lado_da_linha(ponto: Ponto, a: Ponto, b: Ponto) -> float:
    """Produto vetorial ``(b − a) × (ponto − a)``: o sinal indica o lado da reta."""
    return (b[0] - a[0]) * (ponto[1] - a[1]) - (b[1] - a[1]) * (ponto[0] - a[0])


def _sinal(valor: float) -> int:
    return 0 if abs(valor) <= EPS else (1 if valor > 0 else -1)


def sentido_cruzamento(anterior: Ponto, atual: Ponto, a: Ponto, b: Ponto) -> str | None:
    """Sentido em que o deslocamento ``anterior → atual`` cruza o segmento AB, ou ``None``.

    Pontos exatamente sobre a linha não contam como cruzamento: o contador
    compara sempre com a última posição *fora* da linha.
    """
    lado_ant, lado_atu = _sinal(lado_da_linha(anterior, a, b)), _sinal(lado_da_linha(atual, a, b))
    if lado_ant == 0 or lado_atu == 0 or lado_ant == lado_atu:
        return None
    # A e B precisam estar em lados opostos (ou sobre) a reta do deslocamento.
    if _sinal(lado_da_linha(a, anterior, atual)) * _sinal(lado_da_linha(b, anterior, atual)) > 0:
        return None  # cruzou o prolongamento da linha, fora do segmento
    return SENTIDO_AB if lado_ant < 0 else SENTIDO_BA


@dataclass(frozen=True)
class EventoCruzamento:
    """Registro de um veículo contado."""

    id_rastreio: Hashable
    classe: int
    sentido: str
    frame: int
    timestamp_s: float
    velocidade_kmh: float | None = None

    @property
    def nome_classe(self) -> str:
        return CLASSES_VEICULO.get(self.classe, str(self.classe))


class ContadorLinha:
    """Conta cruzamentos da linha ``p1 → p2`` (pixels do frame original)."""

    def __init__(self, p1: Ponto, p2: Ponto) -> None:
        a, b = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
        if a.shape != (2,) or b.shape != (2,) or not np.all(np.isfinite([*a, *b])):
            raise ErroGeometria("A linha de contagem precisa de dois pontos (x, y).")
        if np.linalg.norm(b - a) < 1.0:
            raise ErroGeometria("Os dois pontos da linha de contagem precisam ser diferentes.")
        self.p1: tuple[float, float] = (float(a[0]), float(a[1]))
        self.p2: tuple[float, float] = (float(b[0]), float(b[1]))
        self.reiniciar()

    def reiniciar(self) -> None:
        """Zera contagens e histórico."""
        self._ultimo_fora: dict[Hashable, tuple[float, float]] = {}
        self._contados: set[Hashable] = set()
        self.eventos: list[EventoCruzamento] = []
        self.contagem: Counter[tuple[int, str]] = Counter()
        self.frame_ultimo_cruzamento: int | None = None
        # eventos contados antes de a velocidade estar disponível: ID → posição na lista
        self._sem_velocidade: dict[Hashable, int] = {}

    def atualizar(
        self,
        id_rastreio: Hashable,
        classe: int,
        ponto: Ponto,
        frame: int,
        timestamp_s: float,
        velocidade_kmh: float | None = None,
    ) -> EventoCruzamento | None:
        """Registra a nova posição do ID e devolve o evento se ele acabou de cruzar."""
        if _sinal(lado_da_linha(ponto, self.p1, self.p2)) == 0:
            return None  # sobre a linha: espera sair para decidir
        atual = (float(ponto[0]), float(ponto[1]))
        anterior = self._ultimo_fora.get(id_rastreio)
        self._ultimo_fora[id_rastreio] = atual
        if anterior is None or id_rastreio in self._contados:
            return None

        sentido = sentido_cruzamento(anterior, atual, self.p1, self.p2)
        if sentido is None:
            return None
        evento = EventoCruzamento(id_rastreio, classe, sentido, frame, timestamp_s, velocidade_kmh)
        self._contados.add(id_rastreio)
        self.eventos.append(evento)
        self.contagem[(classe, sentido)] += 1
        self.frame_ultimo_cruzamento = frame
        if velocidade_kmh is None:
            self._sem_velocidade[id_rastreio] = len(self.eventos) - 1
        return evento

    def completar_velocidade(self, id_rastreio: Hashable, velocidade_kmh: float | None) -> bool:
        """Preenche a velocidade de um evento que foi contado sem ela.

        Acontece quando o veículo cruza a linha logo depois de aparecer, antes de
        ter histórico suficiente no trecho calibrado. Devolve ``True`` se preencheu.
        """
        if velocidade_kmh is None or id_rastreio not in self._sem_velocidade:
            return False
        indice = self._sem_velocidade.pop(id_rastreio)
        self.eventos[indice] = replace(self.eventos[indice], velocidade_kmh=velocidade_kmh)
        return True

    def esquecer(self, id_rastreio: Hashable) -> None:
        """Descarta a última posição do ID (ele continua marcado como contado)."""
        self._ultimo_fora.pop(id_rastreio, None)

    # -- consultas ------------------------------------------------------------
    @property
    def total(self) -> int:
        return sum(self.contagem.values())

    def total_por_sentido(self, sentido: str) -> int:
        return sum(n for (_, s), n in self.contagem.items() if s == sentido)

    def por_classe(self) -> dict[str, dict[str, int]]:
        """``{"carro": {"A→B": 3, "B→A": 1, "total": 4}, ...}`` para todas as classes."""
        tabela: dict[str, dict[str, int]] = {}
        for classe, nome in CLASSES_VEICULO.items():
            linha = {s: self.contagem.get((classe, s), 0) for s in SENTIDOS}
            linha["total"] = sum(linha.values())
            tabela[nome] = linha
        return tabela
