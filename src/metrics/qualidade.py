"""Métricas de qualidade: detecção (P/R/F1), contagem (erros, MAE) e rastreamento.

Todas as funções são puras (sem modelo nem vídeo) e testadas com valores conhecidos.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

Caixa = Sequence[float]                 # x1, y1, x2, y2
CaixaRotulada = tuple[Caixa, int]       # (caixa, classe COCO)


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------
def iou(a: Caixa, b: Caixa) -> float:
    """Interseção sobre união de duas caixas ``(x1, y1, x2, y2)``."""
    larg = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    alt = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersecao = larg * alt
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uniao = area_a + area_b - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


@dataclass
class Acertos:
    """Verdadeiros positivos, falsos positivos e falsos negativos."""

    vp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, outro: Acertos) -> Acertos:
        return Acertos(self.vp + outro.vp, self.fp + outro.fp, self.fn + outro.fn)

    @property
    def precisao(self) -> float | None:
        """VP / (VP + FP): das detecções feitas, quantas estavam certas. ``None`` sem detecções."""
        return self.vp / (self.vp + self.fp) if self.vp + self.fp else None

    @property
    def recall(self) -> float | None:
        """VP / (VP + FN): dos veículos reais, quantos foram detectados. ``None`` sem veículos reais."""
        return self.vp / (self.vp + self.fn) if self.vp + self.fn else None

    @property
    def f1(self) -> float | None:
        """Média harmônica de precisão e recall."""
        p, r = self.precisao, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


def casar_deteccoes(
    previstas: Iterable[CaixaRotulada], reais: Iterable[CaixaRotulada], limiar_iou: float = 0.5
) -> dict[int, Acertos]:
    """Casa previsões com anotações (mesma classe, IoU ≥ limiar), de forma gulosa.

    Os pares são aceitos do maior IoU para o menor e cada caixa só pode ser usada
    uma vez — o mesmo critério das avaliações de detecção do tipo PASCAL VOC/COCO.

    Returns:
        ``{classe: Acertos}`` para cada classe que aparece nas previsões ou anotações.
    """
    previstas, reais = list(previstas), list(reais)
    pares = sorted(
        (
            (iou(cp, cr), i, j)
            for i, (cp, kp) in enumerate(previstas)
            for j, (cr, kr) in enumerate(reais)
            if kp == kr
        ),
        reverse=True,
    )
    usadas_p: set[int] = set()
    usadas_r: set[int] = set()
    for valor, i, j in pares:
        if valor < limiar_iou:
            break
        if i not in usadas_p and j not in usadas_r:
            usadas_p.add(i)
            usadas_r.add(j)

    resultado: dict[int, Acertos] = {}
    for i, (_, classe) in enumerate(previstas):
        acertos = resultado.setdefault(classe, Acertos())
        if i in usadas_p:
            acertos.vp += 1
        else:
            acertos.fp += 1
    for j, (_, classe) in enumerate(reais):
        if j not in usadas_r:
            resultado.setdefault(classe, Acertos()).fn += 1
    return resultado


def tabela_prf(acertos_por_classe: Mapping[str, Acertos]) -> pd.DataFrame:
    """Tabela com VP, FP, FN, precisão, recall e F1 por classe e a linha "geral" (micro-média)."""
    linhas = dict(acertos_por_classe)
    linhas["geral"] = sum(linhas.values(), Acertos())
    return pd.DataFrame(
        [
            {"classe": nome, "VP": a.vp, "FP": a.fp, "FN": a.fn,
             "precisao": a.precisao, "recall": a.recall, "F1": a.f1}
            for nome, a in linhas.items()
        ]
    ).set_index("classe")


# ---------------------------------------------------------------------------
# Contagem
# ---------------------------------------------------------------------------
def erro_contagem(real: Mapping[str, int], previsto: Mapping[str, int]) -> pd.DataFrame:
    """Erro absoluto e percentual por categoria (classe ou classe + sentido).

    O erro percentual fica vazio quando a contagem real é 0 (divisão indefinida).
    """
    chaves = list(dict.fromkeys([*real, *previsto]))
    linhas = []
    for chave in chaves:
        r, p = int(real.get(chave, 0)), int(previsto.get(chave, 0))
        linhas.append(
            {"categoria": chave, "real": r, "previsto": p, "erro_abs": abs(p - r),
             "erro_pct": abs(p - r) / r * 100 if r else None}
        )
    return pd.DataFrame(linhas, columns=["categoria", "real", "previsto", "erro_abs", "erro_pct"]).set_index("categoria")


def mae(tabela_erros: pd.DataFrame) -> float:
    """Erro absoluto médio (MAE) entre as categorias da tabela."""
    return float(tabela_erros["erro_abs"].mean()) if len(tabela_erros) else 0.0


# ---------------------------------------------------------------------------
# Rastreamento
# ---------------------------------------------------------------------------
def estabilidade_rastreamento(
    frames_por_id: Mapping[int, int] | Counter[int], veiculos_reais: int | None, min_frames: int = 5
) -> dict[str, float | int | None]:
    """Estima a estabilidade do rastreamento sem anotação quadro a quadro.

    * IDs com menos de ``min_frames`` aparições são **fragmentos** (ruído ou
      reidentificação instantânea) e não entram na conta.
    * **IDs por veículo** = IDs válidos ÷ veículos reais (ideal = 1,0).
    * **Trocas de ID estimadas** = IDs válidos − veículos reais (≥ 0): cada
      veículo que "ganhou" um ID novo no meio do caminho conta como uma troca.

    É uma estimativa: veículos que nunca foram detectados reduzem o número de
    IDs e podem mascarar trocas.
    """
    validos = sum(1 for n in frames_por_id.values() if n >= min_frames)
    por_veiculo = validos / veiculos_reais if veiculos_reais else None
    trocas = max(0, validos - veiculos_reais) if veiculos_reais is not None else None
    return {
        "ids_total": len(frames_por_id),
        "ids_validos": validos,
        "fragmentos": len(frames_por_id) - validos,
        "veiculos_reais": veiculos_reais,
        "ids_por_veiculo": por_veiculo,
        "trocas_estimadas": trocas,
    }
