"""Tabelas e arquivos para a interface: eventos, resumo, série temporal e CSV."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import CLASSES_VEICULO
from src.models.analisador import VeiculoRastreado
from src.models.contagem import SENTIDOS, ContadorLinha, EventoCruzamento

COLUNAS_EVENTOS: list[str] = ["id", "classe", "sentido", "frame", "instante_s", "velocidade_kmh"]


def eventos_para_dataframe(eventos: Iterable[EventoCruzamento]) -> pd.DataFrame:
    """Uma linha por veículo contado."""
    linhas = [
        {
            "id": e.id_rastreio,
            "classe": e.nome_classe,
            "sentido": e.sentido,
            "frame": e.frame,
            "instante_s": round(e.timestamp_s, 2),
            "velocidade_kmh": None if e.velocidade_kmh is None else round(e.velocidade_kmh, 1),
        }
        for e in eventos
    ]
    return pd.DataFrame(linhas, columns=COLUNAS_EVENTOS)


def resumo_contagem(contador: ContadorLinha) -> pd.DataFrame:
    """Tabela classe × sentido, com linha de total."""
    tabela = pd.DataFrame.from_dict(contador.por_classe(), orient="index")[[*SENTIDOS, "total"]]
    tabela.loc["TOTAL"] = tabela.sum()
    tabela.index.name = "classe"
    return tabela


def serie_contagem(
    eventos: Iterable[EventoCruzamento], duracao_s: float | None = None, passo_s: float = 1.0
) -> pd.DataFrame:
    """Contagem **acumulada** por classe ao longo do tempo (para o gráfico).

    Returns:
        DataFrame indexado por ``tempo_s`` (0, passo, 2·passo, …) com uma coluna por classe.
    """
    eventos = list(eventos)
    fim = max([duracao_s or 0.0] + [e.timestamp_s for e in eventos])
    tempos = np.arange(0.0, fim + passo_s, passo_s)
    serie = pd.DataFrame(index=pd.Index(np.round(tempos, 3), name="tempo_s"))
    for nome in CLASSES_VEICULO.values():
        instantes = np.sort([e.timestamp_s for e in eventos if e.nome_classe == nome])
        serie[nome] = np.searchsorted(instantes, tempos, side="right")
    return serie


def deteccoes_para_dataframe(veiculos: Iterable[VeiculoRastreado]) -> pd.DataFrame:
    """Tabela de detecções de uma imagem única."""
    linhas = [
        {
            "classe": v.nome_classe,
            "confianca": round(v.confianca, 3),
            "x1": round(v.caixa[0]), "y1": round(v.caixa[1]),
            "x2": round(v.caixa[2]), "y2": round(v.caixa[3]),
        }
        for v in veiculos
    ]
    return pd.DataFrame(linhas, columns=["classe", "confianca", "x1", "y1", "x2", "y2"])


def para_csv(tabela: pd.DataFrame, indice: bool = False) -> bytes:
    """CSV em UTF-8 com BOM, separador ``;`` e vírgula decimal (abre direto no Excel em pt-BR)."""
    return tabela.to_csv(index=indice, sep=";", decimal=",").encode("utf-8-sig")
