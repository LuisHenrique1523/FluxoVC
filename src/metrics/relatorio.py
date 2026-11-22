"""Montagem do relatório ``docs/resultados.md``: tabelas, gráficos e texto.

As funções ``tabela_*`` devolvem DataFrames (testáveis); :func:`gerar_relatorio`
junta tudo em Markdown e salva os gráficos PNG em ``docs/img/resultados/``.
"""

from __future__ import annotations

import logging
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só gera arquivos
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CLASSES_VEICULO  # noqa: E402
from src.metrics.avaliacao import ResultadoVideo  # noqa: E402
from src.metrics.qualidade import erro_contagem, estabilidade_rastreamento, mae  # noqa: E402
from src.models.contagem import SENTIDOS  # noqa: E402

logger = logging.getLogger(__name__)

ContagemManual = Mapping[tuple[int, str], int]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _formatar(valor: object, casas: int) -> str:
    if valor is None or valor is pd.NA or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    if isinstance(valor, float):
        return f"{valor:.{casas}f}".replace(".", ",")
    return str(valor)


def tabela_markdown(tabela: pd.DataFrame, indice: bool = True, casas: int = 2) -> str:
    """DataFrame → tabela Markdown (números com vírgula decimal, vazios como "—")."""
    if tabela.empty:
        return "_(sem dados)_\n"
    cabecalho = ([str(tabela.index.name or "")] if indice else []) + [str(c) for c in tabela.columns]
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    # formata coluna a coluna: iterrows() converteria inteiros em float nas linhas mistas
    colunas = [[_formatar(v, casas) for v in tabela[c].tolist()] for c in tabela.columns]
    for i, rotulo in enumerate(tabela.index):
        celulas = ([str(rotulo)] if indice else []) + [coluna[i] for coluna in colunas]
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas) + "\n"


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def _manual_por_classe(manual: ContagemManual) -> dict[str, int]:
    return {nome: sum(n for (c, _), n in manual.items() if c == classe) for classe, nome in CLASSES_VEICULO.items()}


def _rotulo(classe: int, sentido: str) -> str:
    return f"{CLASSES_VEICULO[classe]} {sentido}"


def tabela_erro_por_classe(resultado: ResultadoVideo, manual: ContagemManual) -> pd.DataFrame:
    """Erro de contagem por classe (somando os dois sentidos) + linha com o total."""
    real, previsto = _manual_por_classe(manual), resultado.contagem_por_classe()
    real["TOTAL"], previsto["TOTAL"] = sum(real.values()), sum(previsto.values())
    return erro_contagem(real, previsto)


def tabela_erro_por_sentido(resultado: ResultadoVideo, manual: ContagemManual) -> pd.DataFrame:
    """Erro de contagem por classe e sentido (só as combinações que aparecem)."""
    chaves = sorted(set(manual) | set(resultado.contagem))
    real = {_rotulo(c, s): manual.get((c, s), 0) for c, s in chaves}
    previsto = {_rotulo(c, s): resultado.contagem.get((c, s), 0) for c, s in chaves}
    return erro_contagem(real, previsto)


def tabela_mae(resultados: Sequence[ResultadoVideo], manuais: Mapping[str, ContagemManual]) -> pd.DataFrame:
    """MAE por classe entre os vídeos, para cada configuração (+ MAE geral)."""
    linhas = []
    for config in dict.fromkeys(r.configuracao for r in resultados):
        erros: dict[str, list[int]] = {nome: [] for nome in CLASSES_VEICULO.values()}
        for r in resultados:
            if r.configuracao != config or r.video not in manuais:
                continue
            tabela = tabela_erro_por_classe(r, manuais[r.video]).drop(index="TOTAL")
            for nome, erro in tabela["erro_abs"].items():
                erros[nome].append(int(erro))
        if not any(erros.values()):
            continue
        linha: dict[str, object] = {"configuração": config}
        linha.update({nome: (sum(v) / len(v) if v else None) for nome, v in erros.items()})
        todos = [e for v in erros.values() for e in v]
        linha["MAE geral"] = sum(todos) / len(todos)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("configuração") if linhas else pd.DataFrame()


def tabela_estabilidade(
    resultados: Sequence[ResultadoVideo], manuais: Mapping[str, ContagemManual], min_frames: int = 5
) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        reais = sum(manuais[r.video].values()) if r.video in manuais else None
        est = estabilidade_rastreamento(r.frames_por_id, reais, min_frames)
        linhas.append({"vídeo": r.video, "configuração": r.configuracao, **est})
    if not linhas:
        return pd.DataFrame()
    tabela = pd.DataFrame(linhas).set_index("vídeo")
    for coluna in ("veiculos_reais", "trocas_estimadas"):  # inteiros que podem faltar
        tabela[coluna] = tabela[coluna].astype("Int64")
    return tabela


def tabela_desempenho(resultados: Sequence[ResultadoVideo]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        d = r.resumo_desempenho()
        linhas.append({
            "vídeo": r.video, "configuração": r.configuracao, "frames": r.frames,
            "FPS": d.get("fps"), "× tempo real": d.get("tempo_real"),
            "pré-proc. (ms)": d.get("preprocessamento_ms"), "inferência (ms)": d.get("inferencia_ms"),
            "interpretação (ms)": d.get("interpretacao_ms"), "desenho (ms)": d.get("desenho_ms"),
            "total (ms)": d.get("total_ms"), "total p95 (ms)": d.get("total_p95_ms"),
            "memória pico (MB)": d.get("memoria_pico_mb"),
        })
    return pd.DataFrame(linhas).set_index("vídeo") if linhas else pd.DataFrame()


def tabela_contagem_prevista(resultados: Sequence[ResultadoVideo]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        linha: dict[str, object] = {"vídeo": r.video, "configuração": r.configuracao}
        for classe, nome in CLASSES_VEICULO.items():
            linha[nome] = " / ".join(str(r.contagem.get((classe, s), 0)) for s in SENTIDOS)
        linha["total"] = sum(r.contagem.values())
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("vídeo") if linhas else pd.DataFrame()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_contagem(resultado: ResultadoVideo, manual: ContagemManual, caminho: Path) -> Path:
    """Barras: contagem manual × prevista por classe."""
    tabela = tabela_erro_por_classe(resultado, manual).drop(index="TOTAL")
    fig, eixo = plt.subplots(figsize=(6, 3.2), dpi=120)
    posicoes = range(len(tabela))
    eixo.bar([p - 0.2 for p in posicoes], tabela["real"], 0.4, label="manual", color="#4C78A8")
    eixo.bar([p + 0.2 for p in posicoes], tabela["previsto"], 0.4, label="FluxoVC", color="#F58518")
    eixo.set_xticks(list(posicoes), tabela.index)
    eixo.set_ylabel("veículos")
    eixo.set_title(f"{resultado.video} — {resultado.configuracao}")
    eixo.legend()
    eixo.grid(axis="y", alpha=0.3)
    return _salvar(fig, caminho)


def grafico_desempenho(resultados: Sequence[ResultadoVideo], caminho: Path) -> Path:
    """Barras empilhadas: tempo médio por etapa (ms) em cada configuração."""
    tabela = tabela_desempenho(resultados).reset_index()
    rotulos = [f"{v}\n{c}" for v, c in zip(tabela["vídeo"], tabela["configuração"])]
    etapas = ["pré-proc. (ms)", "inferência (ms)", "interpretação (ms)", "desenho (ms)"]
    cores = ["#72B7B2", "#E45756", "#B279A2", "#54A24B"]
    fig, eixo = plt.subplots(figsize=(max(6, 1.6 * len(tabela)), 3.6), dpi=120)
    base = [0.0] * len(tabela)
    for etapa, cor in zip(etapas, cores):
        valores = tabela[etapa].fillna(0).tolist()
        eixo.bar(rotulos, valores, bottom=base, label=etapa.replace(" (ms)", ""), color=cor)
        base = [b + v for b, v in zip(base, valores)]
    eixo.set_ylabel("ms por frame processado")
    eixo.set_title("Latência por etapa")
    eixo.legend(fontsize=8)
    eixo.grid(axis="y", alpha=0.3)
    eixo.tick_params(axis="x", labelsize=7)
    return _salvar(fig, caminho)


def _salvar(fig: plt.Figure, caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
@dataclass
class DadosRelatorio:
    resultados: list[ResultadoVideo]
    manuais: dict[str, ContagemManual]
    deteccao: dict[str, pd.DataFrame] = field(default_factory=dict)   # modelo → tabela P/R/F1
    frames_anotados: int = 0
    ambiente: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


def descrever_ambiente() -> dict[str, str]:
    import cv2
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "não (CPU)"
    return {
        "Data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Sistema": platform.platform(),
        "Processador": platform.processor() or platform.machine(),
        "GPU": gpu,
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "OpenCV": cv2.__version__,
    }


def gerar_relatorio(dados: DadosRelatorio, caminho_md: Path) -> Path:
    """Escreve o Markdown e os gráficos (em ``img/resultados/`` ao lado do ``.md``)."""
    pasta_img = caminho_md.parent / "img" / "resultados"
    partes: list[str] = [
        "# Resultados da avaliação\n",
        "> Gerado automaticamente por `python scripts/evaluate.py`. Não edite à mão: rode o script de novo.\n",
        "## Ambiente\n",
        tabela_markdown(pd.DataFrame({"valor": dados.ambiente}).rename_axis("item")),
    ]
    if dados.avisos:
        partes += ["### Avisos\n", "\n".join(f"- {a}" for a in dados.avisos) + "\n"]

    videos = list(dict.fromkeys(r.video for r in dados.resultados))
    partes += ["## Casos avaliados\n"]
    if videos:
        partes.append("\n".join(
            f"- `{v}` — contagem manual: {'sim' if v in dados.manuais else '**não** (só contagem prevista)'}"
            for v in videos) + "\n")
    partes.append(f"- Frames anotados para detecção: {dados.frames_anotados}\n")

    # 1. Qualidade da contagem
    partes.append("## 1. Qualidade — contagem de veículos\n")
    partes.append(
        "Contagem prevista por classe (A→B / B→A):\n\n" + tabela_markdown(tabela_contagem_prevista(dados.resultados))
    )
    tabela_m = tabela_mae(dados.resultados, dados.manuais)
    if not tabela_m.empty:
        partes += [
            "\n**Erro absoluto médio (MAE) por classe**, média entre os vídeos com contagem manual "
            "(0 = perfeito):\n\n", tabela_markdown(tabela_m),
        ]
        for r in dados.resultados:
            if r.video not in dados.manuais:
                continue
            manual = dados.manuais[r.video]
            nome_img = f"contagem_{Path(r.video).stem}_{r.configuracao.replace(' · ', '_').replace(' ', '')}.png"
            grafico_contagem(r, manual, pasta_img / nome_img)
            partes += [
                f"\n### `{r.video}` — {r.configuracao}\n",
                tabela_markdown(tabela_erro_por_classe(r, manual)),
                "\nPor sentido:\n\n", tabela_markdown(tabela_erro_por_sentido(r, manual)),
                f"\n![Contagem manual × prevista](img/resultados/{nome_img})\n",
            ]
    else:
        partes.append("\n_Sem contagem manual: crie `data_samples/ground_truth/<video>_contagem.csv` "
                      "(ver docs/metricas.md)._\n")

    # 2. Qualidade da detecção
    partes.append("## 2. Qualidade — detecção (precisão, recall, F1 com IoU ≥ 0,5)\n")
    if dados.deteccao:
        for modelo, tabela in dados.deteccao.items():
            partes += [f"\n**{modelo}** ({dados.frames_anotados} frames):\n\n", tabela_markdown(tabela, casas=3)]
    else:
        partes.append("_Sem frames anotados em `data_samples/ground_truth/deteccao/` (ver docs/metricas.md)._\n")

    # 3. Estabilidade
    partes += [
        "## 3. Qualidade — estabilidade do rastreamento\n",
        "IDs válidos = IDs vistos em ≥ 5 frames processados. Ideal: 1,0 ID por veículo e 0 trocas. "
        "É uma estimativa (sem anotação quadro a quadro).\n\n",
        tabela_markdown(tabela_estabilidade(dados.resultados, dados.manuais)),
    ]

    # 4. Desempenho
    partes += ["## 4. Desempenho\n", tabela_markdown(tabela_desempenho(dados.resultados), casas=1)]
    if dados.resultados:
        grafico_desempenho(dados.resultados, pasta_img / "desempenho.png")
        partes.append("\n![Latência por etapa](img/resultados/desempenho.png)\n")
    partes.append(
        "\n- **FPS** conta todos os frames do vídeo (inclusive os pulados pelo frame skip) por segundo de "
        "processamento; **× tempo real** ≥ 1 significa que a análise acompanha o vídeo ao vivo.\n"
        "- Os tempos por etapa são médias por **frame processado**; *p95* = 95% dos frames ficaram abaixo.\n"
    )

    caminho_md.parent.mkdir(parents=True, exist_ok=True)
    caminho_md.write_text("\n".join(partes), encoding="utf-8")
    logger.info("Relatório salvo em %s", caminho_md)
    return caminho_md
