"""Testes da Fase 6 — métricas com valores conhecidos, anotações e configuração em JSON."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.config import AppConfig, carregar_config, config_de_dict, salvar_config
from src.metrics.anotacoes import (
    carregar_contagem_manual,
    classe_coco,
    ler_nomes_classes,
    ler_rotulos_yolo,
    listar_frames_anotados,
    normalizar_sentido,
)
from src.metrics.desempenho import MonitorMemoria, resumo_tempos
from src.metrics.qualidade import (
    Acertos,
    casar_deteccoes,
    erro_contagem,
    estabilidade_rastreamento,
    iou,
    mae,
    tabela_prf,
)
from src.metrics.relatorio import tabela_markdown
from src.models.contagem import SENTIDO_AB, SENTIDO_BA


# ---------------------------------------------------------------------------
# IoU e casamento de detecções
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "esperado"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),
        ((0, 0, 10, 10), (5, 0, 15, 10), 50 / 150),
        ((0, 0, 10, 10), (2, 2, 8, 8), 36 / 100),     # uma dentro da outra
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),       # só encostam
        ((0, 0, 0, 0), (0, 0, 0, 0), 0.0),            # caixas degeneradas
    ],
)
def test_iou(a, b, esperado) -> None:
    assert iou(a, b) == pytest.approx(esperado)


def test_casamento_um_para_um() -> None:
    reais = [((0, 0, 10, 10), 2)]
    previstas = [((0, 0, 10, 10), 2), ((1, 0, 11, 10), 2)]  # duas previsões para o mesmo carro
    r = casar_deteccoes(previstas, reais)
    assert (r[2].vp, r[2].fp, r[2].fn) == (1, 1, 0)


def test_casamento_escolhe_maior_iou() -> None:
    reais = [((0, 0, 10, 10), 2), ((8, 0, 18, 10), 2)]
    previstas = [((7, 0, 17, 10), 2), ((0, 0, 10, 10), 2)]
    r = casar_deteccoes(previstas, reais)
    assert (r[2].vp, r[2].fp, r[2].fn) == (2, 0, 0)


def test_casamento_classe_errada_e_iou_baixo() -> None:
    reais = [((0, 0, 10, 10), 2), ((50, 50, 60, 60), 7)]
    previstas = [((0, 0, 10, 10), 7), ((55, 55, 65, 65), 7)]  # classe trocada; IoU 25/175 < 0,5
    r = casar_deteccoes(previstas, reais)
    assert (r[2].vp, r[2].fp, r[2].fn) == (0, 0, 1)
    assert (r[7].vp, r[7].fp, r[7].fn) == (0, 2, 1)
    assert casar_deteccoes(previstas, reais, limiar_iou=0.1)[7].vp == 1


def test_precisao_recall_f1() -> None:
    a = Acertos(vp=8, fp=2, fn=4)
    assert a.precisao == pytest.approx(0.8)
    assert a.recall == pytest.approx(8 / 12)
    assert a.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert Acertos(0, 0, 3).precisao is None and Acertos(0, 0, 3).recall == 0
    assert Acertos(0, 2, 0).recall is None and Acertos(0, 2, 0).f1 is None
    assert Acertos(0, 1, 1).f1 == 0.0


def test_tabela_prf_com_micro_media() -> None:
    tabela = tabela_prf({"carro": Acertos(8, 2, 4), "moto": Acertos(2, 0, 0)})
    assert tabela.loc["geral", "VP"] == 10 and tabela.loc["geral", "FP"] == 2 and tabela.loc["geral", "FN"] == 4
    assert tabela.loc["geral", "precisao"] == pytest.approx(10 / 12)
    assert tabela.loc["moto", "F1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Contagem e rastreamento
# ---------------------------------------------------------------------------
def test_erro_de_contagem_e_mae() -> None:
    tabela = erro_contagem({"carro": 10, "moto": 0}, {"carro": 8, "moto": 1, "ônibus": 2})
    assert tabela["erro_abs"].tolist() == [2, 1, 2]
    assert tabela.loc["carro", "erro_pct"] == pytest.approx(20.0)
    assert pd.isna(tabela.loc["moto", "erro_pct"])        # real = 0: percentual indefinido
    assert tabela.loc["ônibus", "real"] == 0
    assert mae(tabela) == pytest.approx(5 / 3)
    assert mae(erro_contagem({}, {})) == 0.0


def test_estabilidade_do_rastreamento() -> None:
    est = estabilidade_rastreamento(Counter({1: 30, 2: 3, 3: 12, 4: 8}), veiculos_reais=2)
    assert est["ids_total"] == 4 and est["ids_validos"] == 3 and est["fragmentos"] == 1
    assert est["ids_por_veiculo"] == pytest.approx(1.5) and est["trocas_estimadas"] == 1
    sem_gt = estabilidade_rastreamento({1: 10}, veiculos_reais=None)
    assert sem_gt["ids_por_veiculo"] is None and sem_gt["trocas_estimadas"] is None
    assert estabilidade_rastreamento({1: 10}, veiculos_reais=3)["trocas_estimadas"] == 0


# ---------------------------------------------------------------------------
# Anotações
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("nome", "esperado"),
    [("carro", 2), ("Car", 2), ("ônibus", 5), ("onibus", 5), ("BUS", 5), ("caminhão", 7),
     ("truck", 7), ("moto", 3), ("motorcycle", 3), ("pessoa", None), ("bicicleta", None)],
)
def test_nomes_de_classe(nome: str, esperado: int | None) -> None:
    assert classe_coco(nome) == esperado


@pytest.mark.parametrize("texto", ["A→B", "A->B", "a > b", "AB", " ab "])
def test_sentido_ab(texto: str) -> None:
    assert normalizar_sentido(texto) == SENTIDO_AB


def test_sentido_ba_e_invalido() -> None:
    assert normalizar_sentido("B->A") == SENTIDO_BA
    with pytest.raises(ValueError, match="Sentido"):
        normalizar_sentido("norte")


def test_contagem_manual(tmp_path: Path) -> None:
    csv = tmp_path / "rua_contagem.csv"
    csv.write_text(
        "Classe,Sentido,Quantidade\ncarro,A->B,12\nCarro,A→B,1\nônibus,B->A,2\ntruck,AB,4\n",
        encoding="utf-8-sig",
    )
    assert carregar_contagem_manual(csv) == {(2, SENTIDO_AB): 13, (5, SENTIDO_BA): 2, (7, SENTIDO_AB): 4}
    csv.write_text("classe;sentido;quantidade\nmoto;B→A;3\n", encoding="utf-8")
    assert carregar_contagem_manual(csv) == {(3, SENTIDO_BA): 3}


def test_contagem_manual_invalida(tmp_path: Path) -> None:
    csv = tmp_path / "c.csv"
    csv.write_text("classe;quantidade\ncarro;3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltam"):
        carregar_contagem_manual(csv)
    csv.write_text("classe;sentido;quantidade\nbicicleta;A->B;3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bicicleta"):
        carregar_contagem_manual(csv)


def test_rotulos_yolo(tmp_path: Path) -> None:
    txt = tmp_path / "f.txt"
    txt.write_text("0 0.5 0.5 0.2 0.4\n2 0.1 0.1 0.1 0.1\n\n1 0.25 0.75 0.5 0.5\n", encoding="utf-8")
    nomes = ["carro", "caminhao", "pessoa"]
    caixas = ler_rotulos_yolo(txt, 100, 50, nomes)
    assert caixas == [((40.0, 15.0, 60.0, 35.0), 2), ((0.0, 25.0, 50.0, 50.0), 7)]  # pessoa ignorada
    assert ler_rotulos_yolo(None, 100, 50, nomes) == []


@pytest.mark.parametrize("conteudo", ["0 0.5 0.5\n", "x 0.5 0.5 0.1 0.1\n", "9 0.5 0.5 0.1 0.1\n"])
def test_rotulos_yolo_invalidos(tmp_path: Path, conteudo: str) -> None:
    txt = tmp_path / "f.txt"
    txt.write_text(conteudo, encoding="utf-8")
    with pytest.raises(ValueError):
        ler_rotulos_yolo(txt, 100, 100, ["carro"])


def test_estrutura_exportada_pelo_label_studio(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("carro\nmoto\n", encoding="utf-8")
    for nome in ("a", "b"):
        (tmp_path / "images" / f"{nome}.jpg").write_bytes(b"x")
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    frames = listar_frames_anotados(tmp_path)
    assert [(f.imagem.name, f.rotulos.name if f.rotulos else None) for f in frames] == [("a.jpg", "a.txt"), ("b.jpg", None)]
    assert ler_nomes_classes(tmp_path) == ["carro", "moto"]


def test_estrutura_exportada_pelo_cvat(tmp_path: Path) -> None:
    (tmp_path / "obj_train_data").mkdir()
    (tmp_path / "obj.names").write_text("car\ntruck\n", encoding="utf-8")
    assert ler_nomes_classes(tmp_path) == ["car", "truck"]
    with pytest.raises(FileNotFoundError):
        ler_nomes_classes(tmp_path / "obj_train_data")


# ---------------------------------------------------------------------------
# Desempenho, relatório e configuração
# ---------------------------------------------------------------------------
def test_resumo_de_tempos() -> None:
    tempos = pd.DataFrame({"inferencia": [10.0, 20.0, 30.0], "desenho": [1.0, 1.0, 1.0]})
    resumo = resumo_tempos(tempos)
    assert resumo["inferencia_ms"] == pytest.approx(20.0)
    assert resumo["total_ms"] == pytest.approx(21.0)
    assert resumo["total_p95_ms"] == pytest.approx(30.0, abs=1.1)
    assert resumo_tempos(pd.DataFrame()) == {}


def test_monitor_de_memoria() -> None:
    monitor = MonitorMemoria()
    assert monitor.inicial_mb > 0
    lixo = bytearray(50 * 1024 * 1024)  # noqa: F841 — aloca 50 MB
    monitor.amostrar()
    assert monitor.pico_mb >= monitor.inicial_mb


def test_tabela_markdown() -> None:
    tabela = pd.DataFrame({"valor": [1.234, None], "n": [3, 4]}, index=pd.Index(["a", "b"], name="item"))
    md = tabela_markdown(tabela)
    assert md.splitlines()[0] == "| item | valor | n |"
    assert "| a | 1,23 | 3 |" in md and "| b | — | 4 |" in md
    assert "sem dados" in tabela_markdown(pd.DataFrame())


def test_configuracao_ida_e_volta_em_json(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.1, 0.45), (0.9, 0.55)
    cfg.geometria.roi = [(0.0, 0.3), (1.0, 0.3), (1.0, 1.0)]
    cfg.modelo.confianca = 0.42
    cfg.preproc.usar_clahe = True
    cfg.visualizacao.colormap = "JET"
    salvar_config(cfg, tmp_path / "rua.json")
    assert carregar_config(tmp_path / "rua.json") == cfg


def test_configuracao_parcial_e_invalida(tmp_path: Path) -> None:
    cfg = config_de_dict({"modelo": {"confianca": 0.6}, "desconhecido": {"x": 1}})
    assert cfg.modelo.confianca == 0.6 and cfg.modelo.iou == AppConfig().modelo.iou
    (tmp_path / "ruim.json").write_text("{ não é json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        carregar_config(tmp_path / "ruim.json")
    with pytest.raises(ValueError, match="modelo.confianca"):
        config_de_dict({"modelo": {"confianca": "alta"}})
    assert not math.isnan(cfg.modelo.confianca)
