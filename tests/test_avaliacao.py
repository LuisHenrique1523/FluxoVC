"""Testes da Fase 6 — avaliação de ponta a ponta (scripts/evaluate.py) com dados sintéticos."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import pytest

from src.acquisition.fontes import extrair_frames
from src.config import PROJECT_ROOT, AppConfig, salvar_config
from src.metrics.anotacoes import listar_frames_anotados
from src.metrics.avaliacao import avaliar_deteccao, avaliar_video
from tests.conftest import desenhar_frame_sintetico
from tests.falsos import DetectorPorCor


def _carregar_script(nome: str):
    caminho = PROJECT_ROOT / "scripts" / f"{nome}.py"
    spec = importlib.util.spec_from_file_location(f"script_{nome}", caminho)
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modulo)
    return modulo


def _cfg_linha_no_meio() -> AppConfig:
    cfg = AppConfig()
    cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (0.0, 0.5), (1.0, 0.5)
    return cfg


def _anotar_frames_sinteticos(pasta: Path, indices: list[int]) -> None:
    """Grava frames sintéticos (PNG) + rótulos YOLO exatos dos dois retângulos."""
    (pasta / "images").mkdir(parents=True)
    (pasta / "labels").mkdir()
    (pasta / "classes.txt").write_text("carro\ncaminhao\npessoa\n", encoding="utf-8")
    largura, altura, lado = 640, 360, 45
    for i in indices:
        cv2.imwrite(str(pasta / "images" / f"f{i:03d}.png"), desenhar_frame_sintetico(i, largura, altura))
        y1, y2 = 9 * i, 315 - 9 * i
        caixas = [(0, 160, y1), (1, 435, y2)]  # (classe, x, y) — retângulo de 46 × 46 px (borda inclusiva)
        linhas = [
            f"{c} {(x + 23) / largura:.6f} {(y + 23) / altura:.6f} {46 / largura:.6f} {46 / altura:.6f}"
            for c, x, y in caixas
        ]
        linhas.append("2 0.05 0.05 0.02 0.05")  # uma "pessoa" anotada: deve ser ignorada
        (pasta / "labels" / f"f{i:03d}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def test_avaliar_video(fabrica_video) -> None:
    caminho = fabrica_video(nome="rua.avi", n_frames=40)
    resultado = avaliar_video(caminho, _cfg_linha_no_meio(), DetectorPorCor())
    assert resultado.frames == 40 and resultado.video == "rua.avi"
    assert resultado.contagem_por_classe() == {"carro": 1, "moto": 0, "ônibus": 0, "caminhão": 1}
    assert set(resultado.frames_por_id) == {1, 2}
    assert resultado.fps_medio and resultado.tempo_real and resultado.memoria_pico_mb > 0
    assert avaliar_video(caminho, _cfg_linha_no_meio(), DetectorPorCor(), max_frames=10).frames == 10


def test_avaliar_deteccao_com_rotulos_exatos(tmp_path: Path) -> None:
    _anotar_frames_sinteticos(tmp_path, [5, 10, 15])
    tabela = avaliar_deteccao(listar_frames_anotados(tmp_path), ["carro", "caminhao", "pessoa"], AppConfig(), DetectorPorCor())
    assert tabela.loc["geral", "VP"] == 6 and tabela.loc["geral", "FP"] == 0 and tabela.loc["geral", "FN"] == 0
    assert tabela.loc["carro", "precisao"] == 1.0 and tabela.loc["caminhão", "recall"] == 1.0


def test_extrair_frames(fabrica_video, tmp_path: Path) -> None:
    video = fabrica_video(nome="Rua Central.avi", n_frames=40)
    salvos = extrair_frames(video, 5, tmp_path / "frames")
    assert [p.name for p in salvos] == [f"Rua_Central_f{i:06d}.jpg" for i in (0, 10, 20, 29, 39)]
    assert len(extrair_frames(video, 100, tmp_path / "todos")) == 40  # não repete frames
    with pytest.raises(ValueError):
        extrair_frames(video, 0, tmp_path)


def test_script_evaluate_de_ponta_a_ponta(fabrica_video, tmp_path: Path) -> None:
    video = fabrica_video(nome="rua.avi", n_frames=40)
    gt = tmp_path / "ground_truth"
    gt.mkdir()
    salvar_config(_cfg_linha_no_meio(), gt / "rua.json")
    # contagem manual com um erro proposital: uma moto que o sistema não vai contar
    (gt / "rua_contagem.csv").write_text(
        "classe;sentido;quantidade\ncarro;A->B;1\ncaminhão;B→A;1\nmoto;A->B;1\n", encoding="utf-8"
    )
    _anotar_frames_sinteticos(gt / "deteccao", [5, 12, 20])
    saida = tmp_path / "docs" / "resultados.md"

    evaluate = _carregar_script("evaluate")
    codigo = evaluate.main(
        ["--videos", str(video), "--modelos", "yolo11n.pt", "--frame-skip", "0", "1",
         "--ground-truth", str(gt), "--saida", str(saida)],
        fabrica=lambda cfg: DetectorPorCor(),
    )
    assert codigo == 0
    texto = saida.read_text(encoding="utf-8")
    # MAE: carro 0, moto 1, ônibus 0, caminhão 0 → 0,25, para as duas configurações
    assert "| yolo11n | 0,00 | 1,00 | 0,00 | 0,00 | 0,25 |" in texto
    assert "| yolo11n · skip 1 | 0,00 | 1,00 | 0,00 | 0,00 | 0,25 |" in texto
    assert "| geral | 6 | 0 | 0 | 1,000 | 1,000 | 1,000 |" in texto       # detecção perfeita
    assert "## 4. Desempenho" in texto and "memória pico (MB)" in texto
    imagens = sorted(p.name for p in (saida.parent / "img" / "resultados").iterdir())
    assert "desempenho.png" in imagens and len(imagens) == 3  # + 1 gráfico de contagem por configuração


def test_script_evaluate_sem_nada_para_avaliar(tmp_path: Path) -> None:
    evaluate = _carregar_script("evaluate")
    codigo = evaluate.main(
        ["--videos", "--ground-truth", str(tmp_path), "--saida", str(tmp_path / "r.md")],
        fabrica=lambda cfg: DetectorPorCor(),
    )
    assert codigo == 1 and not (tmp_path / "r.md").exists()


def test_script_evaluate_sem_contagem_manual_avisa(fabrica_video, tmp_path: Path) -> None:
    video = fabrica_video(nome="sem_gt.avi", n_frames=10)
    saida = tmp_path / "r.md"
    evaluate = _carregar_script("evaluate")
    assert evaluate.main(
        ["--videos", str(video), "--modelos", "yolo11n.pt", "--frame-skip", "0",
         "--ground-truth", str(tmp_path), "--saida", str(saida)],
        fabrica=lambda cfg: DetectorPorCor(),
    ) == 0
    texto = saida.read_text(encoding="utf-8")
    assert "sem `sem_gt.json`" in texto and "Sem contagem manual" in texto


def test_script_extrair_frames(fabrica_video, tmp_path: Path) -> None:
    video = fabrica_video(n_frames=12)
    extrair = _carregar_script("extrair_frames")
    assert extrair.main([str(video), "--quantidade", "3", "--saida", str(tmp_path / "f")]) == 0
    assert len(list((tmp_path / "f").glob("*.jpg"))) == 3
    assert extrair.main([str(tmp_path / "nao_existe.mp4")]) == 1
