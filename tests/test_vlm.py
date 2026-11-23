"""Testes da Fase 8 — VLM opcional (anonimização, casos de teste, backends, interface).

Nenhum teste baixa modelo: os backends são substituídos por falsos. O teste com
o SmolVLM real só roda se o modelo já estiver em ``pesos/hf/``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.acquisition import fontes
from src.aplicacao import sessao as modulo_sessao
from src.config import PROJECT_ROOT
from src.vlm import backends
from src.vlm.anonimizacao import anonimizar, reduzir, regiao_placa
from src.vlm.backends import AVISO_IA, ConfigVLM, ErroVLM, RespostaVLM, VLMClaude, criar_vlm
from src.vlm.casos import CASOS, Verdade, classificar, extrair_numero, extrair_sim_nao
from src.vlm.servico import PERGUNTA_DESCRICAO_EN, PERGUNTA_DESCRICAO_PT, descrever, preparar_frame
from tests.falsos import DetectorPorCor


class VLMFalso:
    """Responde sempre o mesmo texto e registra o que recebeu."""

    def __init__(self, texto: str = "2", backend: str = "local") -> None:
        self.texto, self.backend, self.nome = texto, backend, "vlm-falso"
        self.recebidas: list[tuple[tuple[int, ...], str]] = []

    def responder(self, imagem_bgr: np.ndarray, pergunta: str) -> RespostaVLM:
        self.recebidas.append((imagem_bgr.shape, pergunta))
        return RespostaVLM(self.texto, 0.01, self.nome, self.backend)


def _textura(altura: int, largura: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (altura, largura, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Anonimização
# ---------------------------------------------------------------------------
def test_reduz_para_o_lado_maximo_e_nao_altera_o_original() -> None:
    original = _textura(720, 1280)
    copia = original.copy()
    saida = anonimizar(original, [], [])
    assert max(saida.shape[:2]) == 768 and saida.shape[1] / saida.shape[0] == pytest.approx(1280 / 720, abs=0.01)
    assert np.array_equal(original, copia)
    pequena, escala = reduzir(_textura(100, 200))
    assert escala == 1.0 and pequena.shape == (100, 200, 3)


def test_pixeliza_pessoas_e_placas_e_preserva_o_resto() -> None:
    imagem = _textura(400, 600)  # menor que 768: sem redução, coordenadas iguais
    pessoa = (50, 50, 110, 200)
    veiculo = (300, 100, 500, 300)
    saida = anonimizar(imagem, [pessoa], [veiculo])

    def variacao(img: np.ndarray, caixa) -> float:
        x1, y1, x2, y2 = (int(v) for v in caixa)
        return float(np.abs(np.diff(img[y1:y2, x1:x2].astype(int), axis=1)).mean())

    # textura aleatória tem variação alta; pixelizada vira blocos (variação quase nula)
    assert variacao(saida, pessoa) < 0.3 * variacao(imagem, pessoa)
    placa = regiao_placa(veiculo)
    assert variacao(saida, placa) < 0.3 * variacao(imagem, placa)
    # topo do veículo e fundo continuam iguais
    assert np.array_equal(saida[100:180, 300:500], imagem[100:180, 300:500])
    assert np.array_equal(saida[300:400, 0:250], imagem[300:400, 0:250])


def test_regiao_da_placa_fica_embaixo_e_no_centro() -> None:
    x1, y1, x2, y2 = regiao_placa((100, 100, 200, 300))
    assert (x1, x2) == pytest.approx((115, 185)) and (y1, y2) == pytest.approx((220, 300))


def test_preparar_frame_usa_as_pessoas_do_detector() -> None:
    detector = SimpleNamespace(detectar_pessoas=lambda img: [(10, 10, 40, 80)])
    frame = preparar_frame(_textura(200, 300), [(100, 50, 200, 150)], detector)
    assert frame.pessoas == 1 and frame.veiculos == 1 and frame.pessoas_verificadas
    sem = preparar_frame(_textura(200, 300), [], DetectorPorCor())
    assert sem.pessoas == 0 and not sem.pessoas_verificadas


# ---------------------------------------------------------------------------
# Casos de teste e classificação
# ---------------------------------------------------------------------------
VERDADE = Verdade.de_deteccoes(
    [(2, (0, 0, 10, 10)), (2, (20, 0, 30, 10)), (7, (500, 0, 600, 10)), (0, (0, 0, 5, 5))], largura=640
)


def test_verdade_ignora_pessoas_e_separa_os_lados() -> None:
    assert VERDADE.por_classe == {2: 2, 3: 0, 5: 0, 7: 1}
    assert VERDADE.total == 3 and VERDADE.presentes == {2, 7}
    assert (VERDADE.esquerda, VERDADE.direita) == (2, 1)


def test_dez_casos_com_ids_unicos() -> None:
    assert len(CASOS) == 10 and len({c.id_ for c in CASOS}) == 10
    assert all(c.pergunta_pt and c.pergunta_en for c in CASOS)


@pytest.mark.parametrize(("texto", "esperado"), [
    ("3", 3), ("There are 3 cars.", 3), ("two", 2), ("Nenhum", 0), ("No buses.", 0), ("vários", None),
])
def test_extrair_numero(texto: str, esperado: int | None) -> None:
    assert extrair_numero(texto) == esperado


@pytest.mark.parametrize(("texto", "esperado"), [
    ("Sim.", True), ("Yes, there is a truck.", True), ("Não", False), ("No.", False),
    ("There are no buses.", False), ("talvez", None),
])
def test_extrair_sim_nao(texto: str, esperado: bool | None) -> None:
    assert extrair_sim_nao(texto) is esperado


CASO = {c.id_: c for c in CASOS}


@pytest.mark.parametrize(("caso", "resposta", "categoria"), [
    ("C01", "3", "correta"),
    ("C01", "5 vehicles", "incorreta"),
    ("C01", "Many vehicles", "incompleta"),
    ("C02", "2 cars", "correta"),
    ("C03", "1", "alucinada"),        # não há ônibus
    ("C03", "0", "correta"),
    ("C05", "Yes", "correta"),        # há caminhão
    ("C06", "Yes, a bus.", "alucinada"),
    ("C06", "No", "correta"),
    ("C07", "Yes", "correta"),
    ("C08", "Left half", "correta"),
    ("C08", "right", "incorreta"),
    ("C08", "both", "incompleta"),
    ("C09", "cars and a truck", "correta"),
    ("C09", "cars", "incompleta"),
    ("C09", "cars, a truck and a bus", "alucinada"),
    ("C10", "Cars", "correta"),
    ("C10", "Trucks", "incorreta"),
    ("C10", "Motorcycles", "alucinada"),
])
def test_classificacao(caso: str, resposta: str, categoria: str) -> None:
    assert classificar(CASO[caso], resposta, VERDADE) == categoria


def test_esperado_legivel() -> None:
    assert CASO["C01"].esperado(VERDADE) == "3"
    assert CASO["C06"].esperado(VERDADE) == "não"
    assert CASO["C08"].esperado(VERDADE) == "esquerda"
    assert CASO["C09"].esperado(VERDADE) == "carro, caminhão"
    assert CASO["C10"].esperado(VERDADE) == "carro"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def test_config_do_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXOVC_VLM_BACKEND", "Claude")
    monkeypatch.setenv("FLUXOVC_VLM_MODELO_CLAUDE", "")
    cfg = ConfigVLM.do_ambiente(carregar_env=False)
    assert cfg.backend == "claude" and cfg.modelo_claude == "claude-opus-5-5"


def test_backend_invalido() -> None:
    with pytest.raises(ErroVLM, match="inválido"):
        criar_vlm(ConfigVLM(backend="outro"))


def test_claude_sem_chave_orienta_o_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ErroVLM, match=r"\.env"):
        VLMClaude()


class ClienteFalso:
    def __init__(self, resposta) -> None:
        self.chamadas: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._criar))
        self._resposta = resposta

    def _criar(self, **kwargs):
        self.chamadas.append(kwargs)
        return self._resposta


def test_claude_monta_a_requisicao_com_imagem_e_le_o_texto() -> None:
    resposta = SimpleNamespace(
        stop_reason="end_turn", model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Três carros.")],
    )
    cliente = ClienteFalso(resposta)
    vlm = VLMClaude(cliente=cliente)
    saida = vlm.responder(_textura(60, 80), "Quantos carros?")
    assert saida.texto == "Três carros." and saida.backend == "claude"
    req = cliente.chamadas[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]
    imagem, texto = req["messages"][0]["content"]
    assert imagem["source"]["media_type"] == "image/png"
    decodificada = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(imagem["source"]["data"]), np.uint8), 1)
    assert decodificada.shape == (60, 80, 3) and texto["text"] == "Quantos carros?"


def test_claude_recusa_vira_mensagem() -> None:
    cliente = ClienteFalso(SimpleNamespace(stop_reason="refusal", model="claude-opus-5-5", content=[]))
    assert "recusou" in VLMClaude(cliente=cliente).responder(_textura(10, 10), "?").texto


def test_descrever_usa_pergunta_padrao_no_idioma_do_backend() -> None:
    frame = preparar_frame(_textura(100, 100), [], None)
    local, claude = VLMFalso(backend="local"), VLMFalso(backend="claude")
    descrever(local, frame)
    descrever(claude, frame, "   ")
    descrever(claude, frame, "Quantas motos?")
    assert local.recebidas[0][1] == PERGUNTA_DESCRICAO_EN
    assert claude.recebidas[0][1] == PERGUNTA_DESCRICAO_PT
    assert claude.recebidas[1][1] == "Quantas motos?"


# ---------------------------------------------------------------------------
# Script de avaliação e interface
# ---------------------------------------------------------------------------
def test_avaliar_vlm_na_cena_sintetica(tmp_path: Path) -> None:
    from scripts.avaliar_vlm import main

    saida = tmp_path / "vlm.md"
    vlm = VLMFalso("2")
    assert main(["--cena-sintetica", "--frames", "2", "--saida", str(saida)], fabrica_vlm=lambda cfg: vlm) == 0
    texto = saida.read_text(encoding="utf-8")
    assert len(vlm.recebidas) == 20 and all(max(s[:2]) <= 768 for s, _ in vlm.recebidas)
    assert "| correta |" in texto and "## Acerto por caso" in texto
    assert texto.count("| cena sintética f") == 20


def test_app_descreve_cena_com_aviso(monkeypatch: pytest.MonkeyPatch, fabrica_video, tmp_path: Path) -> None:
    monkeypatch.setenv("FLUXOVC_VLM_BACKEND", "local")
    video = fabrica_video(nome="demo.avi", n_frames=10)
    monkeypatch.setattr(fontes, "listar_videos_demo", lambda *a, **k: [video])
    monkeypatch.setattr(modulo_sessao, "PASTA_VIDEOS", tmp_path / "videos")
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.session_state["detectores"] = {"yolo11n.pt": DetectorPorCor()}
    at.session_state["vlm"] = VLMFalso("Two vehicles on a gray road.")
    at.run()
    at.button(key="analisar").click().run()
    at.button(key="vlm_descrever").click().run()
    assert not at.exception
    assert any(w.value == AVISO_IA for w in at.warning)
    assert any("Two vehicles" in m.value for m in at.markdown)
    assert at.session_state["vlm_resultado"]["frame"].veiculos == len(at.session_state["sessao"].ultimo_passo.resultado.veiculos) >= 1


@pytest.mark.skipif(
    not (backends.PASTA_MODELOS_HF / "models--HuggingFaceTB--SmolVLM-256M-Instruct").exists(),
    reason="SmolVLM-256M não está em pesos/hf/ (não baixamos modelos nos testes).",
)
def test_smolvlm_real_responde() -> None:  # pragma: no cover - depende do modelo baixado
    vlm = backends.VLMLocal(max_tokens=20)
    resposta = vlm.responder(_textura(240, 320), "What colors do you see? Answer briefly.")
    assert resposta.texto and resposta.latencia_s > 0


VAZIA = Verdade.de_deteccoes([(0, (0, 0, 50, 100))], largura=640)  # só uma pessoa


@pytest.mark.parametrize(("caso", "resposta", "categoria"), [
    ("C08", "The left half contains more vehicles than the right half.", "alucinada"),
    ("C08", "Right side has more than the left.", "alucinada"),
    ("C09", "There are no vehicles in the image.", "correta"),
    ("C10", "The image does not provide information about vehicles.", "correta"),
    ("C09", "Motorcycle.", "alucinada"),
    ("C01", "10", "alucinada"),
])
def test_classificacao_em_cena_sem_veiculos(caso: str, resposta: str, categoria: str) -> None:
    assert classificar(CASO[caso], resposta, VAZIA) == categoria


def test_lado_citado_primeiro_vale() -> None:
    assert classificar(CASO["C08"], "The left half has more than the right half", VERDADE) == "correta"
    assert classificar(CASO["C08"], "More on the right than on the left", VERDADE) == "incorreta"
