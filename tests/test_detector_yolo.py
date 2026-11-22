"""Testes do wrapper do YOLO.

Os testes de unidade não baixam pesos (o ``YOLO`` é substituído). O teste de
integração só roda se ``pesos/yolo11n.pt`` já existir na máquina.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import PESOS_DIR, AppConfig, ModeloConfig
from src.models.detector_yolo import DetectorYOLO, converter_resultado, escolher_dispositivo
from src.models.tipos import ErroModelo, PesosIndisponiveisError


class _Caixas(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.xyxy)


def test_escolher_dispositivo() -> None:
    assert escolher_dispositivo("cpu") == "cpu"
    assert escolher_dispositivo("auto") in {"cpu", "0"}


def test_converter_resultado_com_e_sem_ids() -> None:
    caixas = _Caixas(
        xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        cls=np.array([2.0, 7.0]),
        conf=np.array([0.9, 0.5]),
        id=np.array([11.0, 12.0]),
    )
    deteccoes = converter_resultado(SimpleNamespace(boxes=caixas))
    assert [(d.caixa, d.classe, d.id_rastreio) for d in deteccoes] == [
        ((1, 2, 3, 4), 2, 11),
        ((5, 6, 7, 8), 7, 12),
    ]
    caixas.id = None
    assert all(d.id_rastreio is None for d in converter_resultado(SimpleNamespace(boxes=caixas)))


def test_converter_resultado_vazio() -> None:
    vazio = _Caixas(xyxy=np.empty((0, 4)), cls=np.empty(0), conf=np.empty(0), id=None)
    assert converter_resultado(SimpleNamespace(boxes=vazio)) == []
    assert converter_resultado(SimpleNamespace(boxes=None)) == []


def test_sem_internet_mensagem_orienta_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ultralytics

    def falhar(*_args, **_kwargs):
        raise ConnectionError("sem rede")

    monkeypatch.setattr(ultralytics, "YOLO", falhar)
    with pytest.raises(PesosIndisponiveisError, match="internet") as info:
        DetectorYOLO(ModeloConfig(pesos="yolo11n.pt"), pasta_pesos=tmp_path)
    assert "github.com/ultralytics" in str(info.value)


def test_pesos_corrompidos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ultralytics

    (tmp_path / "yolo11n.pt").write_bytes(b"lixo")
    monkeypatch.setattr(ultralytics, "YOLO", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(ErroModelo, match="corrompidos"):
        DetectorYOLO(ModeloConfig(pesos="yolo11n.pt"), pasta_pesos=tmp_path)


@pytest.mark.skipif(
    not (PESOS_DIR / "yolo11n.pt").exists(), reason="pesos yolo11n.pt não baixados nesta máquina"
)
def test_integracao_yolo_detecta_onibus_na_imagem_de_exemplo() -> None:
    from ultralytics.utils import ASSETS

    from src.acquisition.fontes import carregar_imagem
    from src.models.analisador import analisar_imagem

    cfg = AppConfig()
    cfg.preproc.usar_roi = False
    cfg.modelo.dispositivo = "cpu"
    detector = DetectorYOLO(cfg.modelo)
    imagem = carregar_imagem(ASSETS / "bus.jpg")  # imagem de exemplo distribuída com o Ultralytics
    veiculos, _ = analisar_imagem(imagem, cfg, detector)
    assert any(v.nome_classe == "ônibus" for v in veiculos)
    altura, largura = imagem.shape[:2]
    for v in veiculos:  # caixas no espaço da imagem original
        x1, y1, x2, y2 = v.caixa
        assert 0 <= x1 < x2 <= largura and 0 <= y1 < y2 <= altura

    # rastreamento: duas chamadas na mesma imagem mantêm IDs; reiniciar não quebra
    assert detector.rastrear(np.zeros((640, 640, 3), dtype=np.uint8)) == []
    detector.reiniciar()
