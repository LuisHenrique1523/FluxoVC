"""Leitura das anotações de referência (ground truth).

* **Contagem manual** (``ground_truth/<video>_contagem.csv``)::

      classe;sentido;quantidade
      carro;A→B;12
      carro;B→A;9
      moto;A→B;3

  Aceita ``,`` ou ``;``, ``A->B``/``A→B``/``AB`` e nomes com ou sem acento
  (``onibus``, ``ônibus``, ``bus``…).

* **Frames anotados** (formato YOLO, exportado pelo CVAT ou Label Studio): uma
  imagem e um ``.txt`` com o mesmo nome, uma linha por objeto::

      <indice_classe> <centro_x> <centro_y> <largura> <altura>   # normalizados 0–1

  e um ``classes.txt`` (Label Studio) ou ``obj.names`` (CVAT) com os nomes das
  classes na ordem dos índices.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.contagem import SENTIDO_AB, SENTIDO_BA

logger = logging.getLogger(__name__)

ALIASES_CLASSE: dict[str, int] = {
    "carro": 2, "car": 2, "automovel": 2, "auto": 2,
    "moto": 3, "motorcycle": 3, "motocicleta": 3, "motorbike": 3,
    "onibus": 5, "bus": 5,
    "caminhao": 7, "truck": 7,
}
ARQUIVOS_CLASSES: tuple[str, ...] = ("classes.txt", "obj.names")
EXTENSOES_IMAGEM: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return sem_acento.strip().lower()


def classe_coco(nome: str) -> int | None:
    """Converte o nome da classe da anotação no id COCO (``None`` se não for veículo)."""
    return ALIASES_CLASSE.get(_normalizar(nome))


def normalizar_sentido(texto: str) -> str:
    """``"A->B"``, ``"A→B"``, ``"ab"`` → ``"A→B"`` (idem para B→A).

    Raises:
        ValueError: sentido desconhecido.
    """
    limpo = _normalizar(texto).replace("->", "").replace(">", "").replace(" ", "")
    if limpo == "ab":
        return SENTIDO_AB
    if limpo == "ba":
        return SENTIDO_BA
    raise ValueError(f"Sentido '{texto}' inválido. Use A→B ou B→A (também aceita A->B).")


def carregar_contagem_manual(caminho: Path) -> dict[tuple[int, str], int]:
    """Lê a contagem manual: ``{(classe_coco, sentido): quantidade}``.

    Raises:
        ValueError: colunas ausentes, classe ou sentido inválidos.
    """
    tabela = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    tabela.columns = [_normalizar(c) for c in tabela.columns]
    faltando = {"classe", "sentido", "quantidade"} - set(tabela.columns)
    if faltando:
        raise ValueError(f"'{caminho.name}': faltam as colunas {sorted(faltando)}.")
    contagem: dict[tuple[int, str], int] = {}
    for numero, linha in enumerate(tabela.itertuples(index=False), start=2):
        classe = classe_coco(linha.classe)
        if classe is None:
            raise ValueError(f"'{caminho.name}', linha {numero}: classe '{linha.classe}' desconhecida.")
        chave = (classe, normalizar_sentido(linha.sentido))
        contagem[chave] = contagem.get(chave, 0) + int(linha.quantidade)
    return contagem


# ---------------------------------------------------------------------------
# Frames anotados em formato YOLO
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameAnotado:
    imagem: Path
    rotulos: Path | None    # None = sem arquivo .txt (tratado como "nenhum veículo")


def ler_nomes_classes(pasta: Path) -> list[str]:
    """Nomes das classes, na ordem dos índices (``classes.txt`` ou ``obj.names``).

    Raises:
        FileNotFoundError: nenhum dos arquivos foi encontrado.
    """
    for nome in ARQUIVOS_CLASSES:
        encontrados = sorted(pasta.rglob(nome))
        if encontrados:
            linhas = encontrados[0].read_text(encoding="utf-8-sig").splitlines()
            return [linha.strip() for linha in linhas if linha.strip()]
    raise FileNotFoundError(
        f"Nenhum {' ou '.join(ARQUIVOS_CLASSES)} em '{pasta}'. Exporte as anotações no formato YOLO."
    )


def listar_frames_anotados(pasta: Path) -> list[FrameAnotado]:
    """Imagens da pasta (recursivo) pareadas com o ``.txt`` de mesmo nome."""
    rotulos = {p.stem: p for p in pasta.rglob("*.txt") if p.name not in ARQUIVOS_CLASSES}
    imagens = sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES_IMAGEM)
    frames = [FrameAnotado(img, rotulos.get(img.stem)) for img in imagens]
    sem_rotulo = [f.imagem.name for f in frames if f.rotulos is None]
    if sem_rotulo:
        logger.warning("%d imagens sem .txt (consideradas sem veículos): %s", len(sem_rotulo), sem_rotulo[:5])
    return frames


def ler_rotulos_yolo(
    caminho: Path | None, largura: int, altura: int, nomes: list[str]
) -> list[tuple[tuple[float, float, float, float], int]]:
    """Lê um ``.txt`` YOLO e devolve caixas em pixels ``(x1, y1, x2, y2)`` com a classe COCO.

    Objetos de classes que não são veículo (ex.: pessoa) são ignorados.

    Raises:
        ValueError: linha malformada ou índice de classe inexistente.
    """
    if caminho is None:
        return []
    caixas = []
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        partes = linha.split()
        if not partes:
            continue
        try:
            indice = int(partes[0])
            cx, cy, w, h = (float(v) for v in partes[1:5])
        except (ValueError, IndexError) as erro:
            raise ValueError(f"'{caminho.name}', linha {numero}: formato YOLO inválido.") from erro
        if not 0 <= indice < len(nomes):
            raise ValueError(f"'{caminho.name}', linha {numero}: classe {indice} não está em classes.txt.")
        classe = classe_coco(nomes[indice])
        if classe is None:
            continue
        caixa = ((cx - w / 2) * largura, (cy - h / 2) * altura, (cx + w / 2) * largura, (cy + h / 2) * altura)
        caixas.append((caixa, classe))
    return caixas
