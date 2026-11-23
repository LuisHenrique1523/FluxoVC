"""Casos de teste do VLM: perguntas cuja resposta a própria pipeline sabe.

Para cada frame, a "verdade" vem das detecções do YOLO (ou da cena sintética):
quantos veículos de cada classe e de que lado da imagem estão. A resposta do
VLM é classificada automaticamente em:

* **correta**: igual à verdade;
* **incompleta**: certa em parte (ex.: citou só alguns dos tipos presentes) ou
  sem uma resposta clara (nenhum número, nenhum sim/não);
* **incorreta**: resposta clara e errada;
* **alucinada**: afirma a presença de um tipo de veículo que não está na cena.

A classificação automática é uma **primeira passada**: a equipe deve revisar
as respostas em ``docs/vlm_resultados.md`` (coluna "revisão").
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.config import CLASSES_VEICULO

CATEGORIAS: tuple[str, ...] = ("correta", "incompleta", "incorreta", "alucinada")

# nomes que aparecem nas respostas (português e inglês), por classe COCO
SINONIMOS: dict[int, tuple[str, ...]] = {
    2: ("carro", "carros", "automovel", "automoveis", "car", "cars", "sedan", "suv"),
    3: ("moto", "motos", "motocicleta", "motocicletas", "motorcycle", "motorcycles", "motorbike", "scooter"),
    5: ("onibus", "bus", "buses", "micro-onibus"),
    7: ("caminhao", "caminhoes", "truck", "trucks", "lorry"),
}
NUMEROS_POR_EXTENSO: dict[str, int] = {
    "zero": 0, "nenhum": 0, "nenhuma": 0, "none": 0, "no": 0,
    "um": 1, "uma": 1, "one": 1, "a single": 1,
    "dois": 2, "duas": 2, "two": 2, "tres": 3, "three": 3, "quatro": 4, "four": 4,
    "cinco": 5, "five": 5, "seis": 6, "six": 6, "sete": 7, "seven": 7, "oito": 8, "eight": 8,
    "nove": 9, "nine": 9, "dez": 10, "ten": 10,
}


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(sem_acento.lower().split())


@dataclass(frozen=True)
class Verdade:
    """O que a pipeline sabe sobre um frame."""

    por_classe: dict[int, int]       # classe COCO → quantidade
    esquerda: int                    # veículos com o centro na metade esquerda
    direita: int

    @classmethod
    def de_deteccoes(cls, deteccoes: Iterable[tuple[int, Sequence[float]]], largura: int) -> Verdade:
        """``deteccoes``: pares ``(classe, (x1, y1, x2, y2))`` em pixels."""
        contagem: Counter[int] = Counter()
        esquerda = direita = 0
        for classe, caixa in deteccoes:
            if classe not in CLASSES_VEICULO:
                continue
            contagem[classe] += 1
            if (caixa[0] + caixa[2]) / 2 < largura / 2:
                esquerda += 1
            else:
                direita += 1
        return cls({c: contagem.get(c, 0) for c in CLASSES_VEICULO}, esquerda, direita)

    @property
    def total(self) -> int:
        return sum(self.por_classe.values())

    @property
    def presentes(self) -> set[int]:
        return {c for c, n in self.por_classe.items() if n > 0}


@dataclass(frozen=True)
class Caso:
    id_: str
    pergunta_pt: str
    pergunta_en: str                 # para o SmolVLM, treinado quase só em inglês
    tipo: str                        # "numero", "sim_nao", "lado", "tipos", "mais_comum"
    classe: int | None = None

    def esperado(self, v: Verdade) -> str:
        if self.tipo == "numero":
            return str(v.total if self.classe is None else v.por_classe[self.classe])
        if self.tipo == "sim_nao":
            n = v.total if self.classe is None else v.por_classe[self.classe]
            return "sim" if n > 0 else "não"
        if self.tipo == "lado":
            if v.esquerda == v.direita:
                return "igual"
            return "esquerda" if v.esquerda > v.direita else "direita"
        if self.tipo == "tipos":
            return ", ".join(CLASSES_VEICULO[c] for c in sorted(v.presentes)) or "nenhum"
        if self.tipo == "mais_comum":
            return _mais_comum(v) or "nenhum"
        raise ValueError(self.tipo)


CASOS: tuple[Caso, ...] = (
    Caso("C01", "Quantos veículos há na imagem? Responda só com um número.",
         "How many vehicles are in the image? Answer with a number only.", "numero"),
    Caso("C02", "Quantos carros há na imagem? Responda só com um número.",
         "How many cars are in the image? Answer with a number only.", "numero", 2),
    Caso("C03", "Quantos ônibus há na imagem? Responda só com um número.",
         "How many buses are in the image? Answer with a number only.", "numero", 5),
    Caso("C04", "Quantas motos há na imagem? Responda só com um número.",
         "How many motorcycles are in the image? Answer with a number only.", "numero", 3),
    Caso("C05", "Há algum caminhão na imagem? Responda sim ou não.",
         "Is there a truck in the image? Answer yes or no.", "sim_nao", 7),
    Caso("C06", "Há algum ônibus na imagem? Responda sim ou não.",
         "Is there a bus in the image? Answer yes or no.", "sim_nao", 5),
    Caso("C07", "Há algum veículo na via? Responda sim ou não.",
         "Is there any vehicle on the road? Answer yes or no.", "sim_nao"),
    Caso("C08", "Há mais veículos na metade esquerda ou na metade direita da imagem?",
         "Are there more vehicles on the left half or on the right half of the image?", "lado"),
    Caso("C09", "Quais tipos de veículo aparecem (carro, moto, ônibus, caminhão)?",
         "Which types of vehicle appear in the image (car, motorcycle, bus, truck)?", "tipos"),
    Caso("C10", "Qual é o tipo de veículo mais comum na imagem?",
         "What is the most common type of vehicle in the image?", "mais_comum"),
)


def _mais_comum(v: Verdade) -> str | None:
    if v.total == 0:
        return None
    maior = max(v.por_classe.values())
    lideres = [c for c, n in v.por_classe.items() if n == maior]
    return CLASSES_VEICULO[lideres[0]] if len(lideres) == 1 else None


# ---------------------------------------------------------------------------
# Interpretação das respostas
# ---------------------------------------------------------------------------
def classes_citadas(texto: str) -> set[int]:
    t = f" {normalizar(texto)} "
    return {c for c, nomes in SINONIMOS.items() if any(re.search(rf"\b{re.escape(n)}\b", t) for n in nomes)}


def extrair_numero(texto: str) -> int | None:
    t = normalizar(texto)
    digitos = re.search(r"\d+", t)
    if digitos:
        return int(digitos.group())
    for palavra, valor in NUMEROS_POR_EXTENSO.items():
        if re.search(rf"\b{palavra}\b", t):
            return valor
    return None


def extrair_sim_nao(texto: str) -> bool | None:
    t = normalizar(texto)
    if re.match(r"^(nao|no)\b", t) or re.search(r"\b(there (is|are) no|nenhum|nenhuma)\b", t):
        return False
    if re.match(r"^(sim|yes)\b", t) or re.search(r"\b(there (is|are)|ha|existe)\b", t):
        return True
    return None


def lado_afirmado(texto: str) -> str | None:
    """Lado citado primeiro na resposta ("a esquerda tem mais que a direita" → esquerda)."""
    t = normalizar(texto)
    posicoes = {}
    for lado, nomes in (("esquerda", ("esquerda", "left")), ("direita", ("direita", "right"))):
        achados = [m.start() for n in nomes for m in re.finditer(rf"\b{n}\b", t)]
        if achados:
            posicoes[lado] = min(achados)
    return min(posicoes, key=posicoes.get) if posicoes else None


def classificar(caso: Caso, resposta: str, v: Verdade) -> str:
    """Classifica a resposta do VLM em uma das :data:`CATEGORIAS`."""
    if caso.tipo == "numero":
        n = extrair_numero(resposta)
        esperado = v.total if caso.classe is None else v.por_classe[caso.classe]
        if n is None:
            return "incompleta"
        if n == esperado:
            return "correta"
        return "alucinada" if esperado == 0 else "incorreta"  # contou o que não existe

    if caso.tipo == "sim_nao":
        r = extrair_sim_nao(resposta)
        esperado_bool = (v.total if caso.classe is None else v.por_classe[caso.classe]) > 0
        if r is None:
            return "incompleta"
        if r == esperado_bool:
            return "correta"
        return "alucinada" if r else "incorreta"  # "sim" para algo ausente é alucinação

    if caso.tipo == "lado":
        lado = lado_afirmado(resposta)
        if lado is None:
            return "incompleta"
        esperado_lado = caso.esperado(v)
        if esperado_lado == "igual":  # empate ou cena sem veículos: afirmar um lado é erro
            return "alucinada" if v.total == 0 else "incorreta"
        return "correta" if lado == esperado_lado else "incorreta"

    citadas = classes_citadas(resposta)
    if citadas - v.presentes:
        return "alucinada"  # citou um tipo de veículo que não está na cena
    if not citadas:
        # cena sem veículos e resposta sem nenhum tipo ("não há veículos") está certa
        return "correta" if v.total == 0 else "incompleta"

    if caso.tipo == "tipos":
        return "correta" if citadas == v.presentes else "incompleta"

    if caso.tipo == "mais_comum":
        esperado = _mais_comum(v)
        if esperado is None:  # empate: qualquer um dos líderes vale
            return "correta"
        classe_esperada = {nome: c for c, nome in CLASSES_VEICULO.items()}[esperado]
        if citadas == {classe_esperada}:
            return "correta"
        return "incompleta" if classe_esperada in citadas else "incorreta"
    raise ValueError(caso.tipo)
