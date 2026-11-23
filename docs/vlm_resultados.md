# Resultados do VLM

> Gerado por `python scripts/avaliar_vlm.py`. **Não edite à mão**, exceto a coluna *revisão*. A classificação automática é uma primeira passada: confira cada linha.

- **Modelo:** `HuggingFaceTB/SmolVLM-256M-Instruct` (backend `local`)
- **Frames:** 2 de `reais`
- **Data:** 22/11/2026 21:56
- **Verdade:** detecções da pipeline no frame inteiro (sem ROI).
- **Privacidade:** todas as perguntas usaram o frame anonimizado (reduzido, pessoas e placas pixelizadas).

## Resumo

| Categoria | Respostas | % |
|---|---|---|
| correta | 15 | 75 % |
| incompleta | 0 | 0 % |
| incorreta | 1 | 5 % |
| alucinada | 4 | 20 % |
| **total** | **20** | |

**Latência por resposta:** média 3,4 s, máxima 18,0 s.

## Acerto por caso

| Caso | Pergunta | Corretas |
|---|---|---|
| C01 | Quantos veículos há na imagem? Responda só com um número. | 1/2 |
| C02 | Quantos carros há na imagem? Responda só com um número. | 1/2 |
| C03 | Quantos ônibus há na imagem? Responda só com um número. | 2/2 |
| C04 | Quantas motos há na imagem? Responda só com um número. | 2/2 |
| C05 | Há algum caminhão na imagem? Responda sim ou não. | 2/2 |
| C06 | Há algum ônibus na imagem? Responda sim ou não. | 2/2 |
| C07 | Há algum veículo na via? Responda sim ou não. | 2/2 |
| C08 | Há mais veículos na metade esquerda ou na metade direita da imagem? | 0/2 |
| C09 | Quais tipos de veículo aparecem (carro, moto, ônibus, caminhão)? | 1/2 |
| C10 | Qual é o tipo de veículo mais comum na imagem? | 2/2 |

## Respostas

| Frame | Caso | Esperado (pipeline) | Resposta do VLM | Automática | Revisão | Latência (s) |
|---|---|---|---|---|---|---|
| bus.jpg | C01 | 1 | 1. | correta | | 2,6 |
| bus.jpg | C02 | 0 | 1. | alucinada | | 2,6 |
| bus.jpg | C03 | 1 | 1. | correta | | 2,6 |
| bus.jpg | C04 | 0 | 0. | correta | | 2,6 |
| bus.jpg | C05 | não | No. | correta | | 2,5 |
| bus.jpg | C06 | sim | Yes. | correta | | 2,5 |
| bus.jpg | C07 | sim | Yes. | correta | | 2,4 |
| bus.jpg | C08 | direita | The image is taken on the left half of the image. | incorreta | | 3,2 |
| bus.jpg | C09 | ônibus | The image shows a bus in motion, with a person standing inside it. The bus is blue and has a roof with a canopy. The person is wearing a wh… | correta | | 18,0 |
| bus.jpg | C10 | ônibus | There is a blue color bus in the image. | correta | | 3,1 |
| zidane.jpg | C01 | 0 | 10 | alucinada | | 2,5 |
| zidane.jpg | C02 | 0 | 0 | correta | | 2,4 |
| zidane.jpg | C03 | 0 | 0 | correta | | 2,5 |
| zidane.jpg | C04 | 0 | 0 | correta | | 2,5 |
| zidane.jpg | C05 | não | No. | correta | | 2,5 |
| zidane.jpg | C06 | não | No. | correta | | 2,4 |
| zidane.jpg | C07 | não | No. | correta | | 2,4 |
| zidane.jpg | C08 | igual | The left half of the image contains more vehicles than the right half. | alucinada | | 3,4 |
| zidane.jpg | C09 | nenhum | Motorcycle. | alucinada | | 2,6 |
| zidane.jpg | C10 | nenhum | The image does not provide information about the type of vehicles in the image. | correta | | 3,4 |
