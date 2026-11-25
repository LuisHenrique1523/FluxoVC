# 🚦 FluxoVC — Monitor de fluxo de veículos em vias urbanas

Trabalho final da disciplina de **Computação Gráfica** (`<Disciplina / turma / semestre>`).
O FluxoVC é uma aplicação interativa de visão computacional em Python que **detecta, rastreia e
conta veículos** em vídeo, desenha trajetórias e um mapa de calor de ocupação e **estima a
velocidade** com uma transformação de perspectiva, sem sensores físicos na via.

![Demonstração na cena sintética](docs/img/demo.gif)

*Demonstração gerada com o código do FluxoVC sobre uma cena sintética com detecções simuladas.
Veja como ela é gerada em [docs/computacao_grafica.md](docs/computacao_grafica.md).*
`<Equipe: acrescentar aqui um GIF ou print com o vídeo real de data_samples/.>`

## Sumário

[Problema](#problema) · [Requisitos](#requisitos) · [Instalação](#instalação) ·
[Execução](#execução) · [Arquitetura](#arquitetura) · [Pré-processamento](#pré-processamento) ·
[Método visual](#método-visual) · [Parâmetros](#parâmetros-da-interface) ·
[Métricas](#métricas-e-resultados) · [VLM](#vlm-opcional-descrever-cena) · [Acertos e erros](#exemplos-de-acertos-e-erros) ·
[Limitações](#limitações-conhecidas) · [Ética](#ética-e-privacidade) ·
[Créditos](#fontes-créditos-e-licenças) · [Uso de IA](#declaração-de-uso-de-ia) ·
[Equipe](#contribuição-dos-integrantes)

## Problema

Prefeituras, órgãos de trânsito e pesquisadores de mobilidade precisam saber **quantos veículos
passam, de que tipo, em que sentido e a que velocidade**. Contadores pneumáticos, laços indutivos
e radares exigem instalação e manutenção na via. O FluxoVC usa só um vídeo, de arquivo, webcam ou
câmera IP, e:

- detecta **carros, motos, ônibus e caminhões** (YOLO11, classes do COCO) e mantém um **ID** para
  cada veículo (ByteTrack);
- **conta** quantos cruzam uma linha virtual, por classe e por sentido (A→B e B→A);
- desenha **trajetórias suavizadas** e um **mapa de calor** da ocupação da pista;
- estima a **velocidade** (km/h) com uma **homografia** calibrada pelo usuário e mostra a **vista
  de cima** da pista;
- exporta os eventos (CSV) e o vídeo anotado (MP4).

**Público:** gestores de trânsito, estudantes e pesquisadores de mobilidade urbana.

**Por que não é só rodar um modelo pronto:** o YOLO só entrega caixas. Foram implementados pela
equipe:
- a conversão entre espaços de coordenadas (letterbox e seu inverso);
- a ROI;
- a contagem por cruzamento de linha;
- a homografia para velocidade e a vista de cima;
- a suavização das trajetórias;
- o mapa de calor;
- a composição por camadas;
- as métricas contra a contagem manual;
- a interface com parâmetros ao vivo.

## Requisitos

- **Python 3.10+**, testado com **Python 3.12.10** no **Windows 10**.
- Roda em **CPU**. Se houver GPU NVIDIA com PyTorch CUDA instalado, ela é usada automaticamente.
- Internet **só na primeira análise**, para baixar os pesos do YOLO (cerca de 5,6 MB o `yolo11n.pt`
  e 19 MB o `yolo11s.pt`). Os pesos ficam em `pesos/`, fora do git.

| Biblioteca | Versão | Uso |
|---|---|---|
| opencv-python | 5.1.0.95 | leitura de vídeo, pré-processamento, desenho, homografia |
| numpy | 2.5.5 | operações vetoriais |
| pillow | 13.0.0 | imagens na interface e GIF |
| ultralytics | 8.4.216 | YOLO11 + ByteTrack |
| lap | 0.5.14 | associação do ByteTrack |
| torch / torchvision | 2.15.0 / 0.29.2 | backend do YOLO |
| streamlit | 1.67.0 | interface |
| streamlit-image-coordinates | 0.4.1 | clique na prévia (marcação com o mouse) |
| pandas / matplotlib | 3.1.1 / 3.11.2 | tabelas, métricas e gráficos |
| psutil | 7.2.2 | memória (métricas de desempenho) |
| python-dotenv | 1.2.4 | `.env` (VLM opcional) |
| transformers / num2words | 5.24.0 / 0.5.14 | VLM local (SmolVLM), opcional |
| anthropic | 2.12.0 | VLM pela API do Claude, opcional |
| pytest | 9.2.0 | testes |

As versões fixadas estão em [requirements.txt](requirements.txt).

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Windows:** o PyTorch tem arquivos com caminhos muito longos. Se o `pip` falhar com
> `OSError: [Errno 2] No such file or directory` e uma dica sobre *Long Path*, clone o projeto numa
> pasta de caminho curto (ex.: `C:\fluxovc`) ou
> [ative o suporte a caminhos longos](https://pip.pypa.io/warnings/enable-long-paths).

Opcional: para configurar o VLM, copie `.env.example` para `.env` (veja [docs/vlm.md](docs/vlm.md)).

## Execução

```bash
streamlit run app.py
```

O navegador abre em <http://localhost:8501>.

1. Na barra lateral, escolha a **fonte**: vídeo de demonstração, envio de vídeo, webcam, URL ou
   imagem única. Vídeos de trânsito com licença livre para analisar podem ser baixados no
   [Pexels](https://www.pexels.com/search/videos/traffic/) e enviados pela opção **Enviar vídeo**.
2. A **prévia** mostra a ROI (ciano), a linha de contagem (amarelo, lados A e B) e a calibração
   (magenta). Para mudar, use **🖱️ Marcar na imagem com o mouse**, acima da prévia, e clique:
   - **📏 Linha de contagem:** 2 cliques, o início (A) e o fim (B), atravessando a pista;
   - **🔷 ROI:** os cantos da área da pista, e depois **Concluir ROI**;
   - **🚀 Calibração:** os 4 cantos de um trecho retangular da pista, em qualquer ordem. Depois
     informe a **largura e o comprimento reais** desse trecho no painel que se abre na barra
     lateral.

   Os pontos clicados aparecem em laranja, e há botões para desfazer e recomeçar. Os valores da
   barra lateral são preenchidos sozinhos e continuam disponíveis para ajuste fino.
3. Clique em **▶️ Analisar**. Ao final, baixe os eventos, o resumo e o vídeo processado.

![Prévia com a ROI, a linha de contagem e a calibração marcadas num vídeo real](docs/img/interface_marcacao.png)

### Outros comandos

```bash
pytest                                   # 354 testes, nenhum depende de baixar pesos (cerca de 1 min)
python scripts/evaluate.py               # avaliação em lote → docs/resultados.md
python scripts/extrair_frames.py <vídeo> # frames para anotar (CVAT / Label Studio)
python scripts/gerar_figuras.py          # figuras de docs/computacao_grafica.md
python scripts/avaliar_vlm.py --video <vídeo> --frames 3   # 10 casos de teste do VLM
```

### Deploy no Render

O repositório traz um [Dockerfile](Dockerfile) e um blueprint [render.yaml](render.yaml).
No [Render](https://render.com), use **New → Blueprint** e escolha este repositório, ou abra
<https://render.com/deploy?repo=https://github.com/LuisHenrique1523/FluxoVC>.

- A imagem instala o PyTorch **só CPU** e baixa o `yolo11n.pt` no build
  ([requirements-deploy.txt](requirements-deploy.txt) deixa de fora o VLM local e o pytest).
- No servidor não há **webcam**: use o envio de vídeo, uma URL ou uma imagem.
- O botão **Descrever cena** usa a API do Claude se a variável `ANTHROPIC_API_KEY` for definida no
  painel do Render; sem ela, aparece só um aviso.
- O plano gratuito tem 512 MB de RAM e CPU compartilhada: a análise é lenta e o serviço "dorme"
  após 15 min sem acesso. Se o processo for encerrado por falta de memória, troque `plan` para
  `standard` no `render.yaml`.

## Arquitetura

```mermaid
flowchart LR
    subgraph Entrada["src/acquisition"]
        A1[Arquivo / demo] --> V
        A2[Webcam] --> V
        A3[URL / câmera IP] --> V
        A4[Imagem única] --> V
        V[Validação<br/>extensão · tamanho · abre? · frames · FPS · resolução]
    end
    subgraph Pre["src/preprocessing"]
        P1[Letterbox 640×640] --> P2[CLAHE no L] --> P3[Bilateral / Gaussiano] --> P4[Máscara de ROI]
    end
    subgraph Modelo["src/models"]
        M1[YOLO11 + ByteTrack<br/>classes 2, 3, 5, 7] --> M2[Letterbox inverso<br/>modelo → original]
        M2 --> M3[Filtro: confiança · ROI]
        M3 --> M4[Trajetórias com EMA]
        M4 --> M5[Contagem por<br/>produto vetorial]
        M4 --> M6[Velocidade via<br/>homografia]
    end
    subgraph Vis["src/visualization"]
        C[Compositor de camadas<br/>calor · ROI · linha · caixas ·<br/>trajetórias · vista de cima · HUD]
    end
    subgraph UI["app.py + src/aplicacao"]
        U1[Streamlit: vídeo ao vivo, gráfico,<br/>eventos, comparação, downloads]
    end
    V --> P1
    P4 --> M1
    M5 --> C
    M6 --> C
    C --> U1
    U1 -. parâmetros ao vivo .-> Pre
    U1 -. parâmetros ao vivo .-> M1
    M5 --> R[src/metrics<br/>evaluate.py → resultados.md]
```

O `app.py` contém só a interface. A lógica fica em `src/`, testável sem Streamlit e sem pesos: os
testes injetam um detector falso que cumpre o mesmo contrato do YOLO.

### Estrutura de pastas

```
FluxoVC/
├── app.py                      # interface Streamlit (só UI)
├── src/
│   ├── acquisition/            # fontes (arquivo, webcam, URL, imagem) e validação
│   ├── preprocessing/          # letterbox, ROI, CLAHE, redução de ruído, frame skip
│   ├── geometry/               # espaços de coordenadas, ROI, homografia, EMA
│   ├── models/                 # wrapper YOLO + ByteTrack, interpretação, trajetórias, contagem
│   ├── visualization/          # camadas, mapa de calor, vista de cima, compositor, gravador MP4
│   ├── metrics/                # qualidade (P/R/F1, erro de contagem, MAE), desempenho, relatório
│   ├── aplicacao/              # sessão de análise frame a frame e tabelas para a interface
│   ├── vlm/                    # VLM opcional: anonimização, backends local/Claude, casos de teste
│   ├── config.py               # todos os parâmetros (dataclasses) + salvar/carregar JSON
│   └── logs.py                 # logging
├── data_samples/               # vídeos curtos + LICENSES.md
│   └── ground_truth/           # configuração, contagem manual e frames anotados
├── docs/                       # pipeline, computação gráfica, métricas, resultados, figuras
├── scripts/                    # evaluate.py, extrair_frames.py, gerar_figuras.py, avaliar_vlm.py…
├── .env.example                # configuração opcional do VLM (o .env não vai para o git)
├── tests/                      # pytest (vídeos sintéticos + detectores falsos)
├── outputs/                    # vídeos processados (ignorado pelo git)
├── pesos/                      # pesos do YOLO baixados (ignorado pelo git)
└── requirements.txt
```

### Documentação

| Documento | Conteúdo |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | cada etapa da pipeline, decisões e parâmetros |
| [docs/computacao_grafica.md](docs/computacao_grafica.md) | os 10 requisitos de CG, com código e figuras |
| [docs/metricas.md](docs/metricas.md) | definição das métricas, como fazer a contagem manual e anotar os frames |
| [docs/resultados.md](docs/resultados.md) | tabelas e gráficos gerados pelo `evaluate.py` |
| [docs/vlm.md](docs/vlm.md) | VLM opcional: anonimização, prompts, 10 casos de teste e resultados |
| [docs/entrega.md](docs/entrega.md) | checklist de entrega, pendências, roteiro de slides e do vídeo de segurança |

## Pré-processamento

Cada operação tem um toggle e uma explicação (ícone ❓) na interface, e todas são aplicadas
**nesta ordem**:

| Operação | O que faz | Por quê | Parâmetros |
|---|---|---|---|
| **Letterbox** | redimensiona para 640 × 640 com a mesma escala nos dois eixos e completa com faixas cinza | o YOLO espera um quadrado; esticar achataria os veículos | liga/desliga (desligado estica, só para comparação) |
| **CLAHE** | equalização adaptativa de contraste só no canal L (LAB) | recupera detalhes em cenas escuras ou em contraluz sem mudar as cores | limite de contraste, tamanho da grade |
| **Redução de ruído** | filtro bilateral (preserva bordas) ou gaussiano (mais rápido) | ruído de sensor e compressão gera detecções instáveis | tipo e tamanho do kernel |
| **Máscara de ROI** | pinta de cinza o que está fora do polígono | o modelo não gasta detecções com calçadas e estacionamentos | pontos do polígono |
| **Frame skip** | processa 1 frame e pula *n* | ganho de FPS em CPU | *n* (0 a 5) |

O letterbox vem primeiro para que o custo das demais operações seja o mesmo em qualquer resolução.
Detalhes em [docs/pipeline.md](docs/pipeline.md#2-pré-processamento--srcpreprocessing).

![Etapas do pré-processamento numa cena escura](docs/img/cg/11_preprocessamento.png)

## Método visual

Todo o desenho é feito com OpenCV, em camadas que podem ser ligadas e desligadas durante a
análise:

![Composição por camadas](docs/img/cg/04_camadas.png)

| Camada | Técnica |
|---|---|
| Mapa de calor | grade de ocupação (terço inferior das caixas), raiz quadrada, `GaussianBlur`, colormap TURBO/JET, `addWeighted` com máscara |
| ROI | `fillPoly` numa máscara e mistura semitransparente só dentro dela |
| Linha de contagem | lados A/B ao longo da normal; pisca em verde ao ser cruzada |
| Caixas | cor por classe e rótulo "classe #ID confiança km/h" |
| Trajetórias | pontos suavizados por EMA, cor por ID, cauda afinando |
| Vista de cima | `warpPerspective` do trecho calibrado, com os veículos e uma régua de 5 m |
| HUD | tabela classe × sentido, FPS e latência |

**Contagem:** o ponto de referência é o **centro da base da caixa**, o contato com o chão. Um
veículo é contado quando o **sinal do produto vetorial** entre esse ponto e a linha muda entre dois
frames **e** o deslocamento atravessa o **segmento** (e não o prolongamento da linha). Cada ID
conta uma vez.

**Velocidade:** 4 pontos de um trecho retangular da pista, mais as medidas reais, dão uma
homografia imagem → metros. A velocidade é a inclinação de uma reta ajustada às posições no plano
da pista numa janela de 1 s. **É obrigatório calibrar para cada vídeo:** a calibração padrão é só
um exemplo, e a interface avisa enquanto ela não for alterada.

Explicação completa, com código e figuras: [docs/computacao_grafica.md](docs/computacao_grafica.md).

## Parâmetros da interface

Todos ficam na barra lateral, sem editar código. Os marcados com ⚡ valem **ao vivo**, durante a
análise. Os demais valem a partir da próxima análise, para que uma contagem não misture duas
geometrias.

| Grupo | Parâmetro | Padrão |
|---|---|---|
| Fonte | demonstração · enviar vídeo (mp4/avi/mov, até 200 MB) · webcam (índice) · URL (http/rtsp) · imagem única | — |
| Modelo | YOLO11n (rápido) ou YOLO11s (mais preciso) | n |
| | ⚡ limiar de confiança | 0,35 |
| | ⚡ limiar de IoU (NMS) | 0,50 |
| Pré-processamento | ⚡ letterbox, ROI, CLAHE (limite, grade), redução de ruído (filtro, kernel), frame skip | letterbox e ROI ligados |
| Linha de contagem | 🖱️ 2 cliques na prévia (ou sliders A e B), ponto do veículo (base/centro) | horizontal em y = 0,60 |
| ROI | 🖱️ cliques nos cantos da pista (ou tabela de pontos, ≥ 3) | faixa inferior do frame |
| Calibração | 🖱️ 4 cliques em qualquer ordem (ou tabela), largura e comprimento reais (m), α da EMA | exemplo genérico, **precisa ser calibrada** |
| Camadas | ⚡ cada camada, opacidade do mapa de calor e da ROI, colormap | todas, exceto o mapa de calor |
| Ações | Analisar · Parar · Salvar configuração (JSON) · Limpar | — |

## Métricas e resultados

| Tipo | Métrica | Referência |
|---|---|---|
| Qualidade | erro de contagem absoluto e percentual, por classe e sentido; **MAE** por classe | contagem manual (`ground_truth/<vídeo>_contagem.csv`) |
| Qualidade | **precisão, recall e F1** de detecção (IoU ≥ 0,5) | cerca de 30 frames anotados no formato YOLO |
| Qualidade | estabilidade do rastreamento: IDs por veículo real e trocas de ID estimadas | contagem manual |
| Desempenho | **FPS**, × tempo real, latência por etapa (média e p95), memória de pico | — |

Comparações: **YOLO11n × YOLO11s** e **com × sem frame skip**. Tudo é gerado por
`python scripts/evaluate.py` em **[docs/resultados.md](docs/resultados.md)**. O procedimento de
anotação está em [docs/metricas.md](docs/metricas.md).

Medido numa CPU AMD sem GPU, com o pipeline completo: **cerca de 9 FPS com o YOLO11n e 4 FPS com o
YOLO11s**.

`<Equipe: depois de rodar o evaluate.py com o vídeo real, resumir aqui os números principais
(MAE, F1 e FPS) e o que eles mostram.>`

## VLM opcional: "Descrever cena"

Depois de uma análise, o painel **🤖 Descrever cena com IA** envia o frame a um modelo de visão e
linguagem, que descreve a cena ou responde a uma pergunta livre. O VLM **complementa** a pipeline:
a resposta aparece ao lado da contagem da pipeline naquele frame, sempre com o aviso *"Resposta
gerada por IA, pode conter erros"*.

- **Anonimizado antes de tudo:** o frame é reduzido para 768 px, e pessoas e regiões de placa são
  pixelizadas.
- **Local por padrão:** SmolVLM-256M (Hugging Face), cerca de 3 s por resposta em CPU. O frame não
  sai do computador.
- **API opcional:** com `FLUXOVC_VLM_BACKEND=claude` e `ANTHROPIC_API_KEY` no `.env` (veja
  [.env.example](.env.example)). Nenhuma chave fica no código.
- **10 casos de teste** com respostas verificáveis pela pipeline, classificados em correta,
  incompleta, incorreta ou alucinada. Primeira rodada com o SmolVLM: 75 % corretas e 20 %
  alucinadas, como contar "10 veículos" numa foto sem nenhum.

Prompts, regras e resultados em [docs/vlm.md](docs/vlm.md).

## Exemplos de acertos e erros

**Acertos verificados (cena sintética, em que a resposta certa é conhecida):**
- Os 8 veículos em movimento foram contados, com a classe e o sentido corretos. O carro
  estacionado na calçada foi ignorado pela ROI (`tests/test_cena_sintetica.py`).
- A velocidade estimada pela homografia ficou a até **5 %** da velocidade real
  ([tabela](docs/computacao_grafica.md#3-transformação-de-perspectiva-homografia)).
- O letterbox inverso devolve a coordenada original com erro ≤ 1 px em 6 resoluções.

**Erros e limitações observados:**
- **Frame skip + suavização forte:** com frame skip 2 e α = 0,4, o ponto suavizado de um veículo
  ficou atrasado, o veículo saiu da ROI antes de cruzar a linha e **não foi contado**
  ([detalhes](docs/metricas.md#efeito-observado-suavização--frame-skip)).
- **Fora da distribuição do modelo:** o YOLO não reconhece veículos desenhados (cena sintética) e
  a interface mostra o aviso de "nenhum veículo detectado":

![Aviso de nenhum veículo detectado](docs/img/interface_sem_deteccoes.png)

`<Equipe: acrescentar 2 prints de acerto e 2 de erro com o vídeo real (por exemplo: oclusão entre
um ônibus e um carro, moto não detectada à distância, troca de ID), com uma frase explicando cada
um.>`

## Limitações conhecidas

- **Oclusão:** veículos escondidos atrás de ônibus ou caminhões podem perder o ID e ser
  recontados, ou nem ser vistos.
- **Noite, chuva e contraluz:** reduzem o recall. O CLAHE e o filtro ajudam, mas não resolvem
  faróis ofuscando a câmera nem gotas na lente.
- **Câmera tremida ou em movimento:** a linha, a ROI e a calibração são fixas na imagem. Se a
  câmera mexer, a contagem e a velocidade erram.
- **Velocidade depende da calibração:** medidas reais erradas viram erro proporcional, e a
  homografia só vale para o plano da pista (rampas e lombadas distorcem).
- **Classes do COCO:** vans, picapes, bicicletas, carroças e veículos especiais são confundidos
  com carro ou caminhão, ou ignorados.
- **Veículos pequenos ou distantes:** com poucos pixels na imagem o detector falha, principalmente
  com o YOLO11n.
- **Desempenho em CPU:** cerca de 9 FPS. Para vídeo ao vivo a 30 fps, use frame skip ou GPU.
- **Contagem perto da borda da ROI:** se a linha estiver perto da borda, veículos podem sair da
  ROI antes de serem contados. Deixe uma margem.

## Ética e privacidade

- **Foco em veículos:** a contagem processa só as classes carro, moto, ônibus e caminhão. A classe
  **pessoa é descartada** já na saída do detector. A única exceção é o VLM opcional: lá, as pessoas
  são detectadas **só para serem pixelizadas** antes de o frame ser enviado.
- **Sem placas e sem identificação:** nenhuma placa é lida, recortada ou salva. O sistema não faz
  OCR nem reconhecimento facial.
- **Dados locais:** a análise roda no computador do usuário. Os vídeos enviados ficam numa pasta
  temporária, e os resultados vão para `outputs/`, fora do git. Nada é enviado para a internet,
  exceto o download dos pesos na primeira vez. A exceção é o backend `claude` do VLM, se a equipe
  o ativar no `.env`: aí o frame **já anonimizado** vai para a API da Anthropic.
- **Vídeos de teste:** só com licença livre (Pexels, Pixabay) ou gravados pela equipe em via
  pública, sem pessoas identificáveis nem placas legíveis. Origem e licença em
  [data_samples/LICENSES.md](data_samples/LICENSES.md).
- **Uso responsável:** o FluxoVC é uma ferramenta de **estatística de fluxo**. A velocidade
  estimada é aproximada e **não serve para fiscalização ou multa**.

## Fontes, créditos e licenças

| Item | Licença / origem |
|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (detector e implementação do ByteTrack) | **AGPL-3.0**. Como o FluxoVC usa essa biblioteca, o código deste projeto também é distribuído publicamente (este repositório). |
| Pesos `yolo11n.pt` / `yolo11s.pt` | treinados pela Ultralytics no dataset **[COCO](https://cocodataset.org/)** (anotações CC BY 4.0), baixados de [ultralytics/assets](https://github.com/ultralytics/assets/releases) |
| ByteTrack (algoritmo) | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022 |
| OpenCV | Apache-2.0 |
| NumPy, pandas, PyTorch, torchvision, psutil, python-dotenv | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| Streamlit | Apache-2.0 |
| streamlit-image-coordinates | MIT |
| matplotlib | Matplotlib License (baseada na PSF) |
| lap | BSD-2-Clause |
| pytest | MIT |
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) (VLM local) | Apache-2.0 (Hugging Face) |
| transformers | Apache-2.0 |
| anthropic (SDK da API do Claude) | MIT |
| num2words | LGPL-2.1 |
| Fotos de exemplo usadas no teste do VLM (`bus.jpg`, `zidane.jpg`) | repositório do Ultralytics (AGPL-3.0); só a versão anonimizada de `bus.jpg` está em `docs/img/` |
| Vídeos de amostra | ver [data_samples/LICENSES.md](data_samples/LICENSES.md) |

**Código adaptado:** nenhum trecho foi copiado de terceiros. O letterbox segue a convenção do
Ultralytics (escala única, imagem centralizada, preenchimento cinza 114), para que a entrada seja
igual à usada no treino. `<Equipe: listar aqui qualquer código adaptado de outra fonte, com o
link.>`

## Declaração de uso de IA

> **Equipe: preencher e revisar este bloco antes da entrega.**

| Campo | Resposta |
|---|---|
| **Ferramenta** | Claude Code (Anthropic) |
| **Finalidade** | `<ex.: gerar o código inicial de cada fase a partir do nosso prompt, escrever testes e documentação, sugerir correções>` |
| **Partes afetadas** | `<ex.: todos os módulos de src/, app.py, tests/, scripts/ e docs/ foram gerados com auxílio da ferramenta, fase a fase>` |
| **Como a equipe revisou e validou** | `<ex.: leitura de cada fase antes do commit; execução do pytest; testes manuais no app com o vídeo real; comparação da contagem com a contagem manual; ajustes feitos por nós: ...>` |
| **O que a equipe fez sem IA** | `<ex.: escolha e gravação dos vídeos, contagem manual, anotação dos frames, calibração das medidas reais, análise dos resultados>` |
| **Limitações percebidas** | `<ex.: erros encontrados no código gerado e como foram corrigidos>` |


## Contribuição dos integrantes

| Integrante | Responsabilidade |
|---|---|
| `<Integrante 1>` | Aquisição e validação + interface |
| `<Integrante 2>` | Pré-processamento + geometria |
| `<Integrante 3>` | Modelo, rastreamento e contagem |
| `<Integrante 4>` | Visualização + métricas + documentação |
