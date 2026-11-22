# FluxoVC — Monitor de fluxo de veículos em vias urbanas

Trabalho final da disciplina de **<Disciplina>**: uma aplicação interativa de visão
computacional em Python que detecta, rastreia e conta veículos em vídeo, sem sensores físicos.

> 🚧 Em desenvolvimento. Este README será completado ao longo das fases do projeto.

## Problema

Prefeituras, órgãos de trânsito e pesquisadores de mobilidade precisam contar veículos, saber
por onde passam e estimar a velocidade média sem instalar sensores. O FluxoVC recebe um vídeo
(arquivo ou câmera) e:

- detecta e rastreia veículos (carro, moto, ônibus, caminhão);
- conta quantos cruzam uma linha virtual, por categoria e por sentido;
- desenha trajetórias e um mapa de calor de ocupação;
- estima a velocidade aproximada com uma transformação de perspectiva (visão de cima).

**Público:** gestores de trânsito, estudantes e pesquisadores de mobilidade urbana.

## Requisitos

- Python 3.10 ou superior (testado com **Python 3.12.10** no Windows 10)
- Roda em CPU; usa GPU NVIDIA automaticamente se houver PyTorch com CUDA instalado
- Bibliotecas e versões: veja [requirements.txt](requirements.txt)

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Na primeira análise, os pesos do YOLO (`yolo11n.pt`) são baixados automaticamente
(é preciso internet nessa primeira vez).

## Execução

```bash
streamlit run app.py
```

## Testes

```bash
pytest
```

## Estrutura de pastas

```
FluxoVC/
├── app.py                      # interface Streamlit (só UI)
├── src/
│   ├── acquisition/            # leitura de arquivo, webcam, imagem; validação
│   ├── preprocessing/          # letterbox, ROI, CLAHE, redução de ruído
│   ├── models/                 # detector/rastreador + contagem
│   ├── geometry/               # coordenadas, homografia, suavização
│   ├── visualization/          # camadas de desenho
│   ├── metrics/                # métricas de qualidade e desempenho
│   ├── config.py               # parâmetros padrão
│   └── logs.py                 # configuração de logging
├── data_samples/               # vídeos curtos + LICENSES.md
│   └── ground_truth/           # contagens manuais e frames anotados
├── outputs/                    # resultados gerados (ignorado pelo git)
├── tests/
├── docs/
└── scripts/
```

## Licenças e créditos

- [Ultralytics](https://github.com/ultralytics/ultralytics) — **AGPL-3.0**. Pesos pré-treinados no
  dataset [COCO](https://cocodataset.org/).
- Vídeos de amostra: origem e licença de cada um em [data_samples/LICENSES.md](data_samples/LICENSES.md).

## Equipe

| Integrante | Responsabilidade |
|---|---|
| `<Integrante 1>` | Aquisição e validação + interface |
| `<Integrante 2>` | Pré-processamento + geometria |
| `<Integrante 3>` | Modelo, rastreamento e contagem |
| `<Integrante 4>` | Visualização + métricas + documentação |
