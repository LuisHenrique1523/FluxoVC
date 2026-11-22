# Métricas e avaliação

Este guia explica **o que** o FluxoVC mede, **como preparar** a referência manual (ground truth)
e **como rodar** a avaliação que gera [`docs/resultados.md`](resultados.md).

## 1. O que é medido

### Qualidade

| Métrica | O que responde | Como é calculada | Referência necessária |
|---|---|---|---|
| **Erro de contagem** (absoluto e %) | "Quantos veículos o sistema errou?" | `|previsto − real|` por classe e por classe + sentido; `%` = erro ÷ real | contagem manual (`<video>_contagem.csv`) |
| **MAE por classe** | "Em média, quanto erra por classe?" | média do erro absoluto da classe entre os vídeos; *MAE geral* = média entre as classes | idem |
| **Precisão / Recall / F1** | "O detector acha os veículos certos?" | casamento previsão × anotação com **mesma classe e IoU ≥ 0,5**, um para um, do maior IoU para o menor. P = VP/(VP+FP), R = VP/(VP+FN), F1 = 2PR/(P+R). A linha *geral* soma VP, FP e FN de todas as classes (micro-média) | ~30 frames anotados (YOLO) |
| **Estabilidade do rastreamento** | "O mesmo veículo mantém o mesmo ID?" | IDs vistos em ≥ 5 frames ÷ veículos reais (ideal 1,0); trocas estimadas = IDs válidos − veículos reais | contagem manual |

A estabilidade é uma **estimativa**: sem anotar o ID de cada veículo quadro a quadro, não dá para
saber exatamente quando houve troca. Um veículo nunca detectado diminui o número de IDs e pode
esconder uma troca de outro veículo.

### Desempenho

| Métrica | Definição |
|---|---|
| **FPS** | frames do vídeo ÷ segundos de processamento (inclui os frames pulados pelo frame skip) |
| **× tempo real** | duração do vídeo ÷ tempo de processamento; ≥ 1 acompanha o vídeo ao vivo |
| **Latência por etapa** | média em ms por frame processado: pré-processamento, inferência, interpretação e desenho; mais o total e o p95 do total |
| **Memória pico** | maior memória residente (RSS) do processo durante a análise, via `psutil` |

O script compara **YOLO11n × YOLO11s** e **frame skip 0 × 2** (configurável).

## 2. Preparando a referência (ground truth)

Estrutura esperada em `data_samples/ground_truth/`:

```
data_samples/
├── avenida.mp4                    # vídeo de teste (licença em LICENSES.md)
└── ground_truth/
    ├── avenida.json               # configuração do vídeo (linha, ROI, calibração)
    ├── avenida_contagem.csv       # contagem manual
    ├── para_anotar/               # frames extraídos (entrada do CVAT/Label Studio)
    └── deteccao/                  # exportação YOLO (imagens + .txt + classes)
```

### 2.1 Configuração do vídeo (`<video>.json`)

A contagem depende de **onde está a linha**. Por isso, cada vídeo tem a sua configuração:

1. `streamlit run app.py` e selecione o vídeo.
2. Ajuste a linha de contagem, a ROI e a calibração e confira na prévia.
3. Clique em **💾 Salvar configuração (JSON)**. O arquivo já vem com o nome do vídeo; mova-o para
   `data_samples/ground_truth/`.

Sem esse arquivo, a avaliação usa a configuração padrão (linha horizontal em y = 0,60) e avisa no
relatório.

### 2.2 Contagem manual (`<video>_contagem.csv`)

1. Abra a prévia na interface e anote **onde a linha está** e **qual lado é A e qual é B**. Os
   rótulos A e B aparecem junto da linha.
2. Assista ao vídeo num player que avance quadro a quadro. No **VLC**, a tecla `E` avança um
   quadro; ative em *Ferramentas → Preferências → Atalhos*. Reduza a velocidade se o trânsito for
   intenso.
3. Conte cada veículo **quando o ponto de contato com o chão (centro da base) cruzar a linha**,
   anotando a classe e o sentido. A→B significa sair do lado A e chegar ao lado B.
4. **Duas pessoas contam de forma independente.** Onde discordarem, revejam o trecho juntas. A
   diferença entre as duas contagens iniciais mostra o quanto a própria referência é incerta e vale
   citar no relatório.
5. Salve o CSV (Excel em pt-BR → "CSV separado por `;`" funciona):

```
classe;sentido;quantidade
carro;A→B;12
carro;B→A;9
moto;A→B;3
ônibus;B→A;1
caminhão;A→B;2
```

Aceita `,` ou `;`, `A->B` ou `A→B`, e nomes com ou sem acento (`onibus`, `bus`, `caminhao`,
`truck`…). Classes com zero podem ser omitidas.

**Regras para combinar com a equipe antes de contar:**
- Veículo que cruza e volta conta uma vez, pelo primeiro cruzamento.
- Veículo que já está sobre a linha no primeiro frame **não conta**, porque o sistema também não
  tem como contá-lo.
- Bicicletas, patinetes e pessoas não contam, porque não são classes do modelo.
- Vans e picapes: decidam se são "carro" ou "caminhão" e mantenham o critério. O COCO não tem a
  classe "van", e o modelo costuma chamá-las de carro ou caminhão.

### 2.3 Frames anotados para detecção (~30)

**1. Extraia os frames**, igualmente espaçados no vídeo:

```bash
python scripts/extrair_frames.py data_samples/avenida.mp4 --quantidade 30
```

Eles vão para `ground_truth/para_anotar/`, com o número do frame no nome
(`avenida_f000123.jpg`).

**2. Anote** numa das ferramentas abaixo (as duas são gratuitas). Use as classes **exatamente**:
`carro`, `moto`, `onibus`, `caminhao`.

<details>
<summary><b>Opção A — CVAT (no navegador, sem instalar nada)</b></summary>

1. Crie uma conta em <https://app.cvat.ai>.
2. **Projects → +** → nome "FluxoVC" → em *Labels*, adicione `carro`, `moto`, `onibus` e `caminhao`
   (tipo *Rectangle*).
3. **Tasks → +** → selecione o projeto → envie as imagens de `para_anotar/` → *Submit*.
4. Abra o *Job* e anote com a ferramenta de retângulo: `N` começa e termina a caixa, `F` vai para o
   próximo frame. Faça **uma caixa por veículo** e ajuste bem as bordas.
5. **Actions → Export task dataset** → formato **YOLO 1.1** → marque **Save images** → OK.
6. Descompacte o `.zip` em `data_samples/ground_truth/deteccao/`. Ficam `obj.names`, `obj.data` e
   `obj_train_data/` com as imagens e os `.txt`.

</details>

<details>
<summary><b>Opção B — Label Studio (local)</b></summary>

1. Num ambiente separado, para não misturar dependências com o projeto:
   `python -m venv .venv-ls && .venv-ls\Scripts\activate && pip install label-studio`
2. `label-studio start` (abre em <http://localhost:8080>) → crie uma conta local.
3. **Create Project** → *Labeling Setup* → **Object Detection with Bounding Boxes** → troque os
   rótulos por `carro`, `moto`, `onibus` e `caminhao`.
4. *Data Import* → envie as imagens de `para_anotar/`.
5. Anote: escolha o rótulo (teclas 1–4) e arraste a caixa. Clique em *Submit* em cada imagem.
6. **Export** → **YOLO with Images** → descompacte em `data_samples/ground_truth/deteccao/`. Ficam
   `images/`, `labels/` e `classes.txt`.

</details>

**Regras de anotação:**
- Anote **todo veículo das 4 classes** visível no frame, inclusive os distantes, desde que dê para
  reconhecer a classe. Inclua veículos fora da ROI: a avaliação de detecção usa o frame inteiro.
- Veículo parcialmente encoberto: desenhe a caixa só da **parte visível**.
- Veículo cortado na borda da imagem: anote se mais da metade estiver visível.
- Não anote pessoas nem placas. Se a ferramenta exigir, nunca transcreva o texto de uma placa.

**Atenção:** dentro de `deteccao/` devem ficar **só** os arquivos exportados. Uma imagem sem `.txt`
correspondente é tratada como "frame sem veículos", e todas as detecções nela viram falsos
positivos.

## 3. Rodando a avaliação

```bash
python scripts/evaluate.py                       # completo: YOLO11n e YOLO11s, frame skip 0 e 2
python scripts/evaluate.py --max-frames 300      # rápido: só os 300 primeiros frames de cada vídeo
python scripts/evaluate.py --modelos yolo11n.pt --frame-skip 0 1 3
python scripts/evaluate.py --videos data_samples/avenida.mp4 --dispositivo cpu
```

O script escreve `docs/resultados.md` e os gráficos em `docs/img/resultados/`. **Não edite o
relatório à mão**: ajuste as referências e rode de novo.

Tempo de execução medido numa CPU AMD sem GPU, com o pipeline completo (inclusive o desenho):
**cerca de 9 FPS com o YOLO11n e 4 FPS com o YOLO11s**. Um vídeo de 30 s a 30 fps (900 frames)
leva uns 2 min com o n e 4 min com o s, para cada valor de frame skip. Para testar o processo
antes, use `--max-frames`.

## 4. Como interpretar

- **Contagem baixa e recall alto** → o detector acha os veículos, mas o rastreamento ou a contagem
  falham. Causas comuns: linha perto da borda da ROI, troca de ID no cruzamento, frame skip alto.
- **Recall baixo** → veículos não detectados: pequenos e distantes, noite, oclusão. Tente reduzir
  a confiança, ligar o CLAHE ou usar o YOLO11s.
- **Precisão baixa** → falsos positivos: sombras, reflexos, veículos estacionados. Ligue a ROI ou
  aumente a confiança.
- **IDs por veículo > 1** → o ByteTrack perdeu e recriou IDs, geralmente por oclusão ou por queda de
  FPS com frame skip.

### Efeito observado: suavização × frame skip

A EMA atrasa o ponto suavizado em relação ao veículo em cerca de
`(1 − α) / α × deslocamento entre frames processados`. Com frame skip, o deslocamento entre frames
processados aumenta, e o atraso também. No vídeo sintético dos testes, com α = 0,4 e frame skip 2,
o atraso chegou a uns 40 px, e o caminhão **saiu da ROI antes de o ponto suavizado cruzar a
linha**, então não foi contado. Com frame skip 0, ele foi contado.

**Como evitar:** deixe a linha de contagem **longe da borda da ROI**, aumente o α quando usar frame
skip, ou não use frame skip quando a contagem for o objetivo principal.
