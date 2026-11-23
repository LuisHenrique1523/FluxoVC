# Entrega: checklist, roteiro de slides e roteiro do vídeo

Este documento é para a **equipe**: o que já está pronto, o que falta fazer antes de entregar e
os roteiros da apresentação e do vídeo.

## 1. Checklist de entrega

Situação em 22/11/2026, depois da Fase 8: ✅ pronto · ⚠️ pronto em parte, falta algo da equipe ·
❌ falta.

| Item | Situação | Evidência / o que falta |
|---|---|---|
| App inicia com o comando do README | ✅ | `streamlit run app.py` sobe sem erros; página verificada no navegador |
| Vídeo de demonstração em `data_samples/` com licença registrada | ❌ | **Falta o vídeo.** 1 a 3 vídeos curtos (< 20 MB, câmera fixa, 10 a 30 s) com origem e licença em `data_samples/LICENSES.md` |
| Interface seleciona entrada e altera ≥ 2 parâmetros | ✅ | 5 fontes de entrada e mais de 20 parâmetros na barra lateral, vários deles ao vivo (`tests/test_aplicacao.py`) |
| ≥ 3 pré-processamentos explicados | ✅ | 4 operações mais o frame skip, cada uma com toggle e ajuda na interface, no README e em `docs/pipeline.md` |
| Resultados gráficos (caixas, trajetórias, heatmap, linha, bird's-eye) | ✅ | `docs/computacao_grafica.md` (figuras 04 a 07) |
| ≥ 1 métrica de qualidade e ≥ 1 de desempenho, em casos identificados | ⚠️ | Código e testes prontos (`scripts/evaluate.py`). FPS medido (cerca de 9 com o n e 4 com o s), velocidade validada na cena sintética (erro de até 5 %) e VLM avaliado em 20 perguntas. **Falta:** contagem manual e cerca de 30 frames anotados do vídeo real, depois rodar o `evaluate.py` (`docs/metricas.md`) |
| Arquivo inválido, sem câmera e sem detecções tratados | ✅ | Tabela na seção 8 de `docs/pipeline.md`; `tests/test_erros.py` |
| Caminhos relativos; sem segredos, venv, caches, pesos ou dados grandes | ✅ | Nenhum caminho absoluto no código; `.env`, `pesos/`, `*.pt`, `outputs/` e `.venv/` no `.gitignore`; repositório com cerca de 10 MB (só figuras) |
| `requirements.txt` testado em ambiente limpo | ✅ | Ambiente novo (Python 3.12.10, Windows 10, cópia do repositório): `pip install -r requirements.txt` sem erros; 352 testes passaram e 2 foram pulados de propósito (precisam dos modelos já baixados). No Windows, a instalação exige um caminho de pasta curto (aviso no README) |
| Créditos, licenças e uso de IA declarados | ⚠️ | Créditos e licenças completos no README. **Falta:** preencher o bloco "Declaração de uso de IA" e os nomes da equipe |
| Exemplos de acertos, erros e limitações documentados | ⚠️ | Acertos e erros na cena sintética, prints da interface, limitações e erros do VLM. **Falta:** 2 prints de acerto e 2 de erro com o vídeo real |

### Pendências da equipe, em ordem

1. **Vídeo:** baixar 1 a 3 vídeos de trânsito com licença livre (Pexels ou Pixabay) para
   `data_samples/` e preencher o `LICENSES.md`.
2. **Configuração por vídeo:** na interface, ajustar linha, ROI e calibração e clicar em
   **Salvar configuração**; mover para `data_samples/ground_truth/<vídeo>.json`.
3. **Contagem manual:** feita por duas pessoas, em `ground_truth/<vídeo>_contagem.csv`
   (`docs/metricas.md`, seção 2.2).
4. **Anotação:** `python scripts/extrair_frames.py data_samples/<vídeo>.mp4`, anotar no CVAT ou
   Label Studio e exportar em YOLO para `ground_truth/deteccao/`.
5. **Avaliação:**
   - `python scripts/evaluate.py`, que gera `docs/resultados.md`;
   - `python scripts/avaliar_vlm.py --video data_samples/<vídeo>.mp4 --frames 3`, e depois revisar
     a coluna "revisão".
6. **Prints:** fazer prints com o vídeo real (2 acertos e 2 erros) e um GIF curto para o topo do
   README.
7. **README:** preencher os campos `<...>` (disciplina, resumo dos resultados, declaração de uso de
   IA, nomes).
8. **Histórico:** se o professor exigir commits por integrante, cada um faz o commit da sua parte
   nos passos acima (dados, anotações, documentação).

---

## 2. Roteiro de slides

Apresentação de cerca de **15 minutos**, com 14 slides. A coluna "Quem" segue a tabela de
contribuição do README (I1 a I4).

| # | Slide | Conteúdo | Visual | Quem | Tempo |
|---|---|---|---|---|---|
| 1 | **FluxoVC** | Título, equipe, disciplina; frase do problema: "contar veículos sem sensores físicos" | GIF `docs/img/demo.gif` (ou do vídeo real) | I1 | 0:30 |
| 2 | **Problema e público** | Por que contar (planejamento, semáforos, pesquisa); custo de laços indutivos e radares; público: gestores, pesquisadores | Foto de uma via (do vídeo) | I1 | 1:00 |
| 3 | **Pipeline** | Aquisição → validação → pré-processamento → YOLO + ByteTrack → letterbox inverso → trajetórias, contagem e velocidade → camadas → interface | Diagrama Mermaid do README | I1 | 1:00 |
| 4 | **Pré-processamento** | Letterbox, CLAHE no canal L, filtro bilateral, ROI e frame skip: o que cada um resolve e por que nessa ordem | `11_preprocessamento.png` | I2 | 1:15 |
| 5 | **Modelo** | YOLO11n × YOLO11s, COCO (classes 2, 3, 5 e 7), ByteTrack; pessoas ignoradas; roda em CPU | Tabela n × s (FPS) | I3 | 1:00 |
| 6 | **Coordenadas** | Modelo → original → tela; a conta do letterbox inverso com números reais; teste de ida e volta ≤ 1 px | `01_coordenadas.png` e `02_letterbox.png` | I2 | 1:15 |
| 7 | **Contagem e velocidade** | Produto vetorial e segmento; contar cada ID uma vez; homografia de 4 pontos; velocidade real × estimada (erro de até 5 %); o bug da velocidade subestimada encontrado com vídeo real | `03_homografia.png` e a tabela de velocidades | I3 | 1:30 |
| 8 | **Gráficos em camadas** | Ordem das camadas, `addWeighted` com máscara, mapa de calor com colormap, EMA (tremor × atraso) | `04_camadas.png`, `05_mapa_calor.png`, `08_ema.png` | I4 | 1:30 |
| 9 | **Interface** | Fontes, parâmetros ao vivo × congelados, prévia, gráfico, eventos, downloads; tratamento de erros | `interface_previa.png` ou demonstração ao vivo | I1 | 1:30 |
| 10 | **Métricas** | Erro de contagem e MAE, P/R/F1 com IoU ≥ 0,5, estabilidade do rastreamento, FPS e latência por etapa | Tabelas de `docs/resultados.md` | I4 | 1:15 |
| 11 | **Acertos e erros** | 2 acertos e 2 erros reais; frame skip × suavização; limitações (oclusão, noite, calibração) | Prints do vídeo real | I3 | 1:00 |
| 12 | **Evolução** | 9 fases em commits; bugs que os testes pegaram; desempenho (composição de 37 → 22 ms; VLM de 37 → 3 s por resposta) | Linha do tempo dos commits | I4 | 0:45 |
| 13 | **VLM** | "Descrever cena" com frame anonimizado; local × API; 10 casos: 75 % corretas, 20 % alucinadas | `vlm_anonimizado.jpg` e tabela de resultados | I2 | 1:00 |
| 14 | **Contribuições e créditos** | Quem fez o quê; Ultralytics (AGPL-3.0), COCO, SmolVLM, vídeos (licenças); declaração de uso de IA | Tabelas do README | todos | 0:30 |

**Exemplos para o slide 12 (bugs que os testes pegaram):**
- **HUD desalinhado:** a fonte do OpenCV tem largura variável, então as colunas passaram a ter
  x fixo.
- **Trajetória escondida:** ficava atrás da borda inferior da caixa, então passou a ser desenhada
  depois das caixas.
- **`append` sem parênteses:** um erro na leitura dos rótulos YOLO.
- **Contagens como "3,00":** o pandas convertia inteiros para float no relatório.
- **Singularidade da homografia:** a cena sintética gerava caixas da largura da imagem inteira.
- **Erro no meio do vídeo:** travava a interface num laço de erro.
- **VLM:** a classificação automática estava errada em cenas sem veículos.

**Dicas:**
- Deixe a demonstração ao vivo para o slide 9. Se algo falhar, mostre o vídeo de segurança (seção
  3).
- Nos slides 6 a 8, mostre **o trecho de código** de cada requisito de CG (estão em
  `docs/computacao_grafica.md`). É o que a disciplina avalia.
- Tenha prontas as respostas para "por que não só rodar o YOLO?" (README, seção Problema) e "o
  VLM substitui a contagem?" (não; slide 13).

---

## 3. Roteiro do vídeo de segurança (2–3 min)

É a gravação da demonstração, para usar se a demonstração ao vivo falhar (sem internet, projetor,
lentidão) ou para entregar junto com o trabalho. Grave a tela (OBS ou a Barra de Jogo do Windows,
`Win + G`) com o vídeo real e a configuração já salva. Narração em ritmo calmo.

| Tempo | Na tela | Narração (sugestão) |
|---|---|---|
| 0:00–0:15 | Título do app e o vídeo na prévia | "Este é o FluxoVC, que conta veículos numa via usando só uma câmera, sem sensores no asfalto." |
| 0:15–0:40 | Barra lateral: fonte "Vídeo de demonstração"; abrir **Linha**, **ROI** e **Calibração** e mostrar a prévia (ciano, amarelo, magenta) | "Escolhemos o vídeo e conferimos a configuração: a região de interesse em ciano, a linha de contagem em amarelo com os lados A e B, e os quatro pontos de calibração em magenta, que dão a escala em metros." |
| 0:40–0:55 | Abrir **Pré-processamento** e passar o mouse no ❓ do CLAHE | "Cada pré-processamento pode ser ligado e desligado e tem uma explicação. O letterbox mantém a proporção e a ROI ignora a calçada." |
| 0:55–1:30 | Clicar em **Analisar**. Mostrar as caixas, as trajetórias, a linha piscando e o HUD. Abrir a comparação original × pré-processado × resultado | "O YOLO detecta e o ByteTrack dá um número a cada veículo. Quando o centro da base cruza a linha, o veículo é contado uma vez, com o sentido. A linha pisca em verde a cada contagem." |
| 1:30–1:50 | **Durante a análise:** mudar o limiar de confiança e ligar o **mapa de calor** | "Os parâmetros mudam ao vivo, sem recomeçar. O mapa de calor mostra as faixas mais usadas." |
| 1:50–2:15 | Fim do vídeo: resumo por classe, gráfico acumulado, tabela de eventos com velocidade; clicar em **Eventos (CSV)** | "No fim temos a contagem por classe e sentido, a velocidade estimada pela homografia e os arquivos para baixar." |
| 2:15–2:35 | Abrir **Descrever cena com IA** e clicar no botão; mostrar a imagem anonimizada e o aviso | "Opcionalmente, um modelo de linguagem descreve a cena. Ele recebe a imagem anonimizada, com pessoas e placas pixelizadas, e a resposta vem marcada como gerada por IA." |
| 2:35–2:50 | Mostrar o erro tratado: escolher uma ROI com 2 pontos (erro na prévia) ou a fonte Webcam sem câmera | "Erros como uma ROI inválida ou a falta de câmera aparecem como mensagens claras, sem travar o programa." |
| 2:50–3:00 | README no GitHub | "O código, os testes e a documentação estão no repositório. Obrigado." |

**Antes de gravar:**
- Rode uma análise antes, para os pesos já estarem baixados e o VLM carregado.
- Use o YOLO11n, para o vídeo ficar fluido em CPU.
- Feche abas e notificações.
- Não mostre o `.env` nem caminhos com o seu nome de usuário.
