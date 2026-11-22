"""Interface Streamlit do FluxoVC.

Execute com:  streamlit run app.py

Este arquivo contém apenas a interface; toda a lógica fica em ``src/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import __version__
from src.acquisition.erros import ErroEntrada
from src.acquisition.fontes import (
    abrir_arquivo,
    abrir_url,
    abrir_webcam,
    decodificar_imagem,
    listar_videos_demo,
    salvar_upload,
)
from src.aplicacao.relatorios import (
    deteccoes_para_dataframe,
    eventos_para_dataframe,
    para_csv,
    resumo_contagem,
    serie_contagem,
)
from src.aplicacao.sessao import (
    SessaoAnalise,
    analisar_e_compor_imagem,
    codificar_png,
    previa_configuracao,
    primeiro_frame,
    validar_configuracao,
)
from src.config import AppConfig
from src.geometry.erros import ErroGeometria
from src.logs import configurar_logging
from src.models.analisador import MSG_SEM_DETECCOES
from src.models.contagem import SENTIDO_AB, SENTIDO_BA
from src.models.detector_yolo import DetectorYOLO, escolher_dispositivo
from src.models.tipos import Detector, ErroModelo
from src.preprocessing.pipeline import EXPLICACOES

configurar_logging()
logger = logging.getLogger("fluxovc.app")

st.set_page_config(page_title="FluxoVC", page_icon="🚦", layout="wide")

PADRAO = AppConfig()
MODELOS = {"yolo11n.pt": "YOLO11n — rápido", "yolo11s.pt": "YOLO11s — mais preciso, mais lento"}
FONTES = ["Vídeo de demonstração", "Enviar vídeo", "Webcam", "URL / câmera IP", "Imagem única"]
ATUALIZAR_PAINEIS_A_CADA = 10  # frames entre atualizações de gráfico/tabela/comparação
NOMES_CALIBRACAO = ["1 · sup. esquerdo", "2 · sup. direito", "3 · inf. direito", "4 · inf. esquerdo"]


@dataclass
class FonteEscolhida:
    tipo: str                    # "arquivo", "webcam", "url", "imagem"
    caminho: Path | None = None
    indice: int = 0
    url: str = ""
    imagem_bytes: bytes | None = None
    nome: str = ""


# ---------------------------------------------------------------------------
# Estado da sessão do navegador
# ---------------------------------------------------------------------------
def estado() -> Any:
    ss = st.session_state
    ss.setdefault("cfg", AppConfig())
    ss.setdefault("detectores", {})
    ss.setdefault("sessao", None)
    ss.setdefault("rodando", False)
    ss.setdefault("resultado_imagem", None)
    ss.setdefault("uploads", {})
    return ss


def limpar() -> None:
    ss = estado()
    parar()
    ss.sessao = None
    ss.resultado_imagem = None


def parar() -> None:
    ss = estado()
    ss.rodando = False
    if ss.sessao is not None:
        ss.sessao.finalizar()


def obter_detector(cfg: AppConfig) -> Detector:
    """Um detector por modelo, guardado na sessão do navegador (carregar o YOLO é caro)."""
    ss = estado()
    pesos = cfg.modelo.pesos
    if pesos not in ss.detectores:
        with st.spinner(f"Carregando {pesos}… (na primeira vez os pesos são baixados da internet)"):
            ss.detectores[pesos] = DetectorYOLO(cfg.modelo, cfg.preproc.tamanho_entrada)
    detector = ss.detectores[pesos]
    # parâmetros ao vivo: o detector passa a ler a configuração atual
    if hasattr(detector, "cfg"):
        detector.cfg = cfg.modelo
    if hasattr(detector, "tamanho_entrada"):
        detector.tamanho_entrada = cfg.preproc.tamanho_entrada
    return detector


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
def tabela_pontos(pontos: list[tuple[float, float]], indice: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(pontos, columns=["x", "y"], index=indice)


def pontos_de(tabela: pd.DataFrame) -> list[tuple[float, float]]:
    limpa = tabela.dropna()
    return [(float(x), float(y)) for x, y in zip(limpa["x"], limpa["y"])]


def escolher_fonte() -> FonteEscolhida | None:
    ss = estado()
    tipo = st.sidebar.radio("Fonte de entrada", FONTES, key="tipo_fonte")

    if tipo == "Vídeo de demonstração":
        videos = listar_videos_demo()
        if not videos:
            st.sidebar.info("Nenhum vídeo em `data_samples/`. Envie um vídeo ou use a webcam.")
            return None
        nome = st.sidebar.selectbox("Vídeo", [v.name for v in videos], key="video_demo")
        return FonteEscolhida("arquivo", caminho=next(v for v in videos if v.name == nome), nome=nome)

    if tipo == "Enviar vídeo":
        enviado = st.sidebar.file_uploader(
            "Vídeo (mp4, avi, mov)", type=["mp4", "avi", "mov"], key="upload_video"
        )
        if enviado is None:
            return None
        if enviado.file_id not in ss.uploads:
            try:
                ss.uploads[enviado.file_id] = salvar_upload(enviado.getvalue(), enviado.name, ss.cfg.aquisicao)
            except ErroEntrada as erro:
                st.sidebar.error(str(erro))
                return None
        return FonteEscolhida("arquivo", caminho=ss.uploads[enviado.file_id], nome=enviado.name)

    if tipo == "Webcam":
        indice = st.sidebar.number_input(
            "Índice da câmera", 0, 10, PADRAO.aquisicao.indice_webcam, key="indice_webcam",
            help="0 = câmera padrão. Tente 1, 2… se houver mais de uma.",
        )
        return FonteEscolhida("webcam", indice=int(indice), nome=f"webcam {indice}")

    if tipo == "URL / câmera IP":
        url = st.sidebar.text_input("Endereço (http, https, rtsp)", key="url_video")
        return FonteEscolhida("url", url=url, nome=url) if url.strip() else None

    imagem = st.sidebar.file_uploader("Imagem (jpg, png)", type=["jpg", "jpeg", "png"], key="upload_imagem")
    if imagem is None:
        return None
    return FonteEscolhida("imagem", imagem_bytes=imagem.getvalue(), nome=imagem.name)


def parametros(cfg: AppConfig) -> None:
    """Desenha os controles e grava os valores em ``cfg`` (a cada execução do script)."""
    sb = st.sidebar
    rodando = estado().rodando

    with sb.expander("🎯 Modelo", expanded=True):
        cfg.modelo.pesos = st.selectbox(
            "Modelo", list(MODELOS), format_func=MODELOS.get, key="pesos",
            help="Trocar o modelo vale a partir da próxima análise.",
        )
        cfg.modelo.confianca = st.slider(
            "Limiar de confiança", 0.05, 0.95, PADRAO.modelo.confianca, 0.05, key="confianca",
            help="Detecções abaixo deste valor são descartadas. Diminua se veículos não forem detectados.",
        )
        cfg.modelo.iou = st.slider(
            "Limiar de IoU (NMS)", 0.1, 0.9, PADRAO.modelo.iou, 0.05, key="iou",
            help="Caixas que se sobrepõem mais que isso são consideradas o mesmo veículo.",
        )
        st.caption(f"Dispositivo: **{'GPU (CUDA)' if escolher_dispositivo() != 'cpu' else 'CPU'}**")

    with sb.expander("🧪 Pré-processamento"):
        pre = cfg.preproc
        pre.manter_proporcao = st.toggle("Letterbox (manter proporção)", True, key="letterbox", help=EXPLICACOES["letterbox"])
        pre.usar_roi = st.toggle("Máscara de ROI", PADRAO.preproc.usar_roi, key="usar_roi", help=EXPLICACOES["roi"])
        pre.usar_clahe = st.toggle("CLAHE (contraste)", PADRAO.preproc.usar_clahe, key="usar_clahe", help=EXPLICACOES["clahe"])
        if pre.usar_clahe:
            pre.clahe_clip = st.slider("CLAHE · limite de contraste", 1.0, 8.0, PADRAO.preproc.clahe_clip, 0.5, key="clahe_clip")
            pre.clahe_grade = st.slider("CLAHE · grade (blocos)", 2, 16, PADRAO.preproc.clahe_grade, key="clahe_grade")
        pre.usar_reducao_ruido = st.toggle("Redução de ruído", PADRAO.preproc.usar_reducao_ruido, key="usar_ruido", help=EXPLICACOES["ruido"])
        if pre.usar_reducao_ruido:
            pre.tipo_filtro = st.radio("Filtro", ["bilateral", "gaussiano"], horizontal=True, key="filtro")
            pre.kernel_ruido = st.slider("Tamanho do kernel", 3, 15, PADRAO.preproc.kernel_ruido, 2, key="kernel")
        pre.frame_skip = st.slider("Frame skip", 0, 5, PADRAO.preproc.frame_skip, key="frame_skip", help=EXPLICACOES["frame_skip"])

    with sb.expander("📏 Linha de contagem"):
        c1, c2 = st.columns(2)
        x1 = c1.slider("A · x", 0.0, 1.0, PADRAO.contagem.linha_p1[0], 0.01, key="lx1", disabled=rodando)
        y1 = c2.slider("A · y", 0.0, 1.0, PADRAO.contagem.linha_p1[1], 0.01, key="ly1", disabled=rodando)
        x2 = c1.slider("B · x", 0.0, 1.0, PADRAO.contagem.linha_p2[0], 0.01, key="lx2", disabled=rodando)
        y2 = c2.slider("B · y", 0.0, 1.0, PADRAO.contagem.linha_p2[1], 0.01, key="ly2", disabled=rodando)
        cfg.contagem.linha_p1, cfg.contagem.linha_p2 = (x1, y1), (x2, y2)
        cfg.contagem.ponto_referencia = st.radio(
            "Ponto do veículo", ["base", "centro"], horizontal=True, key="ponto_ref", disabled=rodando,
            help="'base' = centro da base da caixa (contato com o chão), recomendado.",
        )
        st.caption("Coordenadas normalizadas (0 = esquerda/topo, 1 = direita/base).")

    with sb.expander("🔷 ROI (região de interesse)"):
        tabela = st.data_editor(
            tabela_pontos(PADRAO.geometria.roi), num_rows="dynamic", key="roi", disabled=rodando,
            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0, step=0.01) for c in "xy"},
        )
        cfg.geometria.roi = pontos_de(tabela)
        st.caption("Polígono com pelo menos 3 pontos, em coordenadas normalizadas.")

    with sb.expander("🚀 Calibração da velocidade"):
        tabela = st.data_editor(
            tabela_pontos(PADRAO.geometria.homografia_origem, NOMES_CALIBRACAO), num_rows="fixed",
            key="calibracao", disabled=rodando,
            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0, step=0.01) for c in "xy"},
        )
        cfg.geometria.homografia_origem = pontos_de(tabela)
        c1, c2 = st.columns(2)
        cfg.geometria.largura_pista_m = c1.number_input(
            "Largura real (m)", 0.5, 100.0, PADRAO.geometria.largura_pista_m, 0.5, key="larg_m", disabled=rodando
        )
        cfg.geometria.comprimento_pista_m = c2.number_input(
            "Comprimento real (m)", 0.5, 500.0, PADRAO.geometria.comprimento_pista_m, 0.5, key="comp_m", disabled=rodando
        )
        st.caption(
            "Marque 4 pontos de um trecho retangular da pista (ex.: faixas e marcas no asfalto) "
            "e informe as medidas reais. A velocidade depende dessa calibração."
        )
        cfg.geometria.ema_alpha = st.slider(
            "α da suavização (EMA)", 0.05, 1.0, PADRAO.geometria.ema_alpha, 0.05, key="alpha", disabled=rodando,
            help="Perto de 1: segue o ponto bruto. Perto de 0: suaviza mais, mas atrasa a trajetória.",
        )

    with sb.expander("🎨 Camadas"):
        vis = cfg.visualizacao
        c1, c2 = st.columns(2)
        vis.mostrar_caixas = c1.toggle("Caixas", PADRAO.visualizacao.mostrar_caixas, key="v_caixas")
        vis.mostrar_trajetorias = c2.toggle("Trajetórias", PADRAO.visualizacao.mostrar_trajetorias, key="v_traj")
        vis.mostrar_linha = c1.toggle("Linha", PADRAO.visualizacao.mostrar_linha, key="v_linha")
        vis.mostrar_roi = c2.toggle("ROI", PADRAO.visualizacao.mostrar_roi, key="v_roi")
        vis.mostrar_heatmap = c1.toggle("Mapa de calor", PADRAO.visualizacao.mostrar_heatmap, key="v_heat")
        vis.mostrar_birdseye = c2.toggle("Vista de cima", PADRAO.visualizacao.mostrar_birdseye, key="v_bird")
        vis.mostrar_hud = c1.toggle("Painel (HUD)", PADRAO.visualizacao.mostrar_hud, key="v_hud")
        vis.opacidade_heatmap = st.slider("Opacidade do mapa de calor", 0.0, 1.0, PADRAO.visualizacao.opacidade_heatmap, 0.05, key="op_heat")
        vis.opacidade_roi = st.slider("Opacidade da ROI", 0.0, 1.0, PADRAO.visualizacao.opacidade_roi, 0.05, key="op_roi")
        vis.colormap = st.radio("Colormap", ["TURBO", "JET"], horizontal=True, key="colormap")

    if rodando:
        sb.caption("🔒 Linha, ROI, calibração e α valem a partir da próxima análise.")


# ---------------------------------------------------------------------------
# Ações
# ---------------------------------------------------------------------------
def iniciar_analise(fonte: FonteEscolhida, cfg: AppConfig) -> None:
    ss = estado()
    parar()
    ss.sessao = None
    ss.resultado_imagem = None
    try:
        detector = obter_detector(cfg)
        if fonte.tipo == "imagem":
            imagem = decodificar_imagem(fonte.imagem_bytes or b"", fonte.nome, cfg.aquisicao)
            veiculos, preproc, final = analisar_e_compor_imagem(imagem, cfg, detector)
            ss.resultado_imagem = {"original": imagem, "preproc": preproc.imagem, "final": final,
                                   "veiculos": veiculos, "nome": fonte.nome}
            return
        if fonte.tipo == "arquivo":
            assert fonte.caminho is not None
            video = abrir_arquivo(fonte.caminho, cfg.aquisicao)
        elif fonte.tipo == "webcam":
            with st.spinner("Abrindo a câmera…"):
                video = abrir_webcam(fonte.indice, cfg.aquisicao)
        else:
            with st.spinner("Conectando ao vídeo…"):
                video = abrir_url(fonte.url, cfg.aquisicao)
        ss.sessao = SessaoAnalise(video, cfg, detector)
        ss.rodando = True
    except (ErroEntrada, ErroGeometria, ErroModelo) as erro:
        st.error(str(erro))
    except Exception:  # erro inesperado: registra e mostra mensagem genérica, sem travar
        logger.exception("Falha ao iniciar a análise")
        st.error("Erro inesperado ao iniciar a análise. Veja o terminal para detalhes.")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------
def metricas(sessao: SessaoAnalise, area: Any) -> None:
    contador = sessao.contador
    fps = sessao.compositor.medidor_fps.fps
    ultimo = sessao.ultimo_passo
    latencia = None if ultimo is None or not ultimo.resultado.processado else (
        ultimo.resultado.tempo_total_ms + sessao.compositor.tempo_desenho_ms
    )
    with area.container():
        colunas = st.columns(5)
        colunas[0].metric("Veículos contados", contador.total)
        colunas[1].metric(f"Sentido {SENTIDO_AB}", contador.total_por_sentido(SENTIDO_AB))
        colunas[2].metric(f"Sentido {SENTIDO_BA}", contador.total_por_sentido(SENTIDO_BA))
        colunas[3].metric("FPS", "—" if fps is None else f"{fps:.1f}")
        colunas[4].metric("Latência", "—" if latencia is None else f"{latencia:.0f} ms")


def paineis(sessao: SessaoAnalise, grafico: Any, tabela: Any, comparacao: list[Any]) -> None:
    serie = serie_contagem(sessao.eventos, sessao.duracao_processada_s)
    grafico.line_chart(serie, x_label="tempo (s)", y_label="veículos (acumulado)")
    tabela.dataframe(eventos_para_dataframe(sessao.eventos), hide_index=True, width="stretch")
    passo = sessao.ultimo_passo
    if passo is not None:
        comparacao[0].image(passo.frame.imagem, channels="BGR", caption="Original", width="stretch")
        if sessao.ultimo_preproc is not None:
            comparacao[1].image(sessao.ultimo_preproc.imagem, channels="BGR",
                                caption="Pré-processado (entrada do modelo)", width="stretch")
        comparacao[2].image(passo.imagem_final, channels="BGR", caption="Resultado", width="stretch")


def tela_video(sessao: SessaoAnalise) -> None:
    ss = estado()
    info = sessao.info
    duracao = f" · {info.duracao_s:.1f} s" if info.duracao_s else ""
    st.caption(
        f"**{info.origem}** · {info.largura}×{info.altura} · {info.fps:.1f} fps"
        f"{' (estimado)' if info.fps_estimado else ''}{duracao}"
    )
    for aviso in sessao.analisador.avisos:
        st.warning(aviso)

    area_progresso = st.empty()
    area_metricas = st.empty()
    area_video = st.empty()
    area_alerta = st.empty()
    with st.expander("🔍 Comparação: original × pré-processado × resultado"):
        comparacao = st.columns(3)
    st.subheader("Contagem ao longo do tempo")
    area_grafico = st.empty()
    st.subheader("Eventos de cruzamento")
    area_tabela = st.empty()

    def atualizar_tudo() -> None:
        metricas(sessao, area_metricas)
        paineis(sessao, area_grafico, area_tabela, comparacao)
        if sessao.progresso is not None:
            area_progresso.progress(sessao.progresso, text=f"Frame {sessao.frames_lidos} de {info.total_frames}")
        if sessao.analisador.sem_deteccoes and sessao.analisador.frames_processados >= 30:
            area_alerta.warning(MSG_SEM_DETECCOES)
        else:
            area_alerta.empty()

    if sessao.ultimo_passo is not None:
        area_video.image(sessao.ultimo_passo.imagem_final, channels="BGR", output_format="JPEG")
        atualizar_tudo()

    while ss.rodando:
        passo = sessao.passo()
        if passo is None:
            ss.rodando = False
            st.rerun()  # fim do vídeo: redesenha a barra lateral e mostra os resultados
        area_video.image(passo.imagem_final, channels="BGR", output_format="JPEG")
        if passo.resultado.eventos or sessao.frames_lidos % ATUALIZAR_PAINEIS_A_CADA == 0:
            atualizar_tudo()
        else:
            metricas(sessao, area_metricas)
    atualizar_tudo()

    if not ss.rodando:
        resultados_finais(sessao)


def resultados_finais(sessao: SessaoAnalise) -> None:
    st.divider()
    st.subheader("Resultado da análise")
    if sessao.analisador.sem_deteccoes:
        st.warning(MSG_SEM_DETECCOES)
    c1, c2 = st.columns([1, 1])
    resumo = resumo_contagem(sessao.contador)
    c1.dataframe(resumo, width="stretch")
    fps = sessao.fps_medio()
    tempos = sessao.tempos_dataframe()
    with c2:
        st.markdown(f"**Frames analisados:** {sessao.frames_lidos} · **FPS médio:** {'—' if fps is None else f'{fps:.1f}'}")
        if not tempos.empty:
            st.markdown("**Tempo médio por etapa (ms)**")
            st.dataframe(tempos.mean().round(1).rename("ms").to_frame(), width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Eventos (CSV)", para_csv(eventos_para_dataframe(sessao.eventos)),
                       "fluxovc_eventos.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ Resumo (CSV)", para_csv(resumo, indice=True),
                       "fluxovc_resumo.csv", "text/csv", width="stretch")
    if sessao.caminho_video is not None and sessao.caminho_video.exists():
        dados = sessao.caminho_video.read_bytes()
        d3.download_button("⬇️ Vídeo processado (MP4)", dados, sessao.caminho_video.name, "video/mp4", width="stretch")
        if sessao.video_no_navegador:
            with st.expander("▶️ Assistir ao vídeo processado"):
                st.video(dados)
    else:
        d3.caption("Vídeo processado indisponível.")


def tela_imagem(resultado: dict[str, Any]) -> None:
    st.caption(f"**{resultado['nome']}** · análise estática (sem rastreamento nem contagem)")
    c1, c2, c3 = st.columns(3)
    c1.image(resultado["original"], channels="BGR", caption="Original", width="stretch")
    c2.image(resultado["preproc"], channels="BGR", caption="Pré-processado (entrada do modelo)", width="stretch")
    c3.image(resultado["final"], channels="BGR", caption="Resultado", width="stretch")
    st.image(resultado["final"], channels="BGR")
    tabela = deteccoes_para_dataframe(resultado["veiculos"])
    if tabela.empty:
        st.warning(MSG_SEM_DETECCOES)
    else:
        st.dataframe(tabela.groupby("classe").size().rename("quantidade").to_frame(), width="content")
        st.dataframe(tabela, hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Imagem analisada (PNG)", codificar_png(resultado["final"]),
                       "fluxovc_resultado.png", "image/png", width="stretch")
    d2.download_button("⬇️ Detecções (CSV)", para_csv(tabela), "fluxovc_deteccoes.csv", "text/csv", width="stretch")


def tela_previa(fonte: FonteEscolhida | None, cfg: AppConfig) -> None:
    if fonte is None:
        st.info("👈 Escolha uma fonte de entrada na barra lateral e clique em **Analisar**.")
        return
    imagem = None
    try:
        if fonte.tipo == "arquivo" and fonte.caminho is not None:
            imagem = primeiro_frame(fonte.caminho, cfg)
        elif fonte.tipo == "imagem":
            imagem = decodificar_imagem(fonte.imagem_bytes or b"", fonte.nome, cfg.aquisicao)
    except ErroEntrada as erro:
        st.error(str(erro))
        return
    if imagem is None:
        st.info(f"Fonte selecionada: **{fonte.nome}**. Clique em **Analisar** para começar.")
        return
    altura, largura = imagem.shape[:2]
    erros, avisos = validar_configuracao(cfg, largura, altura)
    for erro in erros:
        st.error(erro)
    for aviso in avisos:
        st.warning(aviso)
    st.image(
        previa_configuracao(imagem, cfg), channels="BGR",
        caption="Prévia: ROI (ciano), linha de contagem (amarelo, lados A e B) e calibração (magenta, pontos 1–4)",
    )


# ---------------------------------------------------------------------------
def main() -> None:
    ss = estado()
    cfg: AppConfig = ss.cfg

    st.title("🚦 FluxoVC")
    st.caption("Monitor de fluxo de veículos em vias urbanas — detecção, rastreamento, contagem e velocidade")

    st.sidebar.header("Configurações")
    fonte = escolher_fonte()
    parametros(cfg)

    c1, c2 = st.sidebar.columns(2)
    analisar = c1.button("▶️ Analisar", type="primary", disabled=fonte is None, width="stretch", key="analisar")
    c2.button("⏹️ Parar", on_click=parar, disabled=not ss.rodando, width="stretch", key="parar")
    if not ss.rodando and (ss.sessao is not None or ss.resultado_imagem is not None):
        st.sidebar.button("🗑️ Limpar resultado", on_click=limpar, width="stretch", key="limpar")
    st.sidebar.caption(f"FluxoVC {__version__} · YOLO11 (Ultralytics, AGPL-3.0) + ByteTrack")

    if analisar and fonte is not None:
        iniciar_analise(fonte, cfg)
        if ss.rodando or ss.resultado_imagem is not None:
            st.rerun()  # redesenha a barra lateral com "Parar" habilitado

    if ss.resultado_imagem is not None and not ss.rodando:
        tela_imagem(ss.resultado_imagem)
    elif ss.sessao is not None:
        tela_video(ss.sessao)
    else:
        tela_previa(fonte, cfg)

    with st.expander("ℹ️ Sobre, privacidade e limitações"):
        st.markdown(
            "- O FluxoVC detecta **apenas veículos** (carro, moto, ônibus, caminhão); pessoas são ignoradas "
            "e nenhuma placa é lida ou salva.\n"
            "- A **velocidade é aproximada** e depende da calibração (4 pontos + medidas reais).\n"
            "- Oclusões, noite, chuva e câmera tremida reduzem a precisão.\n"
            "- Vídeos enviados ficam numa pasta temporária do computador; resultados em `outputs/`."
        )


main()
