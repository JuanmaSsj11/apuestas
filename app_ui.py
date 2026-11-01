"""
Terminal Cuantitativa de Inversion Deportiva [+EV Institutional Edition]
Clinical high-contrast quantitative interface (#0f172a Slate, #059669 Emerald, #0284c7 Sky, #f8fafc Canvas).
Includes:
- Lo Mejor del Dia (MVP): Priority #1 picks across all sports today, featuring Cross-Sport 5-Leg Parlay Ratonero
  OR automatic fallback to 2 Apuestas Fuertes when Ratonero < 5 legs, plus Kalshi CFTC & Action RLM Monitor.
- NCAAF (Futbol Americano Colegial FBS): Strictly separated into 3 independent Horarios (Manana, Tarde, Noche) ->
  Each Horario renders its own unmixed 2 Apuestas Fuertes + 2 Parlays Ganadores (Zero Ratonero, Zero Props).
- Tenis (ATP & WTA Singles): Surface-Adjusted Elo + Hold/Break/DR/TB Sabermetrics + 10,000 Markov Set Sims ->
  Apuesta Fuerte + Parlay Ganador (Zero Ratonero, Zero Props).
- Futbol Expanded: 5 Big European Leagues + Domestic Cups + UEFA + Liga MX (15 competitions).
- NBA, NHL, NFL, MLB + Comparative Sport & Date Historical Audit.
Strictly zero emojis.
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import run_sports_pipeline
from src.ai_agent import QuantitativeAIAgent, GLOSARIO_INDICADORES
from src.history_tracker import HistoryTracker
from src.predictive_models import american_to_decimal, decimal_to_american
from src.playdoit_scraper import PlaydoitScraper

st.set_page_config(
    page_title="Terminal Cuantitativa Deportiva [+EV]",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_clean_html(html_str: str):
    """
    Renders pure HTML without leading whitespace per line,
    strictly preventing Markdown from treating indented HTML blocks as code blocks.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.markdown(" ".join(clean_lines), unsafe_allow_html=True)


# ==============================================================================
# CLINICAL QUANTITATIVE TERMINAL CSS TOKENS & COMPONENT STYLES
# ==============================================================================
render_clean_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --qt-obsidian: #0f172a;
    --qt-slate-dark: #1e293b;
    --qt-slate: #334155;
    --qt-body: #475569;
    --qt-muted: #64748b;
    --qt-emerald: #059669;
    --qt-sky: #0284c7;
    --qt-amber: #d97706;
    --qt-crimson: #dc2626;
    --qt-canvas: #f8fafc;
    --qt-card: #ffffff;
    --qt-border: #e2e8f0;
    --qt-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--qt-obsidian);
}

.stApp {
    background-color: var(--qt-canvas);
}

.qt-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 26px;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    margin-bottom: 16px;
}

.qt-hero-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 4px;
}

.qt-hero-sub {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.45;
}

.qt-hero-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: #f1f5f9;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
    margin-right: 6px;
    margin-top: 10px;
}

.qt-card {
    background-color: #ffffff;
    border: 1px solid var(--qt-border);
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: var(--qt-shadow);
    margin-bottom: 14px;
}

.qt-kpi-card {
    background-color: #ffffff;
    border: 1px solid var(--qt-border);
    border-top: 3px solid var(--qt-obsidian);
    border-radius: 10px;
    padding: 13px 16px;
    box-shadow: var(--qt-shadow);
    margin-bottom: 12px;
}

.qt-label {
    font-size: 10.5px;
    text-transform: uppercase;
    color: var(--qt-muted);
    font-weight: 700;
    letter-spacing: 0.07em;
    margin-bottom: 3px;
}

.qt-value {
    font-size: 22px;
    font-weight: 800;
    color: var(--qt-obsidian);
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.badge-won {
    display: inline-block;
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 6px;
}

.badge-lost {
    display: inline-block;
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 6px;
}

.badge-inplay {
    display: inline-block;
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 6px;
}

.badge-upcoming {
    display: inline-block;
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 6px;
}

.qt-pill-dark {
    display: inline-block;
    background-color: #0f172a;
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 5px;
}

.qt-pill-slate {
    display: inline-block;
    background-color: #e2e8f0;
    color: #1e293b;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 5px;
}

.qt-pill-sky {
    display: inline-block;
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 5px;
}

.qt-pill-emerald {
    display: inline-block;
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 5px;
}

.qt-pill-amber {
    display: inline-block;
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 5px;
}

.stButton > button {
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 8px 18px !important;
    box-shadow: var(--qt-shadow) !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] {
    background-color: #0f172a !important;
    border-color: #0f172a !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #ffffff;
    border: 1px solid var(--qt-border);
    border-radius: 9px;
    padding: 5px 7px;
    box-shadow: var(--qt-shadow);
    margin-bottom: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 6px 13px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--qt-body) !important;
    background-color: transparent !important;
}

.stTabs [aria-selected="true"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
}

.stExpander {
    border-radius: 10px !important;
    border: 1px solid var(--qt-border) !important;
    background-color: #ffffff !important;
    box-shadow: var(--qt-shadow) !important;
    margin-top: 6px !important;
    margin-bottom: 12px !important;
}
</style>
""")

history = HistoryTracker()
CST_TZ = ZoneInfo("America/Mexico_City")
now_cst = datetime.now(CST_TZ)
today_str = (now_cst + timedelta(days=1)).strftime("%Y-%m-%d") if now_cst.hour >= 21 else now_cst.strftime("%Y-%m-%d")

# ==============================================================================
# EXECUTIVE QUANTITATIVE TERMINAL HEADER
# ==============================================================================
render_clean_html(f"""
<div class="qt-hero">
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; gap:12px;">
        <div>
            <div class="qt-hero-title">TERMINAL CUANTITATIVA DE INVERSION DEPORTIVA [+EV]</div>
            <div class="qt-hero-sub">
                Ensamble Bayesiano de 4 Modelos + Sabermetria en Vivo + Mercados Kalshi CFTC + Flujo Sharp/RLM + Consenso Analistas + 10,000 Simulaciones Monte Carlo
            </div>
            <div>
                <span class="qt-hero-pill">7 FILTROS INSTITUCIONALES ACTIVOS</span>
                <span class="qt-hero-pill">KALSHI CFTC + ACTION SHARP MONEY &amp; RLM</span>
                <span class="qt-hero-pill">NCAAF SEPARADO EN 3 HORARIOS (MANANA / TARDE / NOCHE)</span>
                <span class="qt-hero-pill">TENIS ATP/WTA + FUTBOL 15 LIGAS Y COPAS</span>
            </div>
        </div>
        <div style="text-align:right; background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.16); border-radius:8px; padding:10px 16px;">
            <div style="font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:#94a3b8; font-weight:700;">FECHA OPERATIVA</div>
            <div style="font-size:18px; font-weight:800; color:#ffffff;">{today_str}</div>
            <div style="font-size:11px; color:#34d399; font-weight:700;">AUDITORIA ESPN ACTIVA</div>
        </div>
    </div>
</div>
""")

# Action Bar
col_act1, col_act2, col_status = st.columns([2.1, 2.1, 2.3])
with col_act1:
    if st.button("ANALIZAR Y SINCRONIZAR CARTELERA COMPLETA", type="primary", use_container_width=True):
        with st.spinner("Ejecutando scraping ESPN, Kalshi CFTC, Action Sharp/RLM, consenso experto y 10,000 simulaciones Monte Carlo..."):
            st.session_state.results = run_sports_pipeline(record_in_history=True)
            st.session_state.pipeline_date = today_str
            st.session_state.scorecards = history.sync_and_grade_all_tickets()
            st.toast("Cartelera multi-deporte y modelos cuantitativos actualizados.")
            st.rerun()

with col_act2:
    if st.button("ACTUALIZAR MARCADORES Y LIQUIDAR TICKETS", use_container_width=True):
        with st.spinner("Consultando boxscores oficiales ESPN y verificando tickets..."):
            st.session_state.scorecards = history.sync_and_grade_all_tickets()
            st.session_state.results = run_sports_pipeline(record_in_history=True)
            st.session_state.pipeline_date = today_str
            st.toast("Marcadores oficiales ESPN sincronizados.")
            st.rerun()

# Initial Load
if "results" not in st.session_state or st.session_state.get("pipeline_date") != today_str:
    with st.spinner("Inicializando Terminal Cuantitativa y modelos sabermetricos de 4 pilares..."):
        st.session_state.results = run_sports_pipeline(record_in_history=True)
        st.session_state.pipeline_date = today_str

res = st.session_state.results
today_str = res.get("active_date", today_str)
scorecards = res.get("scorecards") or history.sync_and_grade_all_tickets()

with col_status:
    gen_sc = scorecards["sports"]["general"]
    g_pnl = gen_sc.get("units_pnl", 0.0)
    g_col = "#047857" if g_pnl >= 0 else "#b91c1c"
    render_clean_html(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #0f172a; border-radius:8px; padding:8px 14px; font-size:12px; box-shadow:0 1px 2px rgba(15,23,42,0.05); text-align:right;">
        <b>Tickets Registrados:</b> {gen_sc.get('total_bets', 0)} &nbsp;|&nbsp;
        <b>Win Rate:</b> {gen_sc.get('win_rate_pct', 0.0):.1f}% &nbsp;|&nbsp;
        <b>Balance:</b> <span style="color:{g_col}; font-weight:800;">{g_pnl:+.2f} u</span>
    </div>
    """)

# Utility Tools Row
u_col1, u_col2, u_col3 = st.columns(3)
with u_col1:
    with st.expander("[CALIBRADOR PLAYDOIT (PD)] Calculadora +EV Real", expanded=False):
        calc_prob = st.slider("Probabilidad Real del Modelo (%):", min_value=50.0, max_value=95.0, value=60.0, step=0.5)
        user_pd_odds = st.number_input("Momio Americano en Playdoit:", value=-115, step=5)
        pd_result = PlaydoitScraper.calculate_playdoit_ev(calc_prob, user_pd_odds)
        pd_p_ev = pd_result["ev_pct"]
        pd_color = "#047857" if pd_p_ev > 0 else "#b91c1c"
        pd_bg = "#ecfdf5" if pd_p_ev > 0 else "#fef2f2"
        req_am = decimal_to_american(1.0 / (calc_prob / 100.0))
        render_clean_html(f"""
        <div style="background:{pd_bg}; border-radius:8px; padding:10px; text-align:center;">
            <div style="font-size:10.5px; font-weight:700; color:{pd_color};">{pd_result['status_text']}</div>
            <div style="font-size:21px; font-weight:800; color:{pd_color};">{pd_p_ev:+.2f}% EV</div>
            <div style="font-size:11px; color:#475569;">Momio Limite Rentable: <b>{req_am:+d}</b></div>
        </div>
        """)

with u_col2:
    ai_agent_data = res.get("ai_agent", {})
    weights = ai_agent_data.get("model_weights", {})
    ic = ai_agent_data.get("intel_counts", {})
    with st.expander("[MOTOR 4 MODELOS & INTELIGENCIA] Fuentes y 7 Filtros Activos", expanded=False):
        render_clean_html(f"""
        <div style="font-size:11.5px; color:#475569; line-height:1.55;">
            <div><b>Estado:</b> <span style="color:#059669; font-weight:700;">{ai_agent_data.get('status', 'ACTIVO')}</span></div>
            <div style="margin-top:4px;">
                <b>Scraping Vivo:</b> Kalshi CFTC ({ic.get('kalshi_contracts', 0)} contratos) &bull;
                Action Sharp/RLM ({ic.get('action_sharp_games', 0)} juegos) &bull;
                Consenso Analistas/Foros ({ic.get('expert_articles', 0)} reportes)
            </div>
            <div style="margin-top:5px; color:#1e293b;">
                <b>7 Filtros:</b> (1) Consenso &ge;3/4 Modelos &bull; (2) Contraccion Bayesiana (+2% a +28% EV real) &bull;
                (3) Anti-Trampa Publica Sharp/RLM &bull; (4) Cobertura Inteligente Underdog &bull;
                (5) Zona Muerta en Totales &bull; (6) Bloqueo Global Anti-Contradiccion &bull; (7) Auditoria Exacta Fecha/Rival.
            </div>
        </div>
        """)

with u_col3:
    with st.expander("[ACCESO MOVIL] Abrir Terminal en Celular", expanded=False):
        render_clean_html("""
        <div style="font-size:11.5px; color:#475569; line-height:1.5;">
            Conecta tu celular a la misma red Wi-Fi y abre la IP local de tu equipo en el puerto <code>8501</code>
            (ej. <code>http://192.168.1.84:8501</code>). En Chrome/Safari selecciona <b>Agregar a pantalla de inicio</b>
            para vista completa tipo aplicacion nativa.
        </div>
        """)

# ==============================================================================
# COMPACT LEGEND RIBBON + INTERACTIVE INDICATOR GLOSSARY
# ==============================================================================
render_clean_html("""
<div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #0284c7; border-radius:9px; padding:9px 15px; margin-top:2px; margin-bottom:8px; font-size:11.5px; color:#334155; line-height:1.6;">
    <b style="color:#0f172a; text-transform:uppercase; letter-spacing:0.04em;">GUIA RAPIDA DE ETIQUETAS:</b> &nbsp;
    <span class="qt-pill-amber" title="Movimiento Inverso de Linea: el publico apuesta al otro lado pero el dinero pesado profesional movio la linea a favor de este equipo.">RLM CONFIRMADO</span> = Dinero profesional moviendo linea contra el publico &nbsp;|&nbsp;
    <span class="qt-pill-emerald" title="Cuantos de los 4 motores independientes coinciden: (1) Sabermetria, (2) Matchup Situacional, (3) Kalshi/Sharp y (4) Consenso/10k Monte Carlo.">CONSENSO 3/4 O 4/4</span> = Motores independientes a favor &nbsp;|&nbsp;
    <span class="qt-pill-sky" title="Precio real en bolsa regulada Kalshi (CFTC EE. UU.) y porcentaje del dinero total apostado.">KALSHI / SHARP $</span> = Precio CFTC y % dinero grande &nbsp;|&nbsp;
    <span class="qt-pill-dark" title="Calificacion inteligente del 1 al 10 que penaliza apuestas metidas por cuota en horarios cortos o parlays de muchas patas.">SEGURIDAD 1-10</span> = <b>8.5-10</b> Elite &bull; <b>7.2-8.4</b> Alta &bull; <b>5.8-7.1</b> Moderada / Complemento &bull; <b>&lt;5.8</b> Varianza Alta
</div>
""")

with st.expander("[LEYENDA DETALLADA Y GLOSARIO] Significado de RLM Confirmado, Consenso 3/4, Kalshi, Sharp, Probabilidad Real y Seguridad (1-10)", expanded=False):
    g_cols = st.columns(2)
    for g_idx, g_item in enumerate(GLOSARIO_INDICADORES):
        with g_cols[g_idx % 2]:
            render_clean_html(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px 13px; margin-bottom:8px;">
                <div style="margin-bottom:4px;">
                    <span style="display:inline-block; background:{g_item['bg']}; color:{g_item['color']}; border:1px solid {g_item['border']}; font-size:10px; font-weight:800; padding:2px 8px; border-radius:5px; letter-spacing:0.05em;">{g_item['badge']}</span>
                    <b style="font-size:12px; color:#0f172a; margin-left:6px;">{g_item['titulo']}</b>
                </div>
                <div style="font-size:11.5px; color:#475569; line-height:1.5;">{g_item['explicacion']}</div>
            </div>
            """)


# ==============================================================================
# REUSABLE CLINICAL UI COMPONENTS & INTELLIGENT 1-10 SECURITY METER
# ==============================================================================
def get_badge_html(status: str) -> str:
    s = (status or "").upper()
    if s == "GANADA":
        return '<span class="badge-won">GANADA</span>'
    if s == "PERDIDA":
        return '<span class="badge-lost">PERDIDA</span>'
    if s == "EN JUEGO":
        return '<span class="badge-inplay">EN JUEGO</span>'
    return '<span class="badge-upcoming">PROXIMAMENTE</span>'


def build_context_pills(item: Dict[str, Any], default_sport: str = "", sec_audit: Dict[str, Any] = None) -> str:
    pills = []
    sp = item.get("sport") or default_sport
    if sp:
        pills.append(f'<span class="qt-pill-dark">{sp}</span>')
    if item.get("horario"):
        k_str = f" | {item['kickoff_cst']}" if item.get("kickoff_cst") else ""
        pills.append(f'<span class="qt-pill-slate" title="Turno aislado de la jornada colegial FBS.">HORARIO {item["horario"].upper()}{k_str}</span>')
    elif item.get("kickoff_cst"):
        pills.append(f'<span class="qt-pill-slate">{item["kickoff_cst"]}</span>')
    if item.get("league_name"):
        pills.append(f'<span class="qt-pill-slate">{item["league_name"]}</span>')
    if item.get("tournament"):
        pills.append(f'<span class="qt-pill-slate">{item["tournament"]}</span>')
    if item.get("surface"):
        pills.append(f'<span class="qt-pill-sky" title="Superficie oficial del torneo utilizada para el Elo especifico.">{item["surface"]}</span>')

    cons_lbl = item.get("consensus_label")
    if not cons_lbl and sec_audit and sec_audit.get("consensus"):
        cons_lbl = f"Consenso {sec_audit['consensus']}/{sec_audit.get('total_models', 4)} Modelos"
    if cons_lbl:
        pills.append(
            f'<span class="qt-pill-emerald" title="Acuerdo entre los 4 motores: (1) Sabermetria, (2) Matchup Tactico, (3) Kalshi/Sharp y (4) Consenso/10k Monte Carlo.">{cons_lbl}</span>'
        )
    if item.get("kalshi_prob") and float(item.get("kalshi_prob", 0)) > 0:
        kp = float(item["kalshi_prob"])
        pills.append(
            f'<span class="qt-pill-sky" title="Probabilidad cotizada con dinero real en la bolsa regulada Kalshi (CFTC EE. UU.).">KALSHI {kp:.0f}%</span>'
        )
    if item.get("sharp_money_pct") and float(item.get("sharp_money_pct", 0)) >= 56:
        sm = int(item["sharp_money_pct"])
        tk_txt = f" vs {int(item['tickets_pct'])}% tickets" if item.get("tickets_pct") is not None else ""
        pills.append(
            f'<span class="qt-pill-sky" title="Porcentaje del volumen total de dinero apostado en este lado ({sm}% dinero{tk_txt}).">SHARP {sm}% DINERO</span>'
        )
    if item.get("is_rlm"):
        pills.append(
            '<span class="qt-pill-amber" title="Reverse Line Movement: la linea se movio a favor de esta seleccion pese a tener minoria de tickets publicos (entrada de dinero profesional).">RLM CONFIRMADO</span>'
        )
    if sec_audit:
        sc_val = float(sec_audit.get("seguridad_1_10", 7.0))
        sc_bg = sec_audit.get("security_bg", "#f1f5f9")
        sc_col = sec_audit.get("security_color", "#0f172a")
        sc_brd = sec_audit.get("security_border", "#cbd5e1")
        sc_org = sec_audit.get("origin_tag", "")
        pills.append(
            f'<span style="display:inline-block; background:{sc_bg}; color:{sc_col}; border:1px solid {sc_brd}; font-size:10px; font-weight:800; letter-spacing:0.05em; text-transform:uppercase; padding:2px 8px; border-radius:5px; margin-right:5px;" title="{sc_org}">SEGURIDAD {sc_val:.1f}/10</span>'
        )
    return "".join(pills)


def build_security_meter_html(ticket_audit: Dict[str, Any], is_parlay: bool = False, leg_count: int = 1) -> str:
    """Renders the visual 1-10 Intelligent Security Meter, Calibrated Hit Probability, Stake & Anti-Filler Diagnosis."""
    t_score = float(ticket_audit.get("ticket_seguridad_1_10", 7.0))
    t_prob = float(ticket_audit.get("ticket_prob_suceda_pct", 60.0))
    bar_pct = max(8.0, min(100.0, t_score * 10.0))
    t_col = ticket_audit.get("color", "#0369a1")
    t_bg = ticket_audit.get("bg", "#f8fafc")
    t_brd = ticket_audit.get("border", "#cbd5e1")
    bar_col = ticket_audit.get("bar_color", "#0284c7")
    tier_lbl = ticket_audit.get("tier_label", "SEGURIDAD VERIFICADA")
    origin_tag = ticket_audit.get("origin_tag", "SELECCION DEL MOTOR")
    stake_lbl = ticket_audit.get("stake_label", "Stake Estandar: 1.0u")
    dictamen = ticket_audit.get("dictamen_inteligente", "")
    avg_leg = float(ticket_audit.get("avg_leg_seguridad_1_10", t_score))

    prob_caption = f"PROB. REAL DE ACERTAR LAS {leg_count} PATAS JUNTAS" if is_parlay else "PROB. REAL DE QUE SUCEDA (MOTOR)"
    extra_parlay_meta = (
        f' &nbsp;|&nbsp; <span><b>Seguridad Media por Pata:</b> <b style="color:#0f172a;">{avg_leg:.1f}/10</b></span>'
        if is_parlay
        else ""
    )

    return f"""
    <div style="background:{t_bg}; border:1px solid {t_brd}; border-radius:8px; padding:10px 13px; margin-top:8px; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
                <span style="font-size:10.5px; font-weight:800; color:{t_col}; letter-spacing:0.05em; text-transform:uppercase;">INDICE DE SEGURIDAD DEL MOTOR:</span>
                <span style="font-size:16px; font-weight:800; color:{t_col}; margin-left:5px;">{t_score:.1f} / 10</span>
                <span style="font-size:11px; font-weight:700; color:{t_col}; margin-left:6px;">({tier_lbl})</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:10.5px; font-weight:800; color:#334155; letter-spacing:0.04em;">{prob_caption}:</span>
                <span style="font-size:15.5px; font-weight:800; color:#0f172a; margin-left:5px;">{t_prob:.1f}%</span>
            </div>
        </div>
        <div style="width:100%; height:7px; background:#e2e8f0; border-radius:4px; overflow:hidden; margin-top:6px; margin-bottom:6px;">
            <div style="width:{bar_pct:.1f}%; height:100%; background:{bar_col}; border-radius:4px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; font-size:11px; color:#1e293b;">
            <span><b>Rigor de Seleccion:</b> <span style="font-weight:700; color:{t_col};">{origin_tag}</span>{extra_parlay_meta}</span>
            <span><b>Gestion de Capital:</b> <span style="font-weight:800; color:#0f172a;">{stake_lbl}</span></span>
        </div>
        <div style="font-size:11px; color:#334155; margin-top:5px; line-height:1.45; border-top:1px dashed {t_brd}; padding-top:5px;">
            {dictamen}
        </div>
    </div>
    """


def render_scorecard_ribbon(sc: dict, title: str):
    wr = sc.get("win_rate_pct", sc.get("win_rate", 0.0))
    pnl = sc.get("units_pnl", sc.get("total_pnl", 0.0))
    roi = sc.get("roi_pct", sc.get("roi", 0.0))
    pnl_color = "#047857" if pnl >= 0 else "#b91c1c"
    render_clean_html(f"""
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; background:#ffffff; border:1px solid #e2e8f0; border-left:5px solid #0f172a; border-radius:10px; padding:11px 18px; margin-bottom:14px; font-size:12.5px; box-shadow:0 1px 3px rgba(15,23,42,0.05);">
        <span style="font-weight:800; color:#0f172a; text-transform:uppercase; letter-spacing:0.05em;">MARCADOR {title.upper()}</span>
        <span><b>Tickets:</b> {sc.get('total_bets', 0)}</span>
        <span><b>Ganadas:</b> <span style="color:#047857; font-weight:800;">{sc.get('won', 0)}</span></span>
        <span><b>Perdidas:</b> <span style="color:#b91c1c; font-weight:800;">{sc.get('lost', 0)}</span></span>
        <span><b>En Juego:</b> <span style="color:#b45309; font-weight:800;">{sc.get('in_play', 0)}</span></span>
        <span><b>Proximamente:</b> <span style="color:#334155; font-weight:800;">{sc.get('upcoming', 0)}</span></span>
        <span><b>Win Rate:</b> <span style="font-weight:800; color:#0f172a;">{wr:.1f}%</span></span>
        <span><b>ROI:</b> <span style="font-weight:800; color:{pnl_color};">{roi:+.1f}%</span></span>
        <span><b>PnL Neto:</b> <span style="font-weight:800; color:{pnl_color};">{pnl:+.2f} u</span></span>
    </div>
    """)


def render_sniper_card(
    sniper: Any,
    title_label: str = "APUESTA FUERTE",
    sport_name: str = "",
    accent_color: str = "#0f172a",
    slate_size: int = 10,
):
    if sniper and isinstance(sniper, dict) and sniper.get("selection"):
        sn_sel = sniper.get("selection", "N/A")
        raw_odds = int(sniper.get("american_odds", -110))
        sn_odds = f"{raw_odds:+d}"
        sn_dec = float(sniper.get("decimal_odds") or american_to_decimal(raw_odds))
        sn_prob = float(sniper.get("real_prob", 58.0))
        sn_ev = float(sniper.get("ev", 7.5))

        eff_slate = int(sniper.get("slate_size") or slate_size or 10)
        eff_rank = int(sniper.get("slot_rank") or (2 if "#2" in title_label else 1))
        t_audit = QuantitativeAIAgent.audit_ticket_security(
            [sniper], vehicle_type="Apuesta Fuerte", slate_size=eff_slate, slot_rank=eff_rank
        )
        leg_sec = t_audit["leg_audits"][0]
        prob_suceda = t_audit["ticket_prob_suceda_pct"]
        seg_score = t_audit["ticket_seguridad_1_10"]

        pd_odds = PlaydoitScraper.adjust_odds_to_playdoit_margin(raw_odds)
        pd_calc = PlaydoitScraper.calculate_playdoit_ev(prob_suceda, pd_odds)
        pd_ev_val = pd_calc["ev_pct"]
        pd_color = "#047857" if pd_ev_val > 0 else "#b91c1c"

        sn_status = sniper.get("status", "PROXIMAMENTE")
        sn_score = sniper.get("score_result", "")
        pills_html = build_context_pills(sniper, sport_name, sec_audit=leg_sec)
        badge_html = get_badge_html(sn_status)
        meter_html = build_security_meter_html(t_audit, is_parlay=False, leg_count=1)
        filler_tag = " [COMPLEMENTO]" if t_audit.get("is_quota_filler") else ""

        with st.expander(
            f"[{title_label}] {sn_sel} | {sn_odds} | Prob. Exito: {prob_suceda:.1f}% | Seguridad: {seg_score:.1f}/10{filler_tag}",
            expanded=True,
        ):
            render_clean_html(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:5px solid {accent_color}; border-radius:9px; padding:14px 16px;">
                <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:6px; margin-bottom:8px;">
                    <div>{pills_html}</div>
                    <div>{badge_html}</div>
                </div>
                <div style="font-size:16px; font-weight:800; color:#0f172a;">{sn_sel}</div>
                <div style="font-size:12px; color:#475569; margin-top:4px;">
                    <b>Partido:</b> {sniper.get('game', '-')} &nbsp;|&nbsp;
                    <b>Mercado:</b> {sniper.get('market_type', 'Principal')}
                </div>
                {meter_html}
                <div style="display:flex; flex-wrap:wrap; gap:14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:7px; padding:8px 12px; margin-top:6px; font-size:12px;">
                    <span><b>Momio Base:</b> <span style="color:#0f172a; font-weight:800;">{sn_odds}</span> ({sn_dec:.2f})</span>
                    <span><b>Prob. Ensamble (4 Modelos):</b> <span style="color:#0284c7; font-weight:800;">{sn_prob:.1f}%</span></span>
                    <span><b>Prob. de que Suceda:</b> <span style="color:#0f172a; font-weight:800;">{prob_suceda:.1f}%</span></span>
                    <span><b>Valor Esperado:</b> <span style="color:#047857; font-weight:800;">{sn_ev:+.1f}% EV</span></span>
                </div>
                <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:7px; padding:7px 12px; margin-top:7px; font-size:11.5px; color:#1e293b;">
                    <b>Referencia Playdoit (PD):</b> <span style="font-weight:800;">{pd_odds:+d}</span> &nbsp;|&nbsp;
                    <b>EV en PD:</b> <span style="font-weight:800; color:{pd_color};">{pd_ev_val:+.1f}%</span>
                    <span style="color:#64748b;"> (Ajustado a vig local Altenar 6.5%)</span>
                </div>
                {f'<div style="font-size:11.5px; color:#1e293b; margin-top:6px;"><b>Marcador / Estado ESPN:</b> {sn_score}</div>' if sn_score else ''}
                <div style="font-size:12px; color:#334155; margin-top:9px; line-height:1.55; border-top:1px solid #e2e8f0; padding-top:8px;">
                    <b>Fundamento Cuantitativo, Sharp y Sabermetrico:</b> {sniper.get('justification', '-')}
                </div>
            </div>
            """)
    else:
        with st.expander(f"[{title_label}] Sin seleccion activa ({sport_name})", expanded=False):
            st.caption(f"Ningun partido en {sport_name} supero simultaneamente los 7 filtros institucionales de +EV y consenso.")


def render_ganador_card(
    ganador: List[Dict[str, Any]],
    title_label: str = "PARLAY GANADOR",
    sport_name: str = "",
    slate_size: int = 10,
):
    if ganador and isinstance(ganador, list) and len(ganador) > 0:
        comb_dec = 1.0
        for g in ganador:
            d_val = float(g.get("decimal_odds") or american_to_decimal(int(g.get("american_odds", -240))))
            comb_dec *= d_val
        comb_odds = decimal_to_american(comb_dec)

        eff_slate = int(ganador[0].get("slate_size") or slate_size or 10)
        eff_rank = int(ganador[0].get("slot_rank") or (2 if "#2" in title_label else 1))
        t_audit = QuantitativeAIAgent.audit_ticket_security(
            ganador, vehicle_type="Parlay Ganador", slate_size=eff_slate, slot_rank=eff_rank
        )
        joint_prob = t_audit["ticket_prob_suceda_pct"]
        seg_score = t_audit["ticket_seguridad_1_10"]
        filler_tag = " [COMPLEMENTO]" if t_audit.get("is_quota_filler") else ""
        meter_html = build_security_meter_html(t_audit, is_parlay=True, leg_count=len(ganador))

        with st.expander(
            f"[{title_label}] {len(ganador)} Patas | Momio {comb_odds:+d} (<= +100 MAX) | Prob. Exito: {joint_prob:.1f}% | Seguridad: {seg_score:.1f}/10{filler_tag}",
            expanded=True,
        ):
            render_clean_html(f"""
            <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:7px; padding:7px 12px; margin-bottom:6px; font-size:11.5px; color:#0f172a; display:flex; justify-content:space-between; flex-wrap:wrap;">
                <span><b>Momio Combinado:</b> {comb_odds:+d} ({comb_dec:.2f}x)</span>
                <span><b>Prob. Real de Acertar Ambas:</b> {joint_prob:.1f}%</span>
                <span><b>Seguridad Ticket:</b> {seg_score:.1f}/10</span>
                <span><b>Regla:</b> &le; +100 MAX / Sin Doble Under</span>
            </div>
            {meter_html}
            """)
            for idx, (g, l_sec) in enumerate(zip(ganador, t_audit["leg_audits"]), 1):
                p_name = g.get("game") or g.get("game_str") or "-"
                g_stat = g.get("status", "PROXIMAMENTE")
                g_sc = g.get("score_result", "")
                pills_html = build_context_pills(g, sport_name, sec_audit=l_sec)
                b_html = get_badge_html(g_stat)
                render_clean_html(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #059669; border-radius:8px; padding:11px 14px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:4px; margin-bottom:5px;">
                        <div>{pills_html}</div>
                        <div>{b_html} <span style="font-weight:800; color:#0f172a; font-size:13px; margin-left:6px;">[{int(g.get('american_odds', 0)):+d}]</span></div>
                    </div>
                    <div style="font-size:14px; font-weight:800; color:#0f172a;">Pata #{idx}: {g.get('selection', '-')}</div>
                    <div style="font-size:11.5px; color:#475569; margin-top:2px;">
                        <b>Partido:</b> {p_name} &nbsp;|&nbsp; <b>Mercado:</b> {g.get('market_type') or g.get('type') or 'Principal'} &nbsp;|&nbsp;
                        <b>Prob. de que Suceda Pata:</b> <span style="color:#047857; font-weight:800;">{l_sec['prob_suceda_pct']:.1f}%</span> &nbsp;|&nbsp;
                        <b>Seguridad Pata:</b> <span style="color:{l_sec['security_color']}; font-weight:800;">{l_sec['seguridad_1_10']:.1f}/10</span>
                    </div>
                    {f'<div style="font-size:11px; color:#1e293b; margin-top:3px;"><b>Estado ESPN:</b> {g_sc}</div>' if g_sc else ''}
                    <div style="font-size:11.5px; color:#334155; margin-top:6px; line-height:1.5;">
                        {g.get('justification', '-')}
                    </div>
                </div>
                """)
    else:
        with st.expander(f"[{title_label}] Sin combinacion activa ({sport_name})", expanded=False):
            st.caption("No se encontraron 2 lineas abiertas en este bloque que cumplan simultaneamente el tope <= +100 MAX y los filtros cuantitativos.")


def render_props_card(props: List[Dict[str, Any]], sport_name: str = "", slate_size: int = 10):
    if props and isinstance(props, list) and len(props) > 0:
        p_names = [p.get("player", "-") for p in props]
        if sport_name == "NHL" and len(props) == 3:
            p_title = f"[TOP 3 PROPS REMATES NHL] {' + '.join(p_names)}"
        else:
            p_title = f"[PARLAY DE PROPS] {' + '.join(p_names)}"
        comb_dec = 1.0
        for p in props:
            comb_dec *= american_to_decimal(int(p.get("american_odds", -180)))
        comb_odds = decimal_to_american(comb_dec)

        eff_slate = int(props[0].get("slate_size") or slate_size or 10)
        t_audit = QuantitativeAIAgent.audit_ticket_security(
            props, vehicle_type="Parlay Props", slate_size=eff_slate, slot_rank=1
        )
        joint_prob = t_audit["ticket_prob_suceda_pct"]
        seg_score = t_audit["ticket_seguridad_1_10"]
        meter_html = build_security_meter_html(t_audit, is_parlay=True, leg_count=len(props))

        with st.expander(
            f"{p_title} | {comb_odds:+d} | Prob. Exito Parlay: {joint_prob:.1f}% | Seguridad: {seg_score:.1f}/10",
            expanded=True,
        ):
            render_clean_html(meter_html)
            for idx, (p, l_sec) in enumerate(zip(props, t_audit["leg_audits"]), 1):
                p_stat = p.get("status", "PROXIMAMENTE")
                p_sc = p.get("score_result", "")
                pills_html = build_context_pills(p, sport_name, sec_audit=l_sec)
                b_html = get_badge_html(p_stat)
                render_clean_html(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #0284c7; border-radius:8px; padding:11px 14px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:4px;">
                        <div>{pills_html}</div>
                        <div>{b_html} <span style="font-weight:800; color:#0f172a; font-size:13px; margin-left:6px;">[{int(p.get('american_odds', 0)):+d}]</span></div>
                    </div>
                    <div style="font-size:14px; font-weight:800; color:#0f172a;">#{idx}. {p.get('selection', '-')} <span style="font-weight:600; color:#475569;">({p.get('team', '-')})</span></div>
                    <div style="font-size:11.5px; color:#475569; margin-top:2px;">
                        <b>Partido:</b> {p.get('game', '-')} &nbsp;|&nbsp; <b>Rival:</b> {p.get('opponent', '-')} &nbsp;|&nbsp;
                        <b>Prob. de que Suceda Prop:</b> <span style="color:#0369a1; font-weight:800;">{l_sec['prob_suceda_pct']:.1f}%</span> &nbsp;|&nbsp;
                        <b>Seguridad Prop:</b> <span style="color:{l_sec['security_color']}; font-weight:800;">{l_sec['seguridad_1_10']:.1f}/10</span>
                    </div>
                    {f'<div style="font-size:11px; color:#1e293b; margin-top:3px;"><b>Auditoria Boxscore ESPN:</b> {p_sc}</div>' if p_sc else ''}
                    <div style="font-size:11.5px; color:#334155; margin-top:6px; line-height:1.5;">
                        {p.get('justification', '-')}
                    </div>
                </div>
                """)
    else:
        with st.expander(f"[PARLAY DE PROPS] Sin props activos ({sport_name})", expanded=False):
            st.caption("Sin props de jugadores titulares libres de reporte de lesion con +EV suficiente en este corte.")


def render_ratonero_card(ratonero: List[Dict[str, Any]], title_label: str = "PARLAY RATONERO (5 PATAS)", slate_size: int = 10):
    if ratonero and isinstance(ratonero, list) and len(ratonero) >= 5:
        comb_dec = 1.0
        for r in ratonero:
            comb_dec *= american_to_decimal(int(r.get("american_odds", -280)))
        comb_odds = decimal_to_american(comb_dec)
        sports_in_rat = sorted(list(set(r.get("sport", "") for r in ratonero if r.get("sport"))))
        sp_tag = " + ".join(sports_in_rat) if len(sports_in_rat) > 1 else (sports_in_rat[0] if sports_in_rat else "")

        eff_slate = int(ratonero[0].get("slate_size") or slate_size or 10)
        t_audit = QuantitativeAIAgent.audit_ticket_security(
            ratonero, vehicle_type="Parlay Ratonero", slate_size=eff_slate, slot_rank=1
        )
        joint_prob = t_audit["ticket_prob_suceda_pct"]
        seg_score = t_audit["ticket_seguridad_1_10"]
        meter_html = build_security_meter_html(t_audit, is_parlay=True, leg_count=len(ratonero))

        with st.expander(
            f"[{title_label}] {sp_tag} | Momio {comb_odds:+d} | Prob. Exito 5 Patas: {joint_prob:.1f}% | Seguridad Ticket: {seg_score:.1f}/10",
            expanded=True,
        ):
            render_clean_html(meter_html)
            for idx, (r, l_sec) in enumerate(zip(ratonero, t_audit["leg_audits"]), 1):
                r_stat = r.get("status", "PROXIMAMENTE")
                r_sc = r.get("score_result", "")
                pills_html = build_context_pills(r, sec_audit=l_sec)
                b_html = get_badge_html(r_stat)
                render_clean_html(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #d97706; border-radius:8px; padding:9px 13px; margin-bottom:7px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:3px;">
                        <div>{pills_html}</div>
                        <div>{b_html} <span style="font-weight:800; color:#0f172a; font-size:12.5px; margin-left:6px;">[{int(r.get('american_odds', 0)):+d}]</span></div>
                    </div>
                    <div style="font-size:13.5px; font-weight:800; color:#0f172a;">#{idx}. {r.get('selection', '-')}</div>
                    <div style="font-size:11px; color:#475569;">
                        <b>Partido:</b> {r.get('game', '-')} &nbsp;|&nbsp;
                        <b>Prob. Pata:</b> <span style="font-weight:800; color:#0f172a;">{l_sec['prob_suceda_pct']:.1f}%</span> &nbsp;|&nbsp;
                        <b>Seguridad Pata:</b> <span style="font-weight:800; color:{l_sec['security_color']};">{l_sec['seguridad_1_10']:.1f}/10</span>
                        {f' &nbsp;|&nbsp; <b>Estado:</b> {r_sc}' if r_sc else ''}
                    </div>
                    <div style="font-size:11px; color:#334155; margin-top:4px; line-height:1.45;">
                        {r.get('justification', '-')}
                    </div>
                </div>
                """)
    else:
        with st.expander(f"[{title_label}] Reservado (Requiere 5 Partidos Elegibles)", expanded=False):
            st.caption("Se requieren al menos 5 partidos independientes con lineas alternativas abiertas (>= 72% probabilidad real y sin contradiccion direccional).")


def render_standard_4_vehicles(sniper: Any, props: list, ratonero: list, ganador: list, sport_name: str, slate_size: int = 10):
    if not sniper and not props and not ratonero and not ganador:
        st.info(f"Sin selecciones activas para {sport_name} en la fecha operativa actual.")
        return
    c1, c2 = st.columns(2)
    with c1:
        render_sniper_card(sniper, "APUESTA FUERTE", sport_name, "#0f172a", slate_size=slate_size)
        render_props_card(props, sport_name, slate_size=slate_size)
    with c2:
        render_ganador_card(ganador, "PARLAY GANADOR", sport_name, slate_size=slate_size)
        render_ratonero_card(ratonero, "PARLAY RATONERO (5 PATAS)", slate_size=slate_size)


def render_ticket_history_list(tickets_list: list):
    if not tickets_list:
        st.info("No hay tickets registrados en el historial para esta categoria.")
        return

    for t in reversed(tickets_list):
        status_val = (t.get("status") or "PROXIMAMENTE").upper()
        odds_str = f"{int(t.get('odds_american', 0)):+d} ({float(t.get('odds_decimal', 1.0)):.2f}x)"
        pnl_val = float(t.get("units_pnl", 0.0))
        pnl_str = f"{pnl_val:+.2f} u"
        legs = t.get("legs", [])
        veh_str = str(t.get("vehicle") or "Apuesta Fuerte")
        sl_sz = int(legs[0].get("slate_size", 10)) if legs and isinstance(legs[0], dict) else 10
        sl_rk = int(legs[0].get("slot_rank", 2 if "#2" in veh_str else 1)) if legs and isinstance(legs[0], dict) else (2 if "#2" in veh_str else 1)
        t_audit = QuantitativeAIAgent.audit_ticket_security(legs, vehicle_type=veh_str, slate_size=sl_sz, slot_rank=sl_rk)
        sec_str = f"Seguridad: {t_audit['ticket_seguridad_1_10']:.1f}/10 ({t_audit['ticket_prob_suceda_pct']:.1f}% Exito)"

        expander_title = f"[{status_val}] {t.get('date')} | {t.get('sport')} - {veh_str} | {sec_str} | Momio: {odds_str} | PnL: {pnl_str}"
        with st.expander(expander_title, expanded=False):
            render_clean_html(f"""
            <div style="font-size:12px; color:#475569; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #e2e8f0;">
                <b>Fecha:</b> {t.get('date')} &nbsp;|&nbsp; <b>Deporte:</b> {t.get('sport')} &nbsp;|&nbsp;
                <b>Vehiculo:</b> {veh_str} &nbsp;|&nbsp; <b>Seguridad Motor:</b> <b style="color:{t_audit['color']};">{t_audit['ticket_seguridad_1_10']:.1f}/10</b>
                ({t_audit['ticket_prob_suceda_pct']:.1f}% Prob. Exito) &nbsp;|&nbsp;
                <b>Resultado General:</b> {t.get('score_result', '-')}
            </div>
            """)
            for idx, leg in enumerate(legs, 1):
                l_stat = (leg.get("status") or "PROXIMAMENTE").upper()
                l_badge = get_badge_html(l_stat)
                l_score = leg.get("score_result") or "Programado"
                if "GANADA" in l_stat:
                    leg_bg, leg_border, leg_left = "#f0fdf4", "#bbf7d0", "#059669"
                elif "PERDIDA" in l_stat:
                    leg_bg, leg_border, leg_left = "#fef2f2", "#fecaca", "#dc2626"
                elif "EN JUEGO" in l_stat:
                    leg_bg, leg_border, leg_left = "#fffbeb", "#fde68a", "#d97706"
                else:
                    leg_bg, leg_border, leg_left = "#f8fafc", "#e2e8f0", "#334155"

                render_clean_html(f"""
                <div style="background:{leg_bg}; border:1px solid {leg_border}; border-left:5px solid {leg_left}; border-radius:8px; padding:9px 14px; margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                        <div>
                            {l_badge} &nbsp;
                            <b style="font-size:13px; color:#0f172a;">#{idx}. {leg.get('selection', '-')}</b>
                            <span style="font-size:11.5px; color:#475569; margin-left:8px;">({leg.get('game', '-')})</span>
                        </div>
                        <span style="font-size:12px; font-weight:800; color:#0f172a;">[{int(leg.get('american_odds', 0)):+d}]</span>
                    </div>
                    <div style="font-size:11.5px; color:#1e293b; margin-top:4px;">
                        <b>Auditoria ESPN:</b> {l_score}
                    </div>
                    <div style="font-size:11px; color:#475569; margin-top:3px;">
                        {leg.get('justification', '-')}
                    </div>
                </div>
                """)


def _fmt_kalshi_sharp_cell(g: Dict[str, Any]) -> str:
    tokens = []
    hk = float(g.get("home_kalshi_prob") or 0.0)
    ak = float(g.get("away_kalshi_prob") or 0.0)
    if hk > 0 or ak > 0:
        tokens.append(f"Kalshi V/L: {ak:.0f}%/{hk:.0f}%")
    hm = g.get("home_sharp_money_pct")
    am = g.get("away_sharp_money_pct")
    if hm is not None and am is not None:
        tokens.append(f"$ V/L: {int(am)}%/{int(hm)}%")
    if g.get("home_rlm"):
        tokens.append("RLM Loc")
    elif g.get("away_rlm"):
        tokens.append("RLM Vis")
    return " | ".join(tokens) if tokens else "Mercado Estandar"


# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
(
    tab_dia,
    tab_ncaaf,
    tab_tenis,
    tab_soccer,
    tab_nba,
    tab_nhl,
    tab_nfl,
    tab_mlb,
    tab_audit,
) = st.tabs([
    "LO MEJOR DEL DIA (MVP)",
    "NCAAF (COLEGIAL - 3 HORARIOS)",
    "TENIS (ATP / WTA)",
    "FUTBOL (5 LIGAS + COPAS + MX)",
    "NBA",
    "NHL",
    "NFL (DOMINGO - 2 HORARIOS)",
    "BEISBOL (LMP / KBO / NPB / MLB)",
    "HISTORIAL Y RENDIMIENTO",
])


# ==============================================================================
# TAB 1: LO MEJOR DEL DIA (MVP)
# ==============================================================================
with tab_dia:
    render_scorecard_ribbon(scorecards["sports"]["dia_mvp"], "LO MEJOR DEL DIA (MVP)")

    dia = res["dia_mvp"]
    has_cross_ratonero = isinstance(dia.get("ratonero"), list) and len(dia.get("ratonero", [])) >= 5
    mode_banner = (
        "MODO ACTIVO: APUESTA FUERTE #1 + PARLAY GANADOR + PARLAY PROPS + PARLAY RATONERO MULTI-DEPORTE (5 PATAS)"
        if has_cross_ratonero
        else "MODO ACTIVO: 2 APUESTAS FUERTES (#1 Y #2) + PARLAY GANADOR + PARLAY PROPS (RESPALDO AUTOMATICO POR LIQUIDEZ)"
    )
    render_clean_html(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px 16px; margin-bottom:12px; font-size:12px; color:#1e293b;">
        <span class="qt-pill-dark">SELECCION PRIORITARIA #1</span>
        <b>{mode_banner}</b>
    </div>
    """)

    sub_dia_v, sub_dia_w, sub_dia_h = st.tabs([
        "VEHICULOS MVP DEL DIA",
        "FLUJO SHARP KALSHI CFTC & ACTION RLM",
        "HISTORIAL MVP",
    ])

    with sub_dia_v:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            render_sniper_card(dia.get("sniper"), "APUESTA FUERTE #1 DEL DIA", "MVP", "#0f172a")
            render_props_card(dia.get("props", []), "MVP MULTI-DEPORTE")
        with col_m2:
            render_ganador_card(dia.get("ganador", []), "PARLAY GANADOR DEL DIA", "MVP")
            if has_cross_ratonero:
                render_ratonero_card(dia.get("ratonero", []), "PARLAY RATONERO MULTI-DEPORTE (5 PATAS)")
            else:
                render_sniper_card(dia.get("sniper_2"), "APUESTA FUERTE #2 DEL DIA", "MVP", "#059669")

    with sub_dia_w:
        whales = res.get("whales", [])
        if whales:
            df_w = pd.DataFrame(whales)[["sport", "game", "target", "volume", "flow_share_pct", "signal"]]
            df_w.columns = ["Deporte", "Evento", "Lado Institucional", "Liquidez / Muestra", "% Dinero / Flujo", "Senal Cuantitativa"]
            df_w["Liquidez / Muestra"] = df_w["Liquidez / Muestra"].apply(lambda x: f"${x:,.0f}")
            df_w["% Dinero / Flujo"] = df_w["% Dinero / Flujo"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df_w, use_container_width=True, hide_index=True)
        else:
            st.info("Sin anomalias de flujo institucional en Kalshi CFTC ni Action Network en este corte.")

    with sub_dia_h:
        render_ticket_history_list(scorecards["sports"]["dia_mvp"]["tickets"])


# ==============================================================================
# TAB 2: NCAAF (COLEGIAL - STRICTLY SEPARATED INTO 3 INDEPENDENT HORARIOS)
# Each Horario (Manana, Tarde, Noche) has its OWN 2 Apuestas Fuertes + 2 Parlays Ganadores!
# ==============================================================================
with tab_ncaaf:
    ncaaf_data = res.get("ncaaf", {})
    ncaaf_sc = scorecards["sports"].get(
        "ncaaf",
        {"total_bets": 0, "won": 0, "lost": 0, "in_play": 0, "upcoming": 0, "win_rate_pct": 0.0, "roi_pct": 0.0, "units_pnl": 0.0, "tickets": []},
    )
    render_scorecard_ribbon(ncaaf_sc, "NCAAF (FUTBOL AMERICANO COLEGIAL - 3 HORARIOS SEPARADOS)")

    horarios_map = ncaaf_data.get("horarios", {"Manana": [], "Tarde": [], "Noche": []})
    by_horario_picks = ncaaf_data.get("by_horario_picks", {})
    cnt_m = len(horarios_map.get("Manana", []))
    cnt_t = len(horarios_map.get("Tarde", []))
    cnt_n = len(horarios_map.get("Noche", []))
    ncaaf_dt = ncaaf_data.get("target_date", today_str)

    hk1, hk2, hk3, hk4 = st.columns(4)
    with hk1:
        render_clean_html(f"""
        <div class="qt-kpi-card" style="border-top-color:#0f172a;">
            <div class="qt-label">JORNADA COLEGIAL FBS</div>
            <div class="qt-value">{ncaaf_dt}</div>
            <div style="font-size:11px; color:#475569; margin-top:2px;">{cnt_m + cnt_t + cnt_n} Partidos Analizados (SP+ / EPA / Kalshi / RLM)</div>
        </div>
        """)
    with hk2:
        render_clean_html(f"""
        <div class="qt-kpi-card" style="border-top-color:#0284c7;">
            <div class="qt-label">HORARIO 1: MANANA (&lt; 14:00 CST)</div>
            <div class="qt-value">{cnt_m} Juegos</div>
            <div style="font-size:11px; color:#0369a1; margin-top:2px;">2 Fuertes + 2 Parlays Ganadores Propios</div>
        </div>
        """)
    with hk3:
        render_clean_html(f"""
        <div class="qt-kpi-card" style="border-top-color:#059669;">
            <div class="qt-label">HORARIO 2: TARDE (14:00 - 17:59 CST)</div>
            <div class="qt-value">{cnt_t} Juegos</div>
            <div style="font-size:11px; color:#047857; margin-top:2px;">2 Fuertes + 2 Parlays Ganadores Propios</div>
        </div>
        """)
    with hk4:
        render_clean_html(f"""
        <div class="qt-kpi-card" style="border-top-color:#d97706;">
            <div class="qt-label">HORARIO 3: NOCHE (&ge; 18:00 CST)</div>
            <div class="qt-value">{cnt_n} Juegos</div>
            <div style="font-size:11px; color:#b45309; margin-top:2px;">2 Fuertes + 2 Parlays Ganadores Propios (+3.7 HFA)</div>
        </div>
        """)

    render_clean_html("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; margin-bottom:14px; font-size:12px; color:#1e293b;">
        <span class="qt-pill-dark">AISLAMIENTO ESTRICTO POR HORARIO</span>
        <b>Estructura Sabado Colegial:</b> Cada horario (<b>Manana</b>, <b>Tarde</b> y <b>Noche</b>) cuenta con sus propias
        <b>2 Apuestas Fuertes</b> y <b>2 Parlays Ganadores (&le; +100 MAX)</b> exclusivamente con partidos de ese mismo horario (cero mezcla entre turnos, cero Ratoneros y cero Props).
    </div>
    """)

    tab_nc_man, tab_nc_tar, tab_nc_noc, tab_nc_hist = st.tabs([
        f"HORARIO MANANA (< 14:00 CST | {cnt_m} JUEGOS)",
        f"HORARIO TARDE (14:00 - 17:59 CST | {cnt_t} JUEGOS)",
        f"HORARIO NOCHE (>= 18:00 CST | {cnt_n} JUEGOS)",
        "HISTORIAL NCAAF",
    ])

    for turno_tab, turno_key, turno_title, accent_c in (
        (tab_nc_man, "Manana", "HORARIO MANANA (< 14:00 CST)", "#0284c7"),
        (tab_nc_tar, "Tarde", "HORARIO TARDE (14:00 - 17:59 CST)", "#059669"),
        (tab_nc_noc, "Noche", "HORARIO NOCHE (>= 18:00 CST)", "#d97706"),
    ):
        with turno_tab:
            tp = by_horario_picks.get(turno_key, {})
            t_games = horarios_map.get(turno_key, [])
            if not t_games:
                st.info(f"Sin partidos FBS programados en {turno_title}.")
            else:
                c_left, c_right = st.columns(2)
                with c_left:
                    render_sniper_card(tp.get("sniper_1"), f"APUESTA FUERTE #1 ({turno_key.upper()})", "NCAAF", accent_c, slate_size=len(t_games))
                    render_sniper_card(tp.get("sniper_2"), f"APUESTA FUERTE #2 ({turno_key.upper()})", "NCAAF", "#0f172a", slate_size=len(t_games))
                with c_right:
                    render_ganador_card(tp.get("ganador_1", []), f"PARLAY GANADOR #1 ({turno_key.upper()})", "NCAAF", slate_size=len(t_games))
                    render_ganador_card(tp.get("ganador_2", []), f"PARLAY GANADOR #2 ({turno_key.upper()})", "NCAAF", slate_size=len(t_games))

                st.markdown(f"##### TABLA SABERMETRICA, KALSHI & FLUJO SHARP - {turno_title}")
                rows_nc = []
                for g in t_games:
                    rows_nc.append({
                        "Kickoff": g.get("kickoff_cst", "-"),
                        "Partido FBS": g.get("display_matchup") or g.get("matchup", "-"),
                        "Records (V/L)": f"{g.get('away_record', '-')} / {g.get('home_record', '-')}",
                        "Spread Local": f"{g.get('spread', 0.0):+.1f}",
                        "Total O/U": g.get("total", 52.5),
                        "SP+ Calibrado (V/L)": f"{g.get('away_sp_plus', 0.0):+.1f} / {g.get('home_sp_plus', 0.0):+.1f}",
                        "EPA Off (V/L)": f"{g.get('away_epa_off', 0.0):+.2f} / {g.get('home_epa_off', 0.0):+.2f}",
                        "ML (Vis / Loc)": f"{int(g.get('away_odds', 0)):+d} / {int(g.get('home_odds', 0)):+d}",
                        "Prob. Real Loc": f"{float(g.get('home_real_prob', 0.5))*100:.1f}%",
                        "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                        "Consenso (V/L)": f"{g.get('away_consensus', 2)}/4 | {g.get('home_consensus', 2)}/4",
                        "Simulacion 10k": g.get("projected_score", "-"),
                    })
                st.dataframe(pd.DataFrame(rows_nc), use_container_width=True, hide_index=True)

    with tab_nc_hist:
        render_ticket_history_list(ncaaf_sc.get("tickets", []))


# ==============================================================================
# TAB 3: TENIS (ATP & WTA SINGLES)
# ==============================================================================
with tab_tenis:
    ten_data = res.get("tenis", {})
    ten_sc = scorecards["sports"].get(
        "tenis",
        {"total_bets": 0, "won": 0, "lost": 0, "in_play": 0, "upcoming": 0, "win_rate_pct": 0.0, "roi_pct": 0.0, "units_pnl": 0.0, "tickets": []},
    )
    render_scorecard_ribbon(ten_sc, "TENIS (ATP & WTA SINGLES)")

    render_clean_html("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; margin-bottom:14px; font-size:12px; color:#1e293b;">
        <span class="qt-pill-dark">MARKOV + SABERMETRIA ATP/WTA + KALSHI</span>
        <b>Motores Activos:</b> Elo Ajustado por Superficie + Modelo Klaassen-Magnus de Saque/Retorno
        (Hold%, Break%, Dominance Ratio, Tiebreak%) + Contratos Kalshi ATP/WTA + 10,000 Simulaciones por Sets. Vehiculos exclusivos: <b>Apuesta Fuerte</b> y <b>Parlay Ganador</b>.
    </div>
    """)

    sub_ten_v, sub_ten_c, sub_ten_h = st.tabs([
        "VEHICULOS OFICIALES TENIS (+EV)",
        "CARTELERA SABERMETRICA ATP / WTA",
        "HISTORIAL TENIS",
    ])

    with sub_ten_v:
        ten_cnt = max(1, len(ten_data.get("games", [])))
        tc1, tc2 = st.columns(2)
        with tc1:
            render_sniper_card(ten_data.get("sniper"), "APUESTA FUERTE TENIS", "Tenis", "#0f172a", slate_size=ten_cnt)
        with tc2:
            render_ganador_card(ten_data.get("ganador", []), "PARLAY GANADOR TENIS", "Tenis", slate_size=ten_cnt)

    with sub_ten_c:
        t_games = ten_data.get("games", [])
        if t_games:
            t_rows = []
            for m in t_games:
                p1_ev = float(m.get("home_ev", 0.0)) * 100.0
                p2_ev = float(m.get("away_ev", 0.0)) * 100.0
                t_rows.append({
                    "Torneo": m.get("tournament", "-"),
                    "Superficie": m.get("surface", "-"),
                    "Inicio": m.get("kickoff_cst", "-"),
                    "Encuentro Singles": m.get("matchup", "-"),
                    "Momios ML (J1 / J2)": f"{int(m.get('home_odds', 0)):+d} / {int(m.get('away_odds', 0)):+d}",
                    "Elo Superficie": f"{m.get('p1_elo_surface', 0):.0f} vs {m.get('p2_elo_surface', 0):.0f}",
                    "Hold % (J1 / J2)": f"{m.get('p1_hold_pct', 0):.1f}% / {m.get('p2_hold_pct', 0):.1f}%",
                    "Break % (J1 / J2)": f"{m.get('p1_break_pct', 0):.1f}% / {m.get('p2_break_pct', 0):.1f}%",
                    "DR (J1 / J2)": f"{m.get('p1_dr', 1.0):.2f} vs {m.get('p2_dr', 1.0):.2f}",
                    "Prob. Real (J1 / J2)": f"{float(m.get('home_real_prob', 0.5))*100:.1f}% / {float(m.get('away_real_prob', 0.5))*100:.1f}%",
                    "EV Max": f"{max(p1_ev, p2_ev):+.1f}%",
                    "Proyeccion Juegos": m.get("projected_score", "-"),
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Sin encuentros Singles de circuito principal programados en la ventana actual.")

    with sub_ten_h:
        render_ticket_history_list(ten_sc.get("tickets", []))


# ==============================================================================
# TAB 4: FUTBOL (5 GRANDES LIGAS + COPAS DOMESTICAS + UEFA + LIGA MX)
# ==============================================================================
with tab_soccer:
    render_scorecard_ribbon(scorecards["sports"]["futbol"], "FUTBOL (5 GRANDES LIGAS + COPAS + LIGA MX)")

    soc = res["soccer"]
    active_leagues = sorted(list(set(g.get("league_name", "Futbol") for g in soc.get("games", []))))
    lg_pills = "".join(f'<span class="qt-pill-slate">{lg}</span>' for lg in active_leagues) if active_leagues else '<span class="qt-pill-dark">Monitoreando 15 Ligas y Copas</span>'
    render_clean_html(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; margin-bottom:14px; font-size:12px;">
        <b>Competiciones Activas Hoy ({len(soc.get('games', []))} Partidos):</b> &nbsp; {lg_pills}
    </div>
    """)

    sub_soc_v, sub_soc_c, sub_soc_h = st.tabs([
        "VEHICULOS ACTIVOS FUTBOL (+EV)",
        "CARTELERA SABERMETRICA (xG / PPDA / KALSHI)",
        "HISTORIAL FUTBOL",
    ])

    with sub_soc_v:
        # Cero props de jugadores en futbol. Estructura: 2 Apuestas Fuertes + 2 Parlays Ganadores (o Ratonero)
        c1, c2 = st.columns(2)
        with c1:
            render_sniper_card(soc.get("sniper"), "APUESTA FUERTE #1", "FUTBOL", "#0f172a", slate_size=max(1, len(soc.get("games", []))))
            if soc.get("sniper_2"):
                render_sniper_card(soc.get("sniper_2"), "APUESTA FUERTE #2", "FUTBOL", "#334155", slate_size=max(1, len(soc.get("games", []))))
        with c2:
            render_ganador_card(soc.get("ganador", []), "PARLAY GANADOR #1", "FUTBOL", slate_size=max(1, len(soc.get("games", []))))
            if soc.get("ganador_2"):
                render_ganador_card(soc.get("ganador_2", []), "PARLAY GANADOR #2", "FUTBOL", slate_size=max(1, len(soc.get("games", []))))
            if soc.get("ratonero"):
                render_ratonero_card(soc.get("ratonero", []), "PARLAY RATONERO (5 PATAS)", slate_size=max(1, len(soc.get("games", []))))

    with sub_soc_c:
        games_soc = soc.get("games", [])
        if games_soc:
            soc_rows = []
            for g in games_soc:
                hr = float(g.get("home_real_prob", 0))
                dr = float(g.get("draw_real_prob", 0))
                ar = float(g.get("away_real_prob", 0))
                hr_p = hr * 100.0 if hr <= 1.0 else hr
                dr_p = dr * 100.0 if dr <= 1.0 else dr
                ar_p = ar * 100.0 if ar <= 1.0 else ar
                soc_rows.append({
                    "Competicion": g.get("league_name", "-"),
                    "Hora": g.get("kickoff_cst", "-"),
                    "Partido (Visita vs Local)": g.get("matchup", "-"),
                    "xG (Vis / Loc)": f"{g.get('away_xg', 0):.2f} / {g.get('home_xg', 0):.2f}",
                    "Prob. Loc / Emp / Vis": f"{hr_p:.1f}% / {dr_p:.1f}% / {ar_p:.1f}%",
                    "Momios (Loc / Emp / Vis)": f"{int(g.get('home_odds', 0)):+d} / {int(g.get('draw_odds', 240)):+d} / {int(g.get('away_odds', 0)):+d}",
                    "EV Local / Visita": f"{float(g.get('home_ev', 0)):+.1f}% / {float(g.get('away_ev', 0)):+.1f}%",
                    "Kalshi / Sharp": _fmt_kalshi_sharp_cell(g),
                    "Marcador xG": g.get("projected_score", "-"),
                })
            st.dataframe(pd.DataFrame(soc_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Sin partidos programados hoy en las 5 Grandes Ligas, Copas Europeas o Liga MX.")

    with sub_soc_h:
        render_ticket_history_list(scorecards["sports"]["futbol"]["tickets"])


# ==============================================================================
# TAB 5: NBA
# ==============================================================================
with tab_nba:
    render_scorecard_ribbon(scorecards["sports"]["nba"], "NBA")
    nba = res["nba"]
    sub_nba_v, sub_nba_c, sub_nba_h = st.tabs(["VEHICULOS ACTIVOS NBA", "CARTELERA SABERMETRICA NBA", "HISTORIAL NBA"])

    with sub_nba_v:
        render_standard_4_vehicles(nba["sniper"], nba["props"], nba["ratonero"], nba["ganador"], "NBA", slate_size=max(1, len(nba.get("games", []))))

    with sub_nba_c:
        games_nba = nba.get("games", [])
        if games_nba:
            rows_nba = []
            for g in games_nba:
                rows_nba.append({
                    "Partido": g["matchup"],
                    "Spread Local": f"{g['spread']:+.1f}",
                    "Total O/U": g["total"],
                    "ML (Vis / Loc)": f"{int(g['away_odds']):+d} / {int(g['home_odds']):+d}",
                    "Prob. Real (Vis / Loc)": f"{g['away_real_prob']*100:.1f}% / {g['home_real_prob']*100:.1f}%",
                    "EV (Vis / Loc)": f"{g['away_ev']*100:+.1f}% / {g['home_ev']*100:+.1f}%",
                    "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                    "Consenso (V/L)": f"{g.get('away_consensus', 2)}/4 | {g.get('home_consensus', 2)}/4",
                    "Marcador Simulado": g["projected_score"],
                })
            st.dataframe(pd.DataFrame(rows_nba), use_container_width=True, hide_index=True)
        else:
            st.info("Sin partidos de NBA programados para hoy.")

    with sub_nba_h:
        render_ticket_history_list(scorecards["sports"]["nba"]["tickets"])


# ==============================================================================
# TAB 6: NHL
# ==============================================================================
with tab_nhl:
    render_scorecard_ribbon(scorecards["sports"]["nhl"], "NHL")
    nhl = res["nhl"]
    sub_nhl_v, sub_nhl_c, sub_nhl_h = st.tabs(["VEHICULOS ACTIVOS NHL", "CARTELERA SABERMETRICA NHL", "HISTORIAL NHL"])

    with sub_nhl_v:
        render_standard_4_vehicles(nhl["sniper"], nhl["props"], nhl["ratonero"], nhl["ganador"], "NHL", slate_size=max(1, len(nhl.get("games", []))))

    with sub_nhl_c:
        games_nhl = nhl.get("games", [])
        if games_nhl:
            nhl_rows = []
            for g in games_nhl:
                nhl_rows.append({
                    "Partido": g["matchup"],
                    "Portero Local (SV%)": f"{g['home_starter']} ({g['home_sv']:.3f})",
                    "Portero Visita (SV%)": f"{g['away_starter']} ({g['away_sv']:.3f})",
                    "ML (Vis / Loc)": f"{int(g['away_odds']):+d} / {int(g['home_odds']):+d}",
                    "Prob. Real (Vis / Loc)": f"{g['away_real_prob']*100:.1f}% / {g['home_real_prob']*100:.1f}%",
                    "EV (Vis / Loc)": f"{g['away_ev']*100:+.1f}% / {g['home_ev']*100:+.1f}%",
                    "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                    "Consenso (V/L)": f"{g.get('away_consensus', 2)}/4 | {g.get('home_consensus', 2)}/4",
                    "Marcador Poisson": g["projected_score"],
                })
            st.dataframe(pd.DataFrame(nhl_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Sin partidos de NHL programados para hoy.")

    with sub_nhl_h:
        render_ticket_history_list(scorecards["sports"]["nhl"]["tickets"])


# ==============================================================================
# TAB 7: NFL
# ==============================================================================
with tab_nfl:
    render_scorecard_ribbon(scorecards["sports"]["nfl"], "NFL (FUTBOL AMERICANO PROFESIONAL)")
    nfl = res["nfl"]
    is_sunday = nfl.get("is_sunday_slate", False)
    by_horario_picks = nfl.get("by_horario_picks", {})
    horarios_map = nfl.get("horarios", {})
    games_nfl = nfl.get("games", [])
    nfl_dt = nfl.get("target_date", today_str)

    if is_sunday and by_horario_picks:
        cnt_temp = len(horarios_map.get("Temprano", []))
        cnt_tn = len(horarios_map.get("Tarde y Noche", []))

        nk1, nk2, nk3 = st.columns(3)
        with nk1:
            render_clean_html(f"""
            <div class="qt-kpi-card" style="border-top-color:#0f172a;">
                <div class="qt-label">JORNADA DOMINICAL NFL</div>
                <div class="qt-value">{nfl_dt}</div>
                <div style="font-size:11px; color:#475569; margin-top:2px;">{len(games_nfl)} Partidos Analizados (DVOA / EPA / Kalshi / RLM)</div>
            </div>
            """)
        with nk2:
            render_clean_html(f"""
            <div class="qt-kpi-card" style="border-top-color:#0284c7;">
                <div class="qt-label">HORARIO 1: TEMPRANO (&lt; 14:00 CST)</div>
                <div class="qt-value">{cnt_temp} Juegos</div>
                <div style="font-size:11px; color:#0369a1; margin-top:2px;">2 Fuertes + 2 Parlays Ganadores Propios</div>
            </div>
            """)
        with nk3:
            render_clean_html(f"""
            <div class="qt-kpi-card" style="border-top-color:#059669;">
                <div class="qt-label">HORARIO 2: TARDE Y NOCHE (&ge; 14:00 CST)</div>
                <div class="qt-value">{cnt_tn} Juegos</div>
                <div style="font-size:11px; color:#047857; margin-top:2px;">Ventana de la Tarde + Sunday Night Football</div>
            </div>
            """)

        render_clean_html("""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; margin-bottom:14px; font-size:12px; color:#1e293b;">
            <span class="qt-pill-dark">DIVISION DOMINICAL ESTRICTA POR HORARIO</span>
            <b>Estructura Domingo NFL:</b> La jornada dominical se divide estrictamente en dos turnos:
            <b>Horario Temprano</b> (&lt; 14:00 CST) y <b>Horario Tarde y Noche</b> (&ge; 14:00 CST).
            Cada bloque cuenta con sus propias <b>2 Apuestas Fuertes</b> y <b>2 Parlays Ganadores (&le; +100 MAX)</b>, sin mezclar partidos entre horarios.
        </div>
        """)

        tab_nfl_temp, tab_nfl_tn, tab_nfl_all, tab_nfl_h = st.tabs([
            f"HORARIO TEMPRANO (< 14:00 CST | {cnt_temp} JUEGOS)",
            f"HORARIO TARDE Y NOCHE (>= 14:00 CST | {cnt_tn} JUEGOS)",
            f"CARTELERA COMPLETA NFL ({len(games_nfl)} JUEGOS)",
            "HISTORIAL NFL",
        ])

        for turno_tab, turno_key, turno_title, accent_c in (
            (tab_nfl_temp, "Temprano", "HORARIO TEMPRANO (< 14:00 CST)", "#0284c7"),
            (tab_nfl_tn, "Tarde y Noche", "HORARIO TARDE Y NOCHE (>= 14:00 CST)", "#059669"),
        ):
            with turno_tab:
                tp = by_horario_picks.get(turno_key, {})
                t_games = horarios_map.get(turno_key, [])
                if not t_games:
                    st.info(f"Sin partidos NFL en {turno_title}.")
                else:
                    c_left, c_right = st.columns(2)
                    with c_left:
                        render_sniper_card(tp.get("sniper_1"), f"APUESTA FUERTE #1 ({turno_key.upper()})", "NFL", accent_c, slate_size=len(t_games))
                        render_sniper_card(tp.get("sniper_2"), f"APUESTA FUERTE #2 ({turno_key.upper()})", "NFL", "#0f172a", slate_size=len(t_games))
                    with c_right:
                        render_ganador_card(tp.get("ganador_1", []), f"PARLAY GANADOR #1 ({turno_key.upper()})", "NFL", slate_size=len(t_games))
                        render_ganador_card(tp.get("ganador_2", []), f"PARLAY GANADOR #2 ({turno_key.upper()})", "NFL", slate_size=len(t_games))

                    st.markdown(f"##### TABLA ANALITICA Y FLUJO SHARP - {turno_title}")
                    rows_t = []
                    for g in t_games:
                        hp = g.get("home_real_prob", 0.5)
                        ap = g.get("away_real_prob", 0.5)
                        rows_t.append({
                            "Kickoff": g.get("kickoff_cst", "-"),
                            "Partido": g.get("matchup", "-"),
                            "Spread": f"{g.get('spread', 0):+.1f}",
                            "Total O/U": g.get("total", 0),
                            "ML (Vis / Loc)": f"{int(g.get('away_odds', 0)):+d} / {int(g.get('home_odds', 0)):+d}",
                            "Prob. Real (Vis / Loc)": f"{(ap*100 if ap<=1 else ap):.1f}% / {(hp*100 if hp<=1 else hp):.1f}%",
                            "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                            "Proyeccion": g.get("projected_score", "-"),
                        })
                    st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

        with tab_nfl_all:
            if games_nfl:
                rows_all = []
                for g in games_nfl:
                    hp = g.get("home_real_prob", 0.5)
                    ap = g.get("away_real_prob", 0.5)
                    rows_all.append({
                        "Kickoff": g.get("kickoff_cst", "-"),
                        "Partido": g.get("matchup", "-"),
                        "Spread": f"{g.get('spread', 0):+.1f}",
                        "Total O/U": g.get("total", 0),
                        "ML (Vis / Loc)": f"{int(g.get('away_odds', 0)):+d} / {int(g.get('home_odds', 0)):+d}",
                        "Prob. Real (Vis / Loc)": f"{(ap*100 if ap<=1 else ap):.1f}% / {(hp*100 if hp<=1 else hp):.1f}%",
                        "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                        "Proyeccion": g.get("projected_score", "-"),
                    })
                st.dataframe(pd.DataFrame(rows_all), use_container_width=True, hide_index=True)
            else:
                st.info("Sin partidos de NFL programados.")

        with tab_nfl_h:
            render_ticket_history_list(scorecards["sports"]["nfl"]["tickets"])

    else:
        sub_nfl_v, sub_nfl_c, sub_nfl_h = st.tabs(["VEHICULOS ACTIVOS NFL", "CARTELERA NFL", "HISTORIAL NFL"])
        with sub_nfl_v:
            render_standard_4_vehicles(nfl["sniper"], nfl["props"], nfl["ratonero"], nfl["ganador"], "NFL", slate_size=max(1, len(games_nfl)))
        with sub_nfl_c:
            if games_nfl:
                nfl_rows = []
                for g in games_nfl:
                    hp = g.get("home_real_prob", 0.5)
                    ap = g.get("away_real_prob", 0.5)
                    nfl_rows.append({
                        "Kickoff": g.get("kickoff_cst", "-"),
                        "Partido": g.get("matchup", "-"),
                        "Spread": f"{g.get('spread', 0):+.1f}",
                        "Total O/U": g.get("total", 0),
                        "ML (Vis / Loc)": f"{int(g.get('away_odds', 0)):+d} / {int(g.get('home_odds', 0)):+d}",
                        "Prob. Real (Vis / Loc)": f"{(ap*100 if ap<=1 else ap):.1f}% / {(hp*100 if hp<=1 else hp):.1f}%",
                        "Kalshi / Sharp $": _fmt_kalshi_sharp_cell(g),
                        "Proyeccion": g.get("projected_score", "-"),
                    })
                st.dataframe(pd.DataFrame(nfl_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Sin partidos de NFL programados para hoy.")
        with sub_nfl_h:
            render_ticket_history_list(scorecards["sports"]["nfl"]["tickets"])


# ==============================================================================
# TAB 8: BEISBOL (LMP / KBO / NPB / MLB)
# ==============================================================================
with tab_mlb:
    render_scorecard_ribbon(scorecards["sports"]["mlb"], "BEISBOL (LMP / KBO / NPB / MLB)")
    mlb = res["mlb"]
    games_bb = mlb.get("games", [])
    mlb_ws_data = mlb.get("matchup_data", {})

    render_clean_html(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:9px 15px; margin-bottom:14px; font-size:12px; color:#1e293b;">
        <span class="qt-pill-dark">EXPANSION BEISBOL: LMP + KBO + NPB + MLB</span>
        <b>Ligas y Torneos Integrados:</b> Cobertura de la <b>Liga ARCO Mexicana del Pacifico (LMP)</b> con sus 10 franquicias invernales,
        series de campeonato <b>KBO (Corea)</b> y <b>NPB (Japon)</b>, y Serie Mundial de <b>MLB</b>. Modelos con FIP de abridores, wRC+ de lineups y park factors.
    </div>
    """)

    sub_bb_v, sub_bb_c, sub_bb_h = st.tabs([
        f"VEHICULOS ACTIVOS BEISBOL (+EV | {len(games_bb)} JUEGOS)",
        f"CARTELERA LMP (LIGA DEL PACIFICO) & INTERNACIONAL ({len(games_bb)})",
        "HISTORIAL BEISBOL",
    ])

    with sub_bb_v:
        if not mlb.get("sniper") and not mlb.get("ganador") and not mlb.get("sniper_2") and not mlb.get("ganador_2"):
            st.info("Sin selecciones de beisbol activas hoy.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                render_sniper_card(mlb.get("sniper"), "APUESTA FUERTE #1 BEISBOL", "MLB", "#0f172a", slate_size=max(1, len(games_bb)))
                if mlb.get("sniper_2"):
                    render_sniper_card(mlb.get("sniper_2"), "APUESTA FUERTE #2 BEISBOL", "MLB", "#334155", slate_size=max(1, len(games_bb)))
            with c2:
                render_ganador_card(mlb.get("ganador", []), "PARLAY GANADOR #1 BEISBOL", "MLB", slate_size=max(1, len(games_bb)))
                if mlb.get("ganador_2"):
                    render_ganador_card(mlb.get("ganador_2", []), "PARLAY GANADOR #2 BEISBOL", "MLB", slate_size=max(1, len(games_bb)))

    with sub_bb_c:
        if games_bb:
            bb_rows = []
            for g in games_bb:
                hp = float(g.get("home_real_prob", 0.5))
                ap = float(g.get("away_real_prob", 0.5))
                bb_rows.append({
                    "Liga": g.get("league", "MLB"),
                    "Hora CST": g.get("kickoff_cst", "-"),
                    "Partido": g.get("matchup", "-"),
                    "ML (Vis / Loc)": f"{int(g.get('away_odds', 0)):+d} / {int(g.get('home_odds', 0)):+d}",
                    "Total O/U": g.get("total", 8.5),
                    "Prob. Real (Vis / Loc)": f"{(ap*100 if ap<=1 else ap):.1f}% / {(hp*100 if hp<=1 else hp):.1f}%",
                    "EV (Vis / Loc)": f"{float(g.get('away_ev', 0))*100:+.1f}% / {float(g.get('home_ev', 0))*100:+.1f}%",
                    "Sabermetria Local": g.get("saber_home_note", "-"),
                    "Sabermetria Visita": g.get("saber_away_note", "-"),
                    "Proyeccion R-H": g.get("projected_score", "-"),
                })
            st.dataframe(pd.DataFrame(bb_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Sin partidos de beisbol activos para hoy.")

        if mlb_ws_data:
            render_clean_html(f"""
            <div class="qt-card" style="margin-top:14px;">
                <div class="qt-label">ESTADO OFICIAL SERIE MUNDIAL MLB</div>
                <div style="font-size:15px; font-weight:800; color:#0f172a;">{mlb_ws_data.get('matchup', '-')}</div>
                <div style="font-size:12px; color:#475569; margin-top:3px;">
                    <b>Estado:</b> {mlb_ws_data.get('status', '-')} &nbsp;|&nbsp; <b>Fecha:</b> {mlb_ws_data.get('date', '-')}
                </div>
            </div>
            """)

        render_clean_html("""
        <div class="qt-card" style="margin-top:14px;">
            <div class="qt-label">LIGAS ASIATICAS: KBO (COREA) & NPB (JAPON)</div>
            <div style="font-size:12px; color:#334155; line-height:1.6;">
                <b>KBO League (Corea):</b> Kia Tigers, Samsung Lions, LG Twins, Doosan Bears, KT Wiz. Modelo de contacto y bullpen KBO activo.<br>
                <b>NPB Nippon Professional Baseball (Japon):</b> Yomiuri Giants, Hanshin Tigers, Fukuoka SoftBank Hawks, Yokohama DeNA BayStars. Calibracion de pitcheo dominante FIP &lt; 3.20.
            </div>
        </div>
        """)

    with sub_bb_h:
        render_ticket_history_list(scorecards["sports"]["mlb"]["tickets"])


# ==============================================================================
# TAB 9: AUDITORIA Y MATRIZ COMPARATIVA POR DEPORTE Y FECHA
# ==============================================================================
with tab_audit:
    st.markdown("#### MATRIZ COMPARATIVA DE RENDIMIENTO POR DEPORTE (MODELO PROBABILISTICO)")
    sp_labels = [
        ("Lo Mejor del Dia (MVP)", "dia_mvp"),
        ("NCAAF (Colegial)", "ncaaf"),
        ("Tenis (ATP / WTA)", "tenis"),
        ("Futbol (5 Ligas + Copas + MX)", "futbol"),
        ("NBA", "nba"),
        ("NHL", "nhl"),
        ("NFL (Domingo - 2 Horarios)", "nfl"),
        ("Beisbol (LMP / KBO / NPB / MLB)", "mlb"),
    ]
    comp_rows = []
    for lbl, key in sp_labels:
        s_obj = scorecards["sports"].get(key, {})
        comp_rows.append({
            "Deporte / Portafolio": lbl,
            "Tickets Totales": s_obj.get("total_bets", 0),
            "Ganadas": s_obj.get("won", 0),
            "Perdidas": s_obj.get("lost", 0),
            "En Juego": s_obj.get("in_play", 0),
            "Proximamente": s_obj.get("upcoming", 0),
            "Win Rate (%)": f"{s_obj.get('win_rate_pct', 0.0):.1f}%",
            "ROI (%)": f"{s_obj.get('roi_pct', 0.0):+.1f}%",
            "Balance Neto (u)": f"{s_obj.get('units_pnl', 0.0):+.2f} u",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("#### AUDITORIA HISTORICA POR FECHA")
    available_dates = history.get_available_dates()
    cd1, _ = st.columns([1.2, 2.8])
    selected_date = cd1.selectbox("Selecciona Fecha Operativa:", available_dates, index=0)
    day_sc = history.get_scorecard_for_date(selected_date)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_defs = [
        (c1, "Total Tickets", str(day_sc.get("total_bets", 0)), "#0f172a"),
        (c2, "Ganadas", str(day_sc.get("won", 0)), "#059669"),
        (c3, "Perdidas", str(day_sc.get("lost", 0)), "#dc2626"),
        (c4, "En Juego", str(day_sc.get("in_play", 0)), "#d97706"),
        (c5, "Win Rate", f"{day_sc.get('win_rate_pct', 0.0):.1f}%", "#0284c7"),
        (c6, "Balance PnL", f"{day_sc.get('units_pnl', 0.0):+.2f} u", "#059669" if day_sc.get("units_pnl", 0.0) >= 0 else "#dc2626"),
    ]
    for col_obj, k_lbl, k_val, k_col in kpi_defs:
        with col_obj:
            render_clean_html(f"""
            <div class="qt-kpi-card" style="border-top-color:{k_col};">
                <div class="qt-label">{k_lbl}</div>
                <div class="qt-value">{k_val}</div>
            </div>
            """)

    st.markdown(f"#### TICKETS AUDITADOS DE LA FECHA: {selected_date}")
    render_ticket_history_list(day_sc.get("tickets", []))
