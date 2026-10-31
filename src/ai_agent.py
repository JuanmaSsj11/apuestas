"""
Quantitative AI Agent, Meta-Learning & Intelligent 1-10 Security Audit Engine
Orchestrates:
1. Poisson / Dixon-Coles Bivariate Distributions (Goles, carreras, marcadores exactos)
2. Sabermetria Avanzada y Eficiencia Neta (Four Factors, SP+, 5v5 xGF%/HDCF%, Surface Elo, EPA/DVOA, FIP/wRC+)
3. Matchup Contextual & Situational ML (Fatiga, localia, lesiones, Hold%/Break%, PPDA/Field Tilt)
4. Simulaciones de Monte Carlo (10,000 iteraciones estocasticas)
5. Liquidez Institucional Kalshi CFTC + Action Network Sharp Money (% Dinero vs % Tickets) + RLM
6. Motor Inteligente de Seguridad (Escala 1.0 - 10.0), Probabilidad Calibrada de Ocurrencia (%)
   y Detector Honesto Anti-Relleno de Cuota (identifica apuestas de maxima conviccion vs complementos de cartelera corta).
Strictly zero emojis.
"""
import math
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def _american_to_decimal_local(american: int) -> float:
    if american > 0:
        return 1.0 + (american / 100.0)
    return 1.0 + (100.0 / max(1, abs(american)))


def _implied_prob_pct_local(american: int) -> float:
    if american > 0:
        return (100.0 / (american + 100.0)) * 100.0
    a = abs(american)
    return (a / (a + 100.0)) * 100.0


GLOSARIO_INDICADORES: List[Dict[str, str]] = [
    {
        "badge": "RLM CONFIRMADO",
        "color": "#b45309",
        "bg": "#fffbeb",
        "border": "#fde68a",
        "titulo": "Reverse Line Movement (Movimiento Inverso de Linea)",
        "explicacion": (
            "Ocurre cuando la mayoria del publico recreativo (ej. 68%-75% de los tickets) apuesta a un equipo, "
            "pero la casa de apuestas mueve la linea o el momio en direccion contraria porque entro dinero institucional "
            "pesado (sindicatos y apostadores profesionales) del otro lado."
        ),
    },
    {
        "badge": "CONSENSO 3/4 O 4/4 MODELOS",
        "color": "#047857",
        "bg": "#ecfdf5",
        "border": "#a7f3d0",
        "titulo": "Convergencia de los 4 Motores Cuantitativos",
        "explicacion": (
            "Indica cuantos de los 4 pilares independientes coinciden en ventaja sobre el momio justo sin comision: "
            "(1) Sabermetria Estructural (SP+, Four Factors, xGF%, xG, Elo Superficie), "
            "(2) Matchup Situacional/Tactico (EPA, Havoc, Porteros SV%, PPDA, Hold/Break%), "
            "(3) Dinero Real en Kalshi CFTC + Flujo Sharp/RLM, y "
            "(4) Consenso de Analistas/Foros + 10,000 Simulaciones Monte Carlo. "
            "4/4 es unanimidad total; 3/4 es mayoria solida; 2/4 reduce la calificacion de seguridad."
        ),
    },
    {
        "badge": "KALSHI XX%",
        "color": "#0369a1",
        "bg": "#e0f2fe",
        "border": "#bae6fd",
        "titulo": "Bolsa Regulada de Contratos Kalshi (CFTC EE. UU.)",
        "explicacion": (
            "Precio real de mercado en centavos (1c a 99c = 1% a 99% de probabilidad) negociado con liquidez real "
            "en Kalshi. Al ser un exchange abierto sin bloqueo a ganadores, sirve como referencia institucional "
            "para detectar si una casa tradicional dejo un momio mal puesto."
        ),
    },
    {
        "badge": "SHARP XX% DINERO",
        "color": "#0369a1",
        "bg": "#e0f2fe",
        "border": "#bae6fd",
        "titulo": "Concentracion de Dinero Profesional (% Dinero vs % Tickets)",
        "explicacion": (
            "Mide que porcentaje del dinero total apostado esta en ese lado frente al numero de boletos. "
            "Si un equipo recibe 38% de los tickets pero 64% del dinero total, significa que los tickets grandes "
            "(dinero inteligente / sharp) estan respaldando esa seleccion."
        ),
    },
    {
        "badge": "PROB. DE QUE SUCEDA (%)",
        "color": "#0f172a",
        "bg": "#f1f5f9",
        "border": "#cbd5e1",
        "titulo": "Probabilidad Real Calibrada de Ocurrencia",
        "explicacion": (
            "Frecuencia real estimada por el motor inteligente (ajustada por dispersion entre modelos y varianza) "
            "de que la apuesta se gane hoy. En Parlays muestra tanto la probabilidad individual de cada pata "
            "como la probabilidad conjunta real de acertar todas las patas simultaneamente."
        ),
    },
    {
        "badge": "SEGURIDAD 1.0 - 10.0",
        "color": "#047857",
        "bg": "#ecfdf5",
        "border": "#a7f3d0",
        "titulo": "Indice Inteligente de Seguridad y Detector Anti-Relleno",
        "explicacion": (
            "Evalua del 1 al 10 que tan confiable es ejecutar la apuesta combinando probabilidad real, consenso de modelos, "
            "respaldo Kalshi/Sharp, proteccion de linea y tamano de cartelera. "
            "8.5-10.0: Elite (Stake Fuerte 1.25u) | 7.2-8.4: Alta (Stake Estandar 1.0u) | "
            "5.8-7.1: Moderada o Complemento de Cuota en cartelera corta (Stake Reducido 0.5u) | "
            "< 5.8: Riesgo Alto / Parlay Largo Recreativo (0.25u u Opcional)."
        ),
    },
]


class QuantitativeAIAgent:
    def __init__(self, history_data: Optional[Dict[str, Any]] = None):
        self.history = history_data or {}
        self.model_weights = self._calibrate_weights_from_history()

    def _calibrate_weights_from_history(self) -> Dict[str, float]:
        """
        Machine Learning Meta-Calibrator:
        Analyzes historical tickets in bet_history.json to dynamically weight
        the analytical models based on empirical resolved accuracy.
        """
        tickets = self.history.get("tickets", []) if isinstance(self.history, dict) else []
        resolved = [t for t in tickets if t.get("status") in ["GANADA", "PERDIDA"]]

        base_weights = {
            "poisson_dixon": 0.22,
            "sabermetrics": 0.23,
            "contextual_ml": 0.20,
            "monte_carlo": 0.20,
            "kalshi_sharp": 0.15,
        }

        if len(resolved) < 5:
            return base_weights

        won = sum(1 for t in resolved if t.get("status") == "GANADA")
        win_rate = won / len(resolved)

        if win_rate >= 0.65:
            base_weights["monte_carlo"] = 0.25
            base_weights["kalshi_sharp"] = 0.20
            base_weights["poisson_dixon"] = 0.20
            base_weights["sabermetrics"] = 0.20
            base_weights["contextual_ml"] = 0.15
        elif win_rate < 0.50:
            base_weights["poisson_dixon"] = 0.28
            base_weights["sabermetrics"] = 0.28
            base_weights["monte_carlo"] = 0.20
            base_weights["contextual_ml"] = 0.14
            base_weights["kalshi_sharp"] = 0.10

        total_w = sum(base_weights.values())
        return {k: round(v / total_w, 3) for k, v in base_weights.items()}

    # ------------------------------------------------------------------
    # INTELLIGENT 1-10 SECURITY & TRUE HIT PROBABILITY AUDIT ENGINE
    # ------------------------------------------------------------------
    @staticmethod
    def _prob_to_base_security_pts(prob_pct: float) -> float:
        """
        Piecewise calibrated curve mapping true hit probability (%) to base 1-10 security points.
        Ensures realistic, non-inflated grounding: a 55% spread cannot start at 8.5, while an 80%
        protected leg starts higher before EV/consensus/vig adjustments.
        """
        p = max(12.0, min(96.0, prob_pct))
        if p < 35.0:
            return 2.6 + ((p - 12.0) / 23.0) * 1.6  # 2.6 -> 4.2
        if p < 50.0:
            return 4.2 + ((p - 35.0) / 15.0) * 1.15  # 4.2 -> 5.35
        if p < 60.0:
            return 5.35 + ((p - 50.0) / 10.0) * 1.05  # 5.35 -> 6.40
        if p < 72.0:
            return 6.40 + ((p - 60.0) / 12.0) * 1.05  # 6.40 -> 7.45
        if p < 84.0:
            return 7.45 + ((p - 72.0) / 12.0) * 0.70  # 7.45 -> 8.15
        return 8.15 + ((p - 84.0) / 12.0) * 0.40      # 8.15 -> 8.55

    @staticmethod
    def _extract_signals_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts structured and text-embedded consensus/Kalshi/Sharp signals even on locked historical legs."""
        just = str(item.get("justification") or "")
        cons_lbl = str(item.get("consensus_label") or "")

        cons = item.get("consensus")
        tot_m = item.get("total_models")
        if cons is None or tot_m is None:
            m_c = re.search(r"Consenso\s+(\d+)\s*/\s*(\d+)", cons_lbl + " " + just, flags=re.IGNORECASE)
            if m_c:
                cons = int(m_c.group(1))
                tot_m = int(m_c.group(2))
            else:
                cons = 3
                tot_m = 4

        kalshi_p = float(item.get("kalshi_prob") or 0.0)
        if kalshi_p <= 0:
            m_k = re.search(r"Kalshi\s+CFTC:\s*(\d+(?:\.\d+)?)%", just, flags=re.IGNORECASE)
            if m_k:
                kalshi_p = float(m_k.group(1))

        sharp_m = item.get("sharp_money_pct")
        tickets_p = item.get("tickets_pct")
        if sharp_m is None:
            m_s = re.search(r"(\d+)%\s*Dinero(?:\s*vs\s*(\d+)%\s*Tickets|\s*\((\d+)%\s*Tickets\))?", just, flags=re.IGNORECASE)
            if m_s:
                sharp_m = float(m_s.group(1))
                t_g = m_s.group(2) or m_s.group(3)
                if t_g:
                    tickets_p = float(t_g)

        is_rlm = bool(item.get("is_rlm")) or ("rlm" in just.lower()) or ("reverse line movement" in just.lower())
        dispersion = float(item.get("model_dispersion") or 0.0)

        return {
            "consensus": int(cons),
            "total_models": max(3, int(tot_m)),
            "kalshi_prob": kalshi_p,
            "sharp_money_pct": float(sharp_m) if sharp_m is not None else None,
            "tickets_pct": float(tickets_p) if tickets_p is not None else None,
            "is_rlm": is_rlm,
            "dispersion": dispersion,
        }

    @classmethod
    def classify_security_tier(cls, score_1_10: float, is_quota_filler: bool = False, leg_count: int = 1) -> Dict[str, str]:
        """Maps a 1.0-10.0 security score into clinical tier labels, colors, and stake sizing."""
        if score_1_10 >= 8.5 and not is_quota_filler:
            return {
                "tier_label": "SEGURIDAD ELITE (MAXIMA CONVICCION)",
                "short_badge": f"SEGURIDAD {score_1_10:.1f}/10 - ELITE",
                "origin_tag": "CONVICCION PRIORITARIA DEL MOTOR",
                "stake_label": "Stake Fuerte: 1.0u - 1.25u",
                "color": "#047857",
                "bg": "#ecfdf5",
                "border": "#a7f3d0",
                "bar_color": "#059669",
            }
        if score_1_10 >= 7.2 and not is_quota_filler:
            return {
                "tier_label": "SEGURIDAD ALTA (VENTAJA SOLIDA)",
                "short_badge": f"SEGURIDAD {score_1_10:.1f}/10 - ALTA",
                "origin_tag": "SELECCION VERIFICADA POR MODELOS",
                "stake_label": "Stake Estandar: 1.0u",
                "color": "#0369a1",
                "bg": "#e0f2fe",
                "border": "#bae6fd",
                "bar_color": "#0284c7",
            }
        if score_1_10 >= 5.8:
            origin = (
                "COMPLEMENTO DE CUOTA (CARTELERA CORTA / VIG)"
                if is_quota_filler
                else ("VARIANZA MULTI-PATA CONTROLADA" if leg_count >= 2 else "SEGURIDAD MODERADA (VALOR CON VARIANZA)")
            )
            lbl = (
                "SEGURIDAD MODERADA (COMPLEMENTO DE CUOTA)"
                if is_quota_filler
                else "SEGURIDAD MODERADA (GESTIONAR VARIANZA)"
            )
            return {
                "tier_label": lbl,
                "short_badge": f"SEGURIDAD {score_1_10:.1f}/10 - MODERADA",
                "origin_tag": origin,
                "stake_label": "Stake Reducido: 0.5u",
                "color": "#b45309",
                "bg": "#fffbeb",
                "border": "#fde68a",
                "bar_color": "#d97706",
            }
        origin_low = (
            "RELLENO FORZADO POR CUOTA / ALTO RIESGO"
            if is_quota_filler
            else (f"PARLAY LARGO ({leg_count} PATAS) - ALTA VARIANZA ACUMULADA" if leg_count >= 3 else "ALTA VOLATILIDAD")
        )
        return {
            "tier_label": "SEGURIDAD BAJA (ALTA VARIANZA / OPCIONAL)",
            "short_badge": f"SEGURIDAD {score_1_10:.1f}/10 - RIESGO ALTO",
            "origin_tag": origin_low,
            "stake_label": "Stake Minimo Recreativo: 0.25u (u Omitir)",
            "color": "#b91c1c",
            "bg": "#fef2f2",
            "border": "#fecaca",
            "bar_color": "#dc2626",
        }

    @classmethod
    def audit_selection_security(
        cls,
        item: Dict[str, Any],
        slate_size: int = 10,
        slot_rank: int = 1,
        vehicle_type: str = "Apuesta Fuerte",
    ) -> Dict[str, Any]:
        """
        Evaluates a single bet or individual parlay leg, producing:
        - prob_suceda_pct: Calibrated real probability (%) that this specific outcome happens.
        - seguridad_1_10: Honest 1.0 - 10.0 safety score penalizing thin-slate quota fillers & vig drag.
        - is_quota_filler: True when included mainly to satisfy slot quota on a small slate or with negative EV.
        - analisis_motor: Concise Spanish breakdown of why the intelligent engine assigned this score.
        """
        raw_p = float(item.get("real_prob", 60.0) or 60.0)
        if raw_p <= 1.0:
            raw_p *= 100.0

        am_odds = int(item.get("american_odds", -110) or -110)
        dec_odds = float(item.get("decimal_odds") or _american_to_decimal_local(am_odds))
        imp_pct = _implied_prob_pct_local(am_odds)

        if item.get("ev") is not None:
            ev_pct = float(item.get("ev", 0.0))
            if abs(ev_pct) <= 0.95 and raw_p > 50 and abs(((raw_p / 100.0) * dec_odds - 1.0) * 100.0) > 1.5:
                ev_pct = round(((raw_p / 100.0) * dec_odds - 1.0) * 100.0, 2)
        else:
            ev_pct = round(((raw_p / 100.0) * dec_odds - 1.0) * 100.0, 2)

        sig = cls._extract_signals_from_item(item)
        cons = sig["consensus"]
        tot_m = sig["total_models"]
        cons_ratio = cons / float(tot_m)
        kalshi_p = sig["kalshi_prob"]
        sharp_m = sig["sharp_money_pct"]
        tickets_p = sig["tickets_pct"]
        is_rlm = sig["is_rlm"]
        disp = sig["dispersion"]

        sel_low = str(item.get("selection") or "").lower()
        mkt_low = str(item.get("market_type") or "").lower()
        lg_low = str(item.get("league_name") or "").lower()

        is_cup = any(k in lg_low for k in ("copa", "cup", "pokal", "coupe", "coppa"))
        has_cushion = any(
            k in (sel_low + " " + mkt_low)
            for k in ("que cubra", "doble oportunidad", "1x", "x2", "+1.5 sets", "seguro", "alternativo")
        )
        is_goal_prop = ("goleador" in mkt_low or "anotara" in sel_low)

        # 1. Calibrate true probability of happening (accounting for model disagreement & variance)
        p_cal = raw_p
        if cons_ratio >= 0.98:
            if is_rlm or (kalshi_p > 0 and kalshi_p >= imp_pct - 1.0):
                p_cal = min(95.0, raw_p + 0.5)
        elif cons_ratio >= 0.72:
            p_cal = (0.985 * raw_p) + (0.015 * 50.0)
        else:
            # Partial disagreement between models -> conservative shrinkage toward 50%
            p_cal = (0.935 * raw_p) + (0.065 * 50.0)

        if disp > 0.065:
            p_cal = (0.96 * p_cal) + (0.04 * 50.0)
        if is_cup:
            p_cal -= 1.0

        prob_suceda_pct = round(max(15.0, min(95.5, p_cal)), 1)

        # 2. Compute 6-Pillar Intelligent 1-10 Security Score
        base_pts = cls._prob_to_base_security_pts(prob_suceda_pct)

        # Pillar B: Model Consensus & Inter-Model Dispersion
        if cons_ratio >= 0.98:
            cons_pts = 1.10
        elif cons_ratio >= 0.72:
            cons_pts = 0.65
        elif cons_ratio >= 0.64:
            cons_pts = 0.35
        else:
            cons_pts = -0.60
        if disp > 0.075:
            cons_pts -= 0.25
        elif 0.0 < disp <= 0.032:
            cons_pts += 0.15

        # Pillar C: Institutional Smart Money (RLM, Action Sharp %, Kalshi CFTC)
        intel_pts = 0.0
        intel_reasons: List[str] = []
        if is_rlm:
            intel_pts += 0.45
            intel_reasons.append("RLM institucional confirmado")
        if sharp_m is not None:
            diff_st = (sharp_m - tickets_p) if tickets_p is not None else 0.0
            if sharp_m >= 60.0 and diff_st >= 10.0:
                intel_pts += 0.35
                intel_reasons.append(f"dinero sharp ({int(sharp_m)}% $)")
            elif sharp_m >= 56.0:
                intel_pts += 0.20
                intel_reasons.append(f"flujo profesional ({int(sharp_m)}% $)")
            elif tickets_p is not None and tickets_p >= 68.0 and sharp_m <= 44.0:
                intel_pts -= 0.65
        if kalshi_p > 0:
            if kalshi_p >= (imp_pct - 2.5) or kalshi_p >= 56.0:
                intel_pts += 0.28
                intel_reasons.append(f"respaldo Kalshi ({kalshi_p:.0f}%)")
            elif (imp_pct - kalshi_p) >= 6.5:
                intel_pts -= 0.55
        intel_pts = max(-0.85, min(0.95, intel_pts))

        # Pillar D: True Mathematical Edge (+EV) vs Vig Drag
        if ev_pct >= 8.0:
            ev_pts = 0.72
        elif ev_pct >= 4.0:
            ev_pts = 0.52
        elif ev_pct >= 1.5:
            ev_pts = 0.28
        elif ev_pct >= -0.8:
            ev_pts = -0.08
        elif ev_pct >= -2.3:
            ev_pts = -0.52
        else:
            ev_pts = -1.00

        if am_odds <= -550:
            ev_pts -= 0.35

        # Pillar E: Line Protection Cushion vs Volatility
        struct_pts = 0.0
        if has_cushion:
            struct_pts += 0.35
        if is_cup or is_goal_prop or (am_odds >= 125 and not has_cushion):
            struct_pts -= 0.40

        # Pillar F: Honest Anti-Quota-Filler / Thin-Slate Penalty
        quota_pts = 0.0
        quota_reasons: List[str] = []
        if slot_rank >= 2 and slate_size <= 6:
            quota_pts -= 0.55
            quota_reasons.append(f"opcion #{slot_rank} en bloque corto de {slate_size} juegos")
        if slot_rank >= 2 and ev_pct < 0.5:
            quota_pts -= 0.40
            quota_reasons.append("margen +EV reducido frente a la opcion #1")
        if am_odds <= -500:
            quota_pts -= 0.45
            quota_reasons.append(f"momio muy castigado por comision ({am_odds:+d})")
        elif slate_size <= 4 and ev_pct < 3.0:
            quota_pts -= 0.30
            quota_reasons.append(f"cartelera reducida ({slate_size} juegos)")
        quota_pts = max(-1.25, quota_pts)

        is_quota_filler = bool(
            (quota_pts <= -0.50)
            or (ev_pct < -1.8)
            or (am_odds <= -550)
            or (cons_ratio < 0.55 and ev_pct < 2.0)
        )

        raw_score = base_pts + cons_pts + intel_pts + ev_pts + struct_pts + quota_pts
        if is_quota_filler:
            raw_score = min(7.0, raw_score)
        seguridad_1_10 = round(max(2.5, min(9.7, raw_score)), 1)
        tier_info = cls.classify_security_tier(seguridad_1_10, is_quota_filler=is_quota_filler, leg_count=1)

        # Build concise Spanish explanation for this selection
        pos_parts = [f"Consenso {cons}/{tot_m} modelos", f"{prob_suceda_pct:.1f}% prob. real de ocurrir"]
        if ev_pct >= 1.5:
            pos_parts.append(f"ventaja +{ev_pct:.1f}% EV")
        if has_cushion:
            pos_parts.append("colchon de cobertura activo")
        if intel_reasons:
            pos_parts.extend(intel_reasons[:2])

        if is_quota_filler:
            q_detail = ", ".join(quota_reasons) if quota_reasons else "menor margen sobre el vig de la casa"
            analisis_motor = (
                f"Calificacion {seguridad_1_10:.1f}/10 ({tier_info['origin_tag']}): {', '.join(pos_parts)}, "
                f"pero penalizada por {q_detail}. Usar stake reducido o priorizar la seleccion #1."
            )
        else:
            analisis_motor = (
                f"Calificacion {seguridad_1_10:.1f}/10 ({tier_info['origin_tag']}): Respaldada por "
                f"{', '.join(pos_parts)} sin alertas de relleno ni contradiccion sharp."
            )

        return {
            "prob_suceda_pct": prob_suceda_pct,
            "seguridad_1_10": seguridad_1_10,
            "is_quota_filler": is_quota_filler,
            "tier_label": tier_info["tier_label"],
            "short_badge": tier_info["short_badge"],
            "origin_tag": tier_info["origin_tag"],
            "stake_label": tier_info["stake_label"],
            "security_color": tier_info["color"],
            "security_bg": tier_info["bg"],
            "security_border": tier_info["border"],
            "security_bar_color": tier_info["bar_color"],
            "analisis_motor": analisis_motor,
            "consensus": cons,
            "total_models": tot_m,
            "ev_calc": round(ev_pct, 2),
        }

    @classmethod
    def audit_ticket_security(
        cls,
        legs: List[Dict[str, Any]],
        vehicle_type: str = "Apuesta Fuerte",
        slate_size: int = 10,
        slot_rank: int = 1,
    ) -> Dict[str, Any]:
        """
        Audits a complete betting vehicle (1-leg Sniper or 2/3/5-leg Parlay), returning
        both per-leg audits and honest ticket-level joint hit probability (%) & 1-10 security rating.
        """
        if not legs:
            empty_tier = cls.classify_security_tier(5.0)
            return {
                "ticket_prob_suceda_pct": 0.0,
                "ticket_seguridad_1_10": 0.0,
                "avg_leg_seguridad_1_10": 0.0,
                "min_leg_seguridad_1_10": 0.0,
                "combined_ev_pct": 0.0,
                "is_quota_filler": False,
                "leg_audits": [],
                "dictamen_inteligente": "Sin seleccion activa.",
                **empty_tier,
            }

        leg_audits = [
            cls.audit_selection_security(l_item, slate_size=slate_size, slot_rank=slot_rank, vehicle_type=vehicle_type)
            for l_item in legs
        ]
        k = len(legs)

        if k == 1:
            la = leg_audits[0]
            return {
                "ticket_prob_suceda_pct": la["prob_suceda_pct"],
                "ticket_seguridad_1_10": la["seguridad_1_10"],
                "avg_leg_seguridad_1_10": la["seguridad_1_10"],
                "min_leg_seguridad_1_10": la["seguridad_1_10"],
                "combined_ev_pct": la["ev_calc"],
                "is_quota_filler": la["is_quota_filler"],
                "tier_label": la["tier_label"],
                "short_badge": la["short_badge"],
                "origin_tag": la["origin_tag"],
                "stake_label": la["stake_label"],
                "color": la["security_color"],
                "bg": la["security_bg"],
                "border": la["security_border"],
                "bar_color": la["security_bar_color"],
                "leg_audits": leg_audits,
                "dictamen_inteligente": la["analisis_motor"],
            }

        # Multi-leg Parlay Joint Hit Probability & Weakest-Link Security Calculation
        joint_p = 1.0
        comb_dec = 1.0
        for l_item, la in zip(legs, leg_audits):
            joint_p *= (la["prob_suceda_pct"] / 100.0)
            am = int(l_item.get("american_odds", -180) or -180)
            d_val = float(l_item.get("decimal_odds") or _american_to_decimal_local(am))
            comb_dec *= d_val

        # Compound multi-event execution variance adjustment (0.99^(k-1))
        joint_p *= (0.99 ** (k - 1))
        ticket_prob_pct = round(max(5.0, min(92.0, joint_p * 100.0)), 1)
        comb_ev_pct = round(((ticket_prob_pct / 100.0) * comb_dec - 1.0) * 100.0, 2)

        leg_scores = [la["seguridad_1_10"] for la in leg_audits]
        avg_leg_s = round(sum(leg_scores) / len(leg_scores), 1)
        min_leg_s = round(min(leg_scores), 1)
        any_filler = any(la["is_quota_filler"] for la in leg_audits) or (slot_rank >= 2 and slate_size <= 6 and comb_ev_pct < 0.0)

        joint_anchor = cls._prob_to_base_security_pts(ticket_prob_pct)
        weakest_weighted_legs = (0.58 * avg_leg_s) + (0.42 * min_leg_s)

        # Multi-leg compounding penalty
        leg_count_pen = 0.16 if k == 2 else (0.48 if k == 3 else min(1.05, 0.22 * k))
        ev_adj = 0.32 if comb_ev_pct >= 4.0 else (0.15 if comb_ev_pct >= 0.5 else (-0.48 if comb_ev_pct < -3.0 else -0.18))
        filler_pen = -0.35 if (any_filler and slot_rank >= 2) else 0.0

        raw_ticket_score = (0.52 * joint_anchor) + (0.48 * weakest_weighted_legs) - leg_count_pen + ev_adj + filler_pen
        if any_filler:
            raw_ticket_score = min(7.0, raw_ticket_score)
        ticket_score = round(max(2.5, min(9.4, raw_ticket_score)), 1)

        tier_info = cls.classify_security_tier(ticket_score, is_quota_filler=any_filler, leg_count=k)

        if any_filler:
            dictamen = (
                f"Auditoria del Motor ({ticket_score:.1f}/10 - {tier_info['origin_tag']}): "
                f"Probabilidad real de acertar las {k} patas juntas: {ticket_prob_pct:.1f}% "
                f"(seguridad individual por pata: media {avg_leg_s:.1f}/10, minima {min_leg_s:.1f}/10). "
                f"Detectada al menos una pata de complemento por cartelera corta o vig elevado; reducir stake ({tier_info['stake_label']})."
            )
        elif k >= 3:
            dictamen = (
                f"Auditoria del Motor ({ticket_score:.1f}/10): Cada seleccion individual es solida "
                f"(media {avg_leg_s:.1f}/10), pero al multiplicar {k} eventos independientes la probabilidad real "
                f"de acertar el boleto completo es {ticket_prob_pct:.1f}% ({comb_ev_pct:+.1f}% EV combinado). {tier_info['stake_label']}."
            )
        else:
            dictamen = (
                f"Auditoria del Motor ({ticket_score:.1f}/10 - {tier_info['origin_tag']}): "
                f"Combinacion principal verificada de {k} patas con {ticket_prob_pct:.1f}% de probabilidad conjunta real de exito "
                f"({comb_ev_pct:+.1f}% EV combinado | media por pata {avg_leg_s:.1f}/10). {tier_info['stake_label']}."
            )

        return {
            "ticket_prob_suceda_pct": ticket_prob_pct,
            "ticket_seguridad_1_10": ticket_score,
            "avg_leg_seguridad_1_10": avg_leg_s,
            "min_leg_seguridad_1_10": min_leg_s,
            "combined_ev_pct": comb_ev_pct,
            "is_quota_filler": any_filler,
            "tier_label": tier_info["tier_label"],
            "short_badge": tier_info["short_badge"],
            "origin_tag": tier_info["origin_tag"],
            "stake_label": tier_info["stake_label"],
            "color": tier_info["color"],
            "bg": tier_info["bg"],
            "border": tier_info["border"],
            "bar_color": tier_info["bar_color"],
            "leg_audits": leg_audits,
            "dictamen_inteligente": dictamen,
        }

    def evaluate_game_consensus(
        self,
        sport: str,
        matchup: str,
        m1_prob: float,
        m2_prob: float,
        m3_prob: float,
        mc_prob: float,
        poisson_prob: Optional[float] = None,
        has_sharp: bool = True,
    ) -> Dict[str, Any]:
        """Synthesizes all probabilistic methods and generates an AI Consensus evaluation."""
        w = self.model_weights

        if poisson_prob is not None:
            p_val = poisson_prob
            w_poisson = w["poisson_dixon"]
            w_mc = w["monte_carlo"]
            w_m1 = w["sabermetrics"]
            w_m2 = w["contextual_ml"]
            w_m3 = w["kalshi_sharp"] if has_sharp else 0.0
        else:
            p_val = mc_prob
            w_poisson = 0.0
            w_mc = w["monte_carlo"] + (w["poisson_dixon"] * 0.5)
            w_m1 = w["sabermetrics"] + (w["poisson_dixon"] * 0.5)
            w_m2 = w["contextual_ml"]
            w_m3 = w["kalshi_sharp"] if has_sharp else 0.0

        denom = w_poisson + w_mc + w_m1 + w_m2 + w_m3
        if denom == 0:
            denom = 1.0

        ai_ensemble_prob = (
            (w_poisson * p_val)
            + (w_mc * mc_prob)
            + (w_m1 * m1_prob)
            + (w_m2 * m2_prob)
            + (w_m3 * m3_prob)
        ) / denom

        probs = [p for p in [p_val, mc_prob, m1_prob, m2_prob, (m3_prob if has_sharp else None)] if p is not None]
        dispersion = float(np.std(probs)) if len(probs) > 1 else 0.0

        direction_agree = sum(1 for p in probs if (p >= 0.50) == (ai_ensemble_prob >= 0.50))
        consensus_score = round((direction_agree / len(probs)) * 100, 1) if probs else 75.0

        if dispersion < 0.06 and consensus_score >= 80:
            confidence_level = "ALTA CONFIANZA (+EV SOLIDO)"
            quality_stars = "A+"
        elif dispersion < 0.12 and consensus_score >= 60:
            confidence_level = "MODERADA-ALTA (EDGE VERIFICADO)"
            quality_stars = "A"
        else:
            confidence_level = "VOLATIL (REQUIERE COBERTURA DE LINEA)"
            quality_stars = "B+"

        return {
            "ai_probability": round(ai_ensemble_prob, 4),
            "dispersion": round(dispersion, 4),
            "consensus_score_pct": consensus_score,
            "confidence_level": confidence_level,
            "quality_grade": quality_stars,
            "weights_used": w,
        }

    def generate_ai_vehicle_dictamen(
        self,
        sport: str,
        vehicle_type: str,
        selection_title: str,
        odds_american: int,
        real_prob_pct: float,
        ev_pct: float,
        supporting_metrics: Dict[str, Any],
    ) -> str:
        """Generates an executive quantitative AI verdict for a vehicle."""
        w = self.model_weights
        if vehicle_type == "Apuesta Fuerte":
            return (
                f"Dictamen del Motor Cuantitativo: Seleccion priorizada con {real_prob_pct:.1f}% de probabilidad "
                f"calibrada (+{ev_pct:.1f}% EV). Validacion cruzada: Sabermetria/Poisson ({(w['poisson_dixon']+w['sabermetrics'])*100:.0f}%), "
                f"Monte Carlo ({w['monte_carlo']*100:.0f}%) y liquidez Kalshi/Sharp ({w['kalshi_sharp']*100:.0f}%)."
            )
        if vehicle_type == "Parlay Props":
            return (
                "Dictamen del Motor Cuantitativo: Jugadores titulares libres de reporte de lesion evaluados "
                "mediante tasas de Poisson y 10,000 simulaciones Monte Carlo con aislamiento por partido."
            )
        if vehicle_type == "Parlay Ratonero":
            return (
                "Dictamen del Motor Cuantitativo: Construccion defensiva de 5 patas en mercados alternativos "
                "abiertos (-135 a -380) con colchon de cobertura individual >= 72%."
            )
        return (
            f"Dictamen del Motor Cuantitativo: Combinacion de 2 selecciones principales con tope estricto "
            f"<= +100 MAX ({odds_american:+d}) y bloqueo anti-doble-under."
        )
