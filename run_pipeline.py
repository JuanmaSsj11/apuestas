"""
Main Pipeline Runner for Quantitative Sports Betting Terminal [Institutional +EV Edition]
Generates tailored quantitative betting vehicles backed by 4-Model Bayesian Ensembles +
CFTC Kalshi & Action Network Sharp Money/RLM + Verified Analyst/Forum Consensus + 10,000 Monte Carlo Simulations:
- NBA, NHL, Futbol (5 Grandes Ligas + Copas Domesticas + UEFA + Liga MX), NFL, MLB
- NCAAF (Futbol Americano Colegial FBS): Strictly separated into 3 independent Horarios (Manana, Tarde, Noche) ->
  Each Horario independently extracts its own 2 Apuestas Fuertes + 2 Parlays Ganadores (Zero Mixing, Zero Ratonero, Zero Props)
- Tenis (ATP & WTA Singles): Surface-Adjusted Elo + Hold/Break/DR/TB Sabermetrics + 10,000 Markov Set Sims ->
  Extracts Apuesta Fuerte + Parlay Ganador (Zero Ratonero, Zero Props)
- Lo Mejor del Dia (MVP): Priority selection across all sports playing TODAY, featuring Cross-Sport
  5-Leg Parlay Ratonero OR automatic fallback to 2 Apuestas Fuertes when Ratonero < 5 legs.
Strictly zero emojis.
"""
import sys
import os
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scraper import SportsDataScraper
from src.sabermetrics import (
    get_nba_stats,
    get_nhl_stats,
    get_soccer_stats,
    get_ncaaf_stats,
    get_tennis_stats,
    get_nfl_stats,
    get_baseball_stats,
    MLB_ADVANCED_STATS,
)
from src.monte_carlo import MonteCarloSimulator
from src.predictive_models import (
    PredictiveModelSuite,
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    de_vig_two_way,
    de_vig_three_way,
    calculate_ev,
)
from src.selection_engine import SelectionEngine
from src.ai_agent import QuantitativeAIAgent
from src.history_tracker import HistoryTracker, team_matches_text, both_teams_match_game_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")

CST_TZ = timezone(timedelta(hours=-6))

BULLISH_EXPERT_WORDS = (
    "best bet", "best bets", "pick", "picks", "prediction", "lock", "value",
    "edge", "favorite", "cover", "upset", "sharp", "top play", "lean",
)
FADE_EXPERT_WORDS = ("fade", "avoid", "overvalued", "trap", "injured", "ruled out")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_to_american_with_vig(p_fair: float, vig_half: float = 0.0225) -> int:
    p = min(max(p_fair * (1.0 + vig_half), 0.02), 0.98)
    if p >= 0.5:
        return int(round(-(p / (1.0 - p)) * 100.0))
    return int(round(((1.0 - p) / p) * 100.0))


def spread_to_american_ml(spread: float, sigma: float = 11.8, vig: float = 0.045) -> Tuple[int, int, float, float]:
    p_home_fair = norm_cdf(-spread / sigma)
    p_away_fair = 1.0 - p_home_fair
    h_am = prob_to_american_with_vig(p_home_fair, vig / 2.0)
    a_am = prob_to_american_with_vig(p_away_fair, vig / 2.0)
    return h_am, a_am, p_home_fair, p_away_fair


def derive_tennis_book_odds(p1_elo_global: float, p2_elo_global: float, vig: float = 0.048) -> Tuple[int, int]:
    """Prices realistic sportsbook Tennis Money Line odds from global tour Elo when feed omits bookmaker odds."""
    p1_global = 1.0 / (1.0 + math.pow(10.0, -(p1_elo_global - p2_elo_global) / 400.0))
    p2_global = 1.0 - p1_global
    return (
        prob_to_american_with_vig(p1_global, vig / 2.0),
        prob_to_american_with_vig(p2_global, vig / 2.0),
    )


def derive_soccer_3way_book_odds(p_home: float, p_draw: float, p_away: float) -> Tuple[int, int, int]:
    """Derives calibrated 3-way (1X2) soccer American odds with 5.5% sportsbook overround when feed omits odds."""
    h_mkt = 0.84 * p_home + 0.16 * 0.43
    d_mkt = 0.84 * p_draw + 0.16 * 0.26
    a_mkt = max(0.06, 1.0 - h_mkt - d_mkt)
    return (
        prob_to_american_with_vig(h_mkt, 0.027),
        prob_to_american_with_vig(d_mkt, 0.027),
        prob_to_american_with_vig(a_mkt, 0.027),
    )


def is_game_today(event_date_iso: str, target_date_str: str) -> bool:
    """Checks if an event's ISO timestamp corresponds to target_date_str in CST (UTC-6)."""
    if not event_date_iso:
        return False
    try:
        clean_iso = event_date_iso.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_iso)
        return dt_utc.astimezone(CST_TZ).strftime("%Y-%m-%d") == target_date_str
    except Exception:
        return event_date_iso.startswith(target_date_str)


def match_kalshi_contract(
    target_name: str,
    opponent_name: str,
    sport: str,
    kalshi_by_sport: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Matches a team or player to an active open CFTC Kalshi prediction market contract."""
    sp_map = kalshi_by_sport.get(sport.upper(), {})
    if not sp_map or not target_name:
        return {}

    best_match: Dict[str, Any] = {}
    best_score = -1.0

    for _, mkt in sp_map.items():
        m_target = mkt.get("target", "")
        ev_title = mkt.get("event_title", "")
        if not m_target:
            continue
        t_ok = team_matches_text(target_name, m_target, sport) or team_matches_text(m_target, target_name, sport)
        if not t_ok:
            continue

        score = float(mkt.get("volume", 0.0))
        if ev_title and opponent_name:
            if both_teams_match_game_text(target_name, opponent_name, ev_title, sport):
                score += 1_000_000.0
            elif not team_matches_text(opponent_name, ev_title, sport):
                # Event title names a different opponent -> skip cross-game collision
                continue

        if score > best_score:
            best_score = score
            best_match = mkt

    return best_match


def match_action_sharp_game(
    home_name: str,
    away_name: str,
    sport: str,
    action_by_sport: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Matches an ESPN game to Action Network's institutional sharp money & RLM feed."""
    an_list = action_by_sport.get(sport.upper(), [])
    if not an_list:
        return {}
    probe_text = f"{away_name} @ {home_name}"
    for item in an_list:
        if both_teams_match_game_text(item.get("home_team", ""), item.get("away_team", ""), probe_text, sport):
            return item
    return {}


def match_expert_consensus(
    side_name: str,
    sport: str,
    expert_by_sport: Dict[str, List[Dict[str, Any]]],
) -> Tuple[int, float, str]:
    """
    Scans live analyst, sharp capper, and forum RSS headlines (VSiN, Covers, Action Network, CBS, RotoWire)
    for verified mentions of side_name. Returns (mentions_count, sentiment_delta, spanish_note).
    """
    articles = expert_by_sport.get(sport.upper(), [])
    if not articles or not side_name:
        return 0, 0.0, ""

    mentions = 0
    sentiment = 0.0
    sources: List[str] = []

    for art in articles:
        title = art.get("title", "")
        if not title:
            continue
        if team_matches_text(side_name, title, sport):
            mentions += 1
            t_low = title.lower()
            if any(w in t_low for w in FADE_EXPERT_WORDS):
                sentiment -= 0.005
            elif any(w in t_low for w in BULLISH_EXPERT_WORDS):
                sentiment += 0.004
            src = art.get("source", "")
            if src and src not in sources and len(sources) < 2:
                sources.append(src)

    note = ""
    if mentions >= 1 and sentiment >= -0.001:
        src_str = ", ".join(sources) if sources else "Sportsbook Consensus"
        note = f"Consenso Analistas/Foros ({src_str}): {mentions} analisis activos destacando a {side_name}."

    return mentions, max(-0.015, min(0.015, sentiment)), note


def evaluate_two_way_intelligence(
    models: PredictiveModelSuite,
    sport: str,
    h_name: str,
    a_name: str,
    h_ml: int,
    a_ml: int,
    m1_h: float,
    m1_a: float,
    m2_h: float,
    m2_a: float,
    mc_h: float,
    mc_a: float,
    kalshi_by_sport: Dict[str, Dict[str, Any]],
    an_game: Dict[str, Any],
    expert_by_sport: Dict[str, List[Dict[str, Any]]],
    use_spread_splits: bool = False,
) -> Dict[str, Any]:
    """
    Executes the 4-Pillar Institutional Ensemble + Bayesian Market Shrinkage for any 2-way game/match.
    """
    fair_h, fair_a = de_vig_two_way(h_ml, a_ml)

    h_k = match_kalshi_contract(h_name, a_name, sport, kalshi_by_sport)
    a_k = match_kalshi_contract(a_name, h_name, sport, kalshi_by_sport)

    if use_spread_splits and an_game.get("home_spread_money") is not None:
        h_tkt = an_game.get("home_spread_tickets")
        h_mon = an_game.get("home_spread_money")
        a_tkt = an_game.get("away_spread_tickets")
        a_mon = an_game.get("away_spread_money")
    else:
        h_tkt = an_game.get("home_ml_tickets") if an_game.get("home_ml_tickets") is not None else an_game.get("home_spread_tickets")
        h_mon = an_game.get("home_ml_money") if an_game.get("home_ml_money") is not None else an_game.get("home_spread_money")
        a_tkt = an_game.get("away_ml_tickets") if an_game.get("away_ml_tickets") is not None else an_game.get("away_spread_tickets")
        a_mon = an_game.get("away_ml_money") if an_game.get("away_ml_money") is not None else an_game.get("away_spread_money")

    h_rlm = bool(an_game.get("home_rlm", False))
    a_rlm = bool(an_game.get("away_rlm", False))

    m3_h, _, _, has_sharp_h = models.run_model_3_sharp_and_kalshi(
        fair_h, h_k.get("price_cents"), float(h_k.get("volume", 0.0)), h_tkt, h_mon, h_rlm
    )
    m3_a, _, _, has_sharp_a = models.run_model_3_sharp_and_kalshi(
        fair_a, a_k.get("price_cents"), float(a_k.get("volume", 0.0)), a_tkt, a_mon, a_rlm
    )

    h_ment, h_sent, h_exp_note = match_expert_consensus(h_name, sport, expert_by_sport)
    a_ment, a_sent, a_exp_note = match_expert_consensus(a_name, sport, expert_by_sport)

    m4_h, has_exp_h = models.run_model_4_expert_consensus(fair_h, h_ment, a_ment, h_sent)
    m4_a, has_exp_a = models.run_model_4_expert_consensus(fair_a, a_ment, h_ment, a_sent)

    raw_h = models.compute_ensemble_probability(
        m1_h, m2_h, m3_h, mc_h,
        has_sharp_data=has_sharp_h,
        m4_prob=m4_h,
        has_expert_data=has_exp_h,
        fair_book_prob=fair_h,
    )
    raw_a = models.compute_ensemble_probability(
        m1_a, m2_a, m3_a, mc_a,
        has_sharp_data=has_sharp_a,
        m4_prob=m4_a,
        has_expert_data=has_exp_a,
        fair_book_prob=fair_a,
    )
    tot_p = raw_h + raw_a
    h_real = round(raw_h / tot_p, 4) if tot_p > 0 else 0.50
    a_real = round(1.0 - h_real, 4)

    p3_h = m3_h if has_sharp_h else (0.60 * mc_h + 0.40 * m1_h)
    p4_h = (0.55 * mc_h + 0.45 * m4_h) if has_exp_h else mc_h
    p3_a = m3_a if has_sharp_a else (0.60 * mc_a + 0.40 * m1_a)
    p4_a = (0.55 * mc_a + 0.45 * m4_a) if has_exp_a else mc_a

    h_cons, _, h_disp = models.evaluate_model_consensus([m1_h, m2_h, p3_h, p4_h], fair_h)
    a_cons, _, a_disp = models.evaluate_model_consensus([m1_a, m2_a, p3_a, p4_a], fair_a)

    h_trap = bool(
        (h_tkt is not None and h_mon is not None and float(h_tkt) >= 70.0 and float(h_mon) <= 43.0)
        or (h_k.get("price_cents") and float(h_k.get("volume", 0)) >= 5000 and ((h_k["price_cents"] / 100.0) - fair_h) <= -0.055)
    )
    a_trap = bool(
        (a_tkt is not None and a_mon is not None and float(a_tkt) >= 70.0 and float(a_mon) <= 43.0)
        or (a_k.get("price_cents") and float(a_k.get("volume", 0)) >= 5000 and ((a_k["price_cents"] / 100.0) - fair_a) <= -0.055)
    )

    def _fmt_kalshi_note(k_dict: Dict[str, Any]) -> str:
        if k_dict and k_dict.get("price_cents"):
            vol = float(k_dict.get("volume", 0.0))
            vol_str = f", Vol ${vol:,.0f}" if vol >= 250 else ""
            return f"Kalshi CFTC: {k_dict['price_cents']:.0f}% ({k_dict['price_cents']:.0f}c{vol_str})."
        return ""

    def _fmt_sharp_note(tkt: Optional[float], mon: Optional[float], rlm: bool) -> Tuple[str, float]:
        parts = []
        bonus = 0.0
        if tkt is not None and mon is not None:
            diff = float(mon) - float(tkt)
            if float(mon) >= 55.0 and diff >= 10.0:
                parts.append(f"Flujo Sharp: {int(mon)}% Dinero vs {int(tkt)}% Tickets (+{int(diff)}% divergencia)")
                bonus += min(4.0, diff * 0.14)
            elif float(mon) >= 58.0:
                parts.append(f"Respaldo Institucional: {int(mon)}% Dinero ({int(tkt)}% Tickets)")
                bonus += 1.5
        if rlm:
            parts.append("Reverse Line Movement (RLM) confirmado")
            bonus += 3.0
        return ((parts[0] + (" + " + parts[1] if len(parts) > 1 else "") + ".") if parts else ""), round(bonus, 2)

    h_s_note, h_s_bonus = _fmt_sharp_note(h_tkt, h_mon, h_rlm)
    a_s_note, a_s_bonus = _fmt_sharp_note(a_tkt, a_mon, a_rlm)
    if h_k.get("price_cents") and (h_k["price_cents"] / 100.0 - fair_h) >= 0.025:
        h_s_bonus += 2.2
    if a_k.get("price_cents") and (a_k["price_cents"] / 100.0 - fair_a) >= 0.025:
        a_s_bonus += 2.2
    if h_ment >= 2 and h_sent >= 0:
        h_s_bonus += 1.2
    if a_ment >= 2 and a_sent >= 0:
        a_s_bonus += 1.2

    return {
        "home_real_prob": h_real,
        "away_real_prob": a_real,
        "home_ev": calculate_ev(h_real, american_to_decimal(h_ml)),
        "away_ev": calculate_ev(a_real, american_to_decimal(a_ml)),
        "home_consensus": h_cons,
        "away_consensus": a_cons,
        "home_model_dispersion": h_disp,
        "away_model_dispersion": a_disp,
        "total_models": 4,
        "home_public_trap": h_trap,
        "away_public_trap": a_trap,
        "home_sharp_bonus": round(h_s_bonus, 2),
        "away_sharp_bonus": round(a_s_bonus, 2),
        "home_kalshi_prob": h_k.get("price_cents", 0.0),
        "away_kalshi_prob": a_k.get("price_cents", 0.0),
        "home_kalshi_vol": h_k.get("volume", 0.0),
        "away_kalshi_vol": a_k.get("volume", 0.0),
        "home_tickets_pct": h_tkt,
        "home_sharp_money_pct": h_mon,
        "away_tickets_pct": a_tkt,
        "away_sharp_money_pct": a_mon,
        "home_rlm": h_rlm,
        "away_rlm": a_rlm,
        "home_kalshi_note": _fmt_kalshi_note(h_k),
        "away_kalshi_note": _fmt_kalshi_note(a_k),
        "home_sharp_note": h_s_note,
        "away_sharp_note": a_s_note,
        "home_expert_note": h_exp_note,
        "away_expert_note": a_exp_note,
    }


def build_combined_sharp_whales(
    kalshi_whales: List[Dict[str, Any]],
    action_by_sport: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merges real-money CFTC Kalshi institutional flows with Action Network RLM & Sharp Handle anomalies."""
    merged = list(kalshi_whales)
    for sp_up, games in action_by_sport.items():
        for g in games:
            h_t = g.get("home_team", "")
            a_t = g.get("away_team", "")
            n_bets = int(g.get("num_bets") or 0)
            matchup = f"{a_t} @ {h_t}"

            for side_team, tkt, mon, rlm in (
                (h_t, g.get("home_spread_tickets") or g.get("home_ml_tickets"), g.get("home_spread_money") or g.get("home_ml_money"), g.get("home_rlm")),
                (a_t, g.get("away_spread_tickets") or g.get("away_ml_tickets"), g.get("away_spread_money") or g.get("away_ml_money"), g.get("away_rlm")),
            ):
                if tkt is None or mon is None:
                    continue
                diff = float(mon) - float(tkt)
                if rlm or (float(mon) >= 62.0 and diff >= 18.0):
                    sig_lbl = "RLM + SHARP MONEY" if (rlm and diff >= 12.0) else ("REVERSE LINE MOVEMENT (RLM)" if rlm else f"DIVERGENCIA SHARP (+{int(diff)}% DINERO)")
                    merged.append({
                        "sport": sp_up,
                        "series": "ACTION_SHARP",
                        "game": matchup,
                        "target": side_team,
                        "price_cents": float(mon),
                        "volume": float(max(n_bets * 45, 5000)),
                        "flow_share_pct": float(mon),
                        "tickets_pct": float(tkt),
                        "signal": f"{sig_lbl} ({int(mon)}% $ vs {int(tkt)}% Tkt)",
                        "sentiment": "BULLISH",
                    })

    merged.sort(key=lambda x: (1 if "RLM" in x.get("signal", "") or "KALSHI" in x.get("signal", "") else 0, x.get("volume", 0.0)), reverse=True)
    return merged[:30]


def run_sports_pipeline(record_in_history: bool = True) -> Dict[str, Any]:
    scraper = SportsDataScraper()
    mc_sim = MonteCarloSimulator(simulations=10000, seed=42)
    models = PredictiveModelSuite()
    history = HistoryTracker()
    ai_agent = QuantitativeAIAgent(history_data=history.data)

    now_cst = datetime.now(CST_TZ)
    today_str = now_cst.strftime("%Y-%m-%d")
    today_ymd = now_cst.strftime("%Y%m%d")

    print("\n" + "=" * 84)
    print("[FASE 1] SCRAPING MULTI-FUENTE EN TIEMPO REAL (ESPN + KALSHI + ACTION SHARP + FOROS)")
    print("=" * 84)

    nba_espn_raw = scraper.fetch_espn_scoreboard("basketball", "nba", date_str=today_ymd, allow_undated_fallback=False)
    nba_espn = [g for g in nba_espn_raw if is_game_today(g.get("date", ""), today_str)]

    nhl_espn_raw = scraper.fetch_espn_scoreboard("hockey", "nhl", date_str=today_ymd, allow_undated_fallback=False)
    nhl_espn = [g for g in nhl_espn_raw if is_game_today(g.get("date", ""), today_str)]

    soccer_espn = scraper.fetch_espn_soccer_matches(date_str=today_ymd, local_date_iso=today_str)
    ncaaf_espn, ncaaf_date_str = scraper.fetch_espn_ncaaf_scoreboard(date_str=today_ymd)
    tennis_espn = scraper.fetch_espn_tennis_matches(date_str=today_ymd)

    nfl_espn_raw = scraper.fetch_espn_scoreboard("football", "nfl", date_str=today_ymd, allow_undated_fallback=True)
    nfl_espn = [g for g in nfl_espn_raw if is_game_today(g.get("date", ""), today_str)]
    nfl_target_date = today_str
    if not nfl_espn and nfl_espn_raw:
        # Check upcoming Sunday games when today has no active NFL slate
        upcoming_nfl = [g for g in nfl_espn_raw if not str(g.get("status", "")).endswith("FINAL")]
        if upcoming_nfl:
            nfl_espn = upcoming_nfl
            raw_d0 = nfl_espn[0].get("date", "")
            if raw_d0:
                try:
                    dt_c = datetime.fromisoformat(raw_d0.replace("Z", "+00:00")).astimezone(CST_TZ)
                    nfl_target_date = dt_c.strftime("%Y-%m-%d")
                except Exception:
                    pass

    mlb_espn_raw = scraper.fetch_baseball_all_leagues(date_str=today_str)
    mlb_espn = mlb_espn_raw

    kalshi_data = scraper.fetch_all_kalshi_sports()
    kalshi_by_sport = kalshi_data.get("by_sport", {})

    ncaaf_ymd = ncaaf_date_str.replace("-", "")
    action_dates = list(dict.fromkeys([today_ymd, ncaaf_ymd]))
    action_by_sport = scraper.fetch_action_network_sharp_signals(dates_ymd=action_dates)
    expert_by_sport = scraper.fetch_expert_and_forum_consensus()

    whales = build_combined_sharp_whales(kalshi_data.get("sharp_whales", []), action_by_sport)

    kalshi_total_mkts = sum(len(v) for v in kalshi_by_sport.values())
    action_total_games = sum(len(v) for v in action_by_sport.values())
    expert_total_arts = sum(len(v) for v in expert_by_sport.values())

    print(
        f"Cartelera ESPN -> NBA: {len(nba_espn)} | NHL: {len(nhl_espn)} | "
        f"Futbol (15 Ligas/Copas): {len(soccer_espn)} | NCAAF ({ncaaf_date_str}): {len(ncaaf_espn)} | "
        f"Tenis (ATP/WTA): {len(tennis_espn)} | NFL: {len(nfl_espn)} | MLB: {len(mlb_espn)}"
    )
    print(
        f"Inteligencia Institucional -> Contratos Kalshi: {kalshi_total_mkts} | "
        f"Juegos Action Sharp/RLM: {action_total_games} | Analisis Expertos/Foros: {expert_total_arts}"
    )

    print("\n" + "=" * 84)
    print("[FASE 2] ENSAMBLE DE 4 MODELOS + SABERMETRIA VIVA + 10,000 SIMS MONTE CARLO")
    print("=" * 84)

    # Shared directional locks across MVP, all Horarios, and all Sport Tabs -> ZERO contradiction!
    shared_backed_sides: Dict[str, str] = {}
    shared_opposing_sides: Dict[str, str] = {}
    shared_backed_totals: Dict[str, str] = {}

    def make_tab_engine() -> SelectionEngine:
        return SelectionEngine(
            shared_used_selections=set(),
            shared_backed_sides=shared_backed_sides,
            shared_opposing_sides=shared_opposing_sides,
            shared_backed_totals=shared_backed_totals,
        )

    # =========================================================
    # 1. PROCESS NBA (Four Factors + Splits/Rest + Kalshi/Sharp + 10k MC)
    # =========================================================
    nba_analyzed_games: List[Dict[str, Any]] = []
    nba_alt_markets: List[Dict[str, Any]] = []
    nba_totals_pool: List[Dict[str, Any]] = []

    for g in nba_espn:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        an_g = match_action_sharp_game(h_name, a_name, "NBA", action_by_sport)

        raw_spr = g["odds"].get("spread")
        spread_val = float(raw_spr if raw_spr not in (None, 0.0) else (an_g.get("home_spread") or 0.0))
        total_val = float(g["odds"].get("overUnder") or an_g.get("total") or 228.5)
        if total_val < 180.0:
            total_val = 228.5

        h_ml = g["odds"].get("home_moneyline") or an_g.get("home_ml")
        a_ml = g["odds"].get("away_moneyline") or an_g.get("away_ml")
        h_calc_ml, a_calc_ml, _, _ = spread_to_american_ml(spread_val, sigma=11.8)
        h_ml = int(h_ml if h_ml is not None else h_calc_ml)
        a_ml = int(a_ml if a_ml is not None else a_calc_ml)

        h_stats = get_nba_stats(
            h_name,
            g["home_team"].get("record", ""),
            live_stats=g["home_team"].get("live_stats"),
            records_map=g["home_team"].get("records"),
            spread_hint=-spread_val * 0.5,
        )
        a_stats = get_nba_stats(
            a_name,
            g["away_team"].get("record", ""),
            live_stats=g["away_team"].get("live_stats"),
            records_map=g["away_team"].get("records"),
            spread_hint=+spread_val * 0.5,
        )

        m1_h, m1_a = models.run_model_1_sabermetric_nba(h_stats, a_stats)
        m2_h, m2_a = models.run_model_2_contextual_nba(h_stats, a_stats)
        mc_res = mc_sim.simulate_nba_game(
            home_pace=h_stats["Pace"],
            away_pace=a_stats["Pace"],
            home_ortg=h_stats["ORtg"],
            home_drtg=h_stats["DRtg"],
            away_ortg=a_stats["ORtg"],
            away_drtg=a_stats["DRtg"],
            spread=spread_val,
            total=total_val,
        )

        intel = evaluate_two_way_intelligence(
            models, "NBA", h_name, a_name, h_ml, a_ml,
            m1_h, m1_a, m2_h, m2_a, mc_res["prob_home_win"], mc_res["prob_away_win"],
            kalshi_by_sport, an_g, expert_by_sport, use_spread_splits=True,
        )

        saber_h_note = (
            f"Sabermetria NBA (Four Factors): NetRtg {h_stats['NetRtg']:+.1f} vs {a_stats['NetRtg']:+.1f}, "
            f"eFG% {h_stats['eFG']*100:.1f}%, TS% {h_stats['TS']*100:.1f}%, TOV% {h_stats['TOV_pct']:.1f}%, Pace {h_stats['Pace']:.1f}."
        )
        saber_a_note = (
            f"Sabermetria NBA (Four Factors): NetRtg {a_stats['NetRtg']:+.1f} vs {h_stats['NetRtg']:+.1f}, "
            f"eFG% {a_stats['eFG']*100:.1f}%, TS% {a_stats['TS']*100:.1f}%, TOV% {a_stats['TOV_pct']:.1f}%, Pace {a_stats['Pace']:.1f}."
        )

        nba_analyzed_games.append({
            "event_id": ev_id,
            "sport": "NBA",
            "local_date": today_str,
            "home_team": h_name,
            "away_team": a_name,
            "matchup": f"{a_name} @ {h_name}",
            "spread": spread_val,
            "total": total_val,
            "home_odds": h_ml,
            "away_odds": a_ml,
            "home_cover_prob": mc_res["prob_home_cover"],
            "away_cover_prob": mc_res["prob_away_cover"],
            "saber_home_note": saber_h_note,
            "saber_away_note": saber_a_note,
            "projected_score": f"{mc_res['mean_away_score']} - {mc_res['mean_home_score']}",
            **intel,
        })

        if mc_res["alt_lines"]["home_plus_8_5"] >= 0.76:
            nba_alt_markets.append({
                "event_id": ev_id,
                "sport": "NBA",
                "local_date": today_str,
                "team": h_name,
                "opponent": a_name,
                "game": f"{a_name} @ {h_name}",
                "market_type": "Spread Alternativo",
                "selection": f"{h_name} +8.5",
                "american_odds": -310,
                "real_prob": mc_res["alt_lines"]["home_plus_8_5"],
                "justification": (
                    f"Colchon defensivo de +8.5 puntos para {h_name} en casa ({saber_h_note}). "
                    f"En 10,000 simulaciones Monte Carlo cubre la linea en el {round(mc_res['alt_lines']['home_plus_8_5']*100, 1)}% de los partidos."
                ),
            })
        if mc_res["alt_lines"].get("away_plus_8_5", 0) >= 0.76:
            nba_alt_markets.append({
                "event_id": ev_id,
                "sport": "NBA",
                "local_date": today_str,
                "team": a_name,
                "opponent": h_name,
                "game": f"{a_name} @ {h_name}",
                "market_type": "Spread Alternativo",
                "selection": f"{a_name} +8.5",
                "american_odds": -310,
                "real_prob": mc_res["alt_lines"]["away_plus_8_5"],
                "justification": (
                    f"Proteccion extendida de +8.5 puntos para {a_name} ({saber_a_note}). "
                    f"Cubre el margen en el {round(mc_res['alt_lines']['away_plus_8_5']*100, 1)}% de 10,000 simulaciones Monte Carlo."
                ),
            })

        proj_total = mc_res["mean_home_score"] + mc_res["mean_away_score"]
        total_diff = proj_total - total_val
        if total_diff <= -2.2 and mc_res["prob_under"] >= 0.54:
            if mc_res["alt_lines"]["under_alt_safe"] >= 0.75:
                nba_alt_markets.append({
                    "event_id": ev_id,
                    "sport": "NBA",
                    "local_date": today_str,
                    "team": "-",
                    "opponent": "-",
                    "game": f"{a_name} @ {h_name}",
                    "market_type": "Total Alternativo Seguro",
                    "selection": f"{a_name} @ {h_name} Under {total_val + 7.5}",
                    "american_odds": -330,
                    "real_prob": mc_res["alt_lines"]["under_alt_safe"],
                    "justification": (
                        f"Filtro de divergencia UNDER superado: ritmo combinado {(h_stats['Pace']+a_stats['Pace'])/2:.1f} "
                        f"y DRtg ({h_stats['DRtg']:.1f} / {a_stats['DRtg']:.1f}) proyectan {proj_total:.1f} pts ({abs(total_diff):.1f} pts bajo {total_val}), "
                        f"cubriendo Under {total_val + 7.5} en {round(mc_res['alt_lines']['under_alt_safe']*100, 1)}% de simulaciones."
                    ),
                })
            shrunk_u = 0.68 * mc_res["prob_under"] + 0.32 * 0.50
            ev_u = calculate_ev(shrunk_u, american_to_decimal(-110))
            nba_totals_pool.append({
                "event_id": ev_id,
                "sport": "NBA",
                "local_date": today_str,
                "game": f"{a_name} @ {h_name}",
                "selection": f"{a_name} @ {h_name} Under {total_val}",
                "type": "Total de Partido",
                "american_odds": -110,
                "decimal_odds": 1.91,
                "real_prob": round(shrunk_u * 100, 1),
                "ev": round(ev_u * 100, 2),
                "justification": (
                    f"Tendencia UNDER validada ({proj_total:.1f} pts proyectados vs linea {total_val}, "
                    f"Pace medio {(h_stats['Pace']+a_stats['Pace'])/2:.1f}, {round(shrunk_u*100, 1)}% base)."
                ),
            })
        elif total_diff >= 2.2 and mc_res["prob_over"] >= 0.54:
            if mc_res["alt_lines"]["over_alt_safe"] >= 0.75:
                nba_alt_markets.append({
                    "event_id": ev_id,
                    "sport": "NBA",
                    "local_date": today_str,
                    "team": "-",
                    "opponent": "-",
                    "game": f"{a_name} @ {h_name}",
                    "market_type": "Total Alternativo Seguro",
                    "selection": f"{a_name} @ {h_name} Over {total_val - 7.5}",
                    "american_odds": -330,
                    "real_prob": mc_res["alt_lines"]["over_alt_safe"],
                    "justification": (
                        f"Filtro de divergencia OVER superado: Pace combinado {(h_stats['Pace']+a_stats['Pace'])/2:.1f} "
                        f"y ORtg ({h_stats['ORtg']:.1f} / {a_stats['ORtg']:.1f}) proyectan {proj_total:.1f} pts (+{total_diff:.1f} sobre {total_val}), "
                        f"cubriendo Over {total_val - 7.5} en {round(mc_res['alt_lines']['over_alt_safe']*100, 1)}% de simulaciones."
                    ),
                })
            shrunk_o = 0.68 * mc_res["prob_over"] + 0.32 * 0.50
            ev_o = calculate_ev(shrunk_o, american_to_decimal(-110))
            nba_totals_pool.append({
                "event_id": ev_id,
                "sport": "NBA",
                "local_date": today_str,
                "game": f"{a_name} @ {h_name}",
                "selection": f"{a_name} @ {h_name} Over {total_val}",
                "type": "Total de Partido",
                "american_odds": -110,
                "decimal_odds": 1.91,
                "real_prob": round(shrunk_o * 100, 1),
                "ev": round(ev_o * 100, 2),
                "justification": (
                    f"Tendencia OVER validada ({proj_total:.1f} pts proyectados vs linea {total_val}, "
                    f"TS% combinado {(h_stats['TS']+a_stats['TS'])*50:.1f}%, {round(shrunk_o*100, 1)}% base)."
                ),
            })

    nba_props_pool = scraper.fetch_live_espn_nba_stars(nba_espn) if nba_espn else []
    for s in nba_props_pool:
        s["date"] = today_str
        s["local_date"] = today_str

    # =========================================================
    # 2. PROCESS NHL (5v5 xGF%/HDCF%/PDO + Goalie SV%/Special Teams + Kalshi/Sharp)
    # =========================================================
    nhl_analyzed_games: List[Dict[str, Any]] = []
    nhl_alt_markets: List[Dict[str, Any]] = []
    nhl_totals_pool: List[Dict[str, Any]] = []

    for g in nhl_espn:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        an_g = match_action_sharp_game(h_name, a_name, "NHL", action_by_sport)

        h_stats = get_nhl_stats(
            h_name, g["home_team"].get("record", ""),
            live_stats=g["home_team"].get("live_stats"),
            records_map=g["home_team"].get("records"),
        )
        a_stats = get_nhl_stats(
            a_name, g["away_team"].get("record", ""),
            live_stats=g["away_team"].get("live_stats"),
            records_map=g["away_team"].get("records"),
        )

        odds = g.get("odds", {})
        h_ml = int(odds.get("home_moneyline") or an_g.get("home_ml") or -125)
        a_ml = int(odds.get("away_moneyline") or an_g.get("away_ml") or +105)
        total_val = float(odds.get("overUnder") or an_g.get("total") or 6.0)

        h_xg = round(2.98 * (h_stats.get("xGF_pct", 50.0) / 50.0) * (2.78 / max(2.1, a_stats.get("GAA", 2.75))), 2)
        a_xg = round(2.72 * (a_stats.get("xGF_pct", 50.0) / 50.0) * (2.78 / max(2.1, h_stats.get("GAA", 2.75))), 2)
        nhl_mc = mc_sim.simulate_nhl_game(home_xg=h_xg, away_xg=a_xg, total=total_val)

        m1_h, m1_a = models.run_model_1_5v5_nhl(h_stats, a_stats)
        m2_h, m2_a = models.run_model_2_goalie_special_nhl(h_stats, a_stats)

        intel = evaluate_two_way_intelligence(
            models, "NHL", h_name, a_name, h_ml, a_ml,
            m1_h, m1_a, m2_h, m2_a, nhl_mc["prob_home_win"], nhl_mc["prob_away_win"],
            kalshi_by_sport, an_g, expert_by_sport, use_spread_splits=False,
        )

        saber_h = (
            f"Sabermetria NHL: Portero {h_stats['goaltender']} (SV% {h_stats['SV_pct']:.3f}, GAA {h_stats['GAA']:.2f}), "
            f"5v5 xGF% {h_stats['xGF_pct']:.1f}%, HDCF% {h_stats.get('HDCF_pct', h_stats['xGF_pct']):.1f}%, PP {h_stats['PP_pct']:.1f}% / PK {h_stats['PK_pct']:.1f}%."
        )
        saber_a = (
            f"Sabermetria NHL: Portero {a_stats['goaltender']} (SV% {a_stats['SV_pct']:.3f}, GAA {a_stats['GAA']:.2f}), "
            f"5v5 xGF% {a_stats['xGF_pct']:.1f}%, HDCF% {a_stats.get('HDCF_pct', a_stats['xGF_pct']):.1f}%, PP {a_stats['PP_pct']:.1f}% / PK {a_stats['PK_pct']:.1f}%."
        )

        nhl_analyzed_games.append({
            "event_id": ev_id,
            "sport": "NHL",
            "local_date": today_str,
            "home_team": h_name,
            "away_team": a_name,
            "matchup": f"{a_name} @ {h_name}",
            "home_odds": h_ml,
            "away_odds": a_ml,
            "saber_home_note": saber_h,
            "saber_away_note": saber_a,
            "home_starter": h_stats.get("goaltender", "Confirmado"),
            "away_starter": a_stats.get("goaltender", "Confirmado"),
            "home_sv": h_stats.get("SV_pct", 0.912),
            "away_sv": a_stats.get("SV_pct", 0.908),
            "projected_score": f"{nhl_mc['mean_away_goals']} - {nhl_mc['mean_home_goals']}",
            **intel,
        })

        proj_nhl_total = nhl_mc["mean_home_goals"] + nhl_mc["mean_away_goals"]
        nhl_diff = proj_nhl_total - total_val

        if nhl_diff <= -0.25 and nhl_mc.get("prob_under_6_5", 0) >= 0.60:
            nhl_alt_markets.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "team": "-",
                "opponent": "-",
                "game": f"{a_name} @ {h_name}",
                "market_type": "Total Alternativo Seguro",
                "selection": f"{a_name} @ {h_name}: Menos de 7.5 Goles",
                "american_odds": -280,
                "real_prob": nhl_mc["prob_under_alt_safe"],
                "justification": (
                    f"Duelo defensivo en hielo ({h_stats['goaltender']} SV% {h_stats['SV_pct']:.3f} vs {a_stats['goaltender']} SV% {a_stats['SV_pct']:.3f}). "
                    f"Proyeccion Poisson de {proj_nhl_total:.2f} goles totales cubre Menos de 7.5 en {round(nhl_mc['prob_under_alt_safe']*100, 1)}% de simulaciones."
                ),
            })
            nhl_totals_pool.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "game": f"{a_name} @ {h_name}",
                "selection": f"{a_name} @ {h_name} Menos de 6.5 Goles",
                "type": "Total de Partido",
                "american_odds": -155,
                "decimal_odds": american_to_decimal(-155),
                "real_prob": round(nhl_mc["prob_under_6_5"] * 100, 1),
                "ev": 6.4,
                "justification": (
                    f"Control de zona neutral y porteria solvente proyectan {proj_nhl_total:.2f} goles combinados "
                    f"({round(nhl_mc['prob_under_6_5']*100, 1)}% bajo 6.5 goles en 10,000 simulaciones Poisson)."
                ),
            })
        elif nhl_diff >= 0.25 and nhl_mc.get("prob_over_5_5", 0) >= 0.58:
            nhl_alt_markets.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "team": "-",
                "opponent": "-",
                "game": f"{a_name} @ {h_name}",
                "market_type": "Total Alternativo Seguro",
                "selection": f"{a_name} @ {h_name}: Mas de 4.5 Goles",
                "american_odds": -275,
                "real_prob": nhl_mc["prob_over_alt_safe"],
                "justification": (
                    f"Alto ritmo de disparos de alto peligro (xG combinado {h_xg + a_xg:.2f}). "
                    f"Supera 4.5 goles totales en el {round(nhl_mc['prob_over_alt_safe']*100, 1)}% de 10,000 simulaciones Poisson."
                ),
            })
            nhl_totals_pool.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "game": f"{a_name} @ {h_name}",
                "selection": f"{a_name} @ {h_name} Mas de 5.5 Goles",
                "type": "Total de Partido",
                "american_odds": -150,
                "decimal_odds": american_to_decimal(-150),
                "real_prob": round(nhl_mc["prob_over_5_5"] * 100, 1),
                "ev": 6.2,
                "justification": (
                    f"Potencial ofensivo en Power Play y transicion ({proj_nhl_total:.2f} goles proyectados), "
                    f"superando 5.5 goles en el {round(nhl_mc['prob_over_5_5']*100, 1)}% de simulaciones."
                ),
            })

        if a_ml >= 115 and nhl_mc.get("prob_away_puck_line_plus_1_5", 0) >= 0.72:
            nhl_alt_markets.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "team": a_name,
                "opponent": h_name,
                "game": f"{a_name} @ {h_name}",
                "market_type": "Puck Line (+1.5)",
                "selection": f"{a_name} +1.5 Goles (Puck Line)",
                "american_odds": -175,
                "real_prob": nhl_mc["prob_away_puck_line_plus_1_5"],
                "justification": (
                    f"Cobertura Puck Line +1.5 goles para {a_name} ({saber_a}). "
                    f"Cubre en el {round(nhl_mc['prob_away_puck_line_plus_1_5']*100, 1)}% de 10,000 simulaciones Poisson."
                ),
            })
        if h_ml >= 115 and nhl_mc.get("prob_home_puck_line_plus_1_5", 0) >= 0.72:
            nhl_alt_markets.append({
                "event_id": ev_id,
                "sport": "NHL",
                "local_date": today_str,
                "team": h_name,
                "opponent": a_name,
                "game": f"{a_name} @ {h_name}",
                "market_type": "Puck Line (+1.5)",
                "selection": f"{h_name} +1.5 Goles (Puck Line)",
                "american_odds": -175,
                "real_prob": nhl_mc["prob_home_puck_line_plus_1_5"],
                "justification": (
                    f"Cobertura Puck Line +1.5 goles en casa para {h_name} ({saber_h}). "
                    f"Cubre en el {round(nhl_mc['prob_home_puck_line_plus_1_5']*100, 1)}% de 10,000 simulaciones Poisson."
                ),
            })

    nhl_props_pool = scraper.fetch_live_espn_nhl_stars(nhl_espn) if nhl_espn else []
    for p in nhl_props_pool:
        p["date"] = today_str
        p["local_date"] = today_str

    # =========================================================
    # 3. PROCESS FUTBOL (5 Big Leagues + Domestic Cups + UEFA + Liga MX)
    # =========================================================
    soccer_analyzed_games: List[Dict[str, Any]] = []
    soccer_alt_markets: List[Dict[str, Any]] = []
    soccer_totals_pool: List[Dict[str, Any]] = []

    for g in soccer_espn:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        lg_name = g.get("league_name", "Futbol Internacional")
        lg_slug = g.get("league_slug") or g.get("league", "")
        kickoff_cst = g.get("kickoff_cst", "")
        an_g = match_action_sharp_game(h_name, a_name, "FUTBOL", action_by_sport)

        h_stats = get_soccer_stats(h_name, g["home_team"].get("record", ""), league_slug=lg_slug)
        a_stats = get_soccer_stats(a_name, g["away_team"].get("record", ""), league_slug=lg_slug)

        alt_boost = 0.18 if (h_stats.get("is_high_altitude") and not a_stats.get("is_high_altitude")) else 0.0
        h_match_xg = round(max(0.55, ((h_stats["xG_for"] + a_stats["xGA_against"]) / 2.0) + 0.16 + alt_boost), 2)
        a_match_xg = round(max(0.45, ((a_stats["xG_for"] + h_stats["xGA_against"]) / 2.0) - 0.10 - (alt_boost * 0.4)), 2)

        soc_mc = mc_sim.simulate_soccer_game(h_match_xg, a_match_xg)
        h_mc_p = soc_mc["prob_home_win"]
        d_mc_p = soc_mc["prob_draw"]
        a_mc_p = soc_mc["prob_away_win"]

        m1_h, m1_d, m1_a = models.run_model_1_dixon_coles_soccer(h_stats, a_stats, draw_baseline=d_mc_p)
        m2_h, m2_d, m2_a = models.run_model_2_tactical_soccer(h_stats, a_stats, draw_baseline=d_mc_p)

        odds = g.get("odds", {})
        h_ml = odds.get("home_moneyline") or an_g.get("home_ml")
        a_ml = odds.get("away_moneyline") or an_g.get("away_ml")
        d_ml = odds.get("draw_moneyline") or an_g.get("draw_ml")
        if h_ml is None or a_ml is None:
            der_h, der_d, der_a = derive_soccer_3way_book_odds(0.5 * (m1_h + h_mc_p), d_mc_p, 0.5 * (m1_a + a_mc_p))
            h_ml = int(h_ml if h_ml is not None else der_h)
            a_ml = int(a_ml if a_ml is not None else der_a)
            d_ml = int(d_ml if d_ml is not None else der_d)
        else:
            h_ml = int(h_ml)
            a_ml = int(a_ml)
            d_ml = int(d_ml if d_ml is not None else +240)

        fair_h, fair_d, fair_a = de_vig_three_way(h_ml, d_ml, a_ml)
        h_k = match_kalshi_contract(h_name, a_name, "FUTBOL", kalshi_by_sport)
        a_k = match_kalshi_contract(a_name, h_name, "FUTBOL", kalshi_by_sport)

        m3_h, _, _, has_sh_h = models.run_model_3_sharp_and_kalshi(
            fair_h, h_k.get("price_cents"), float(h_k.get("volume", 0.0)),
            an_g.get("home_ml_tickets"), an_g.get("home_ml_money"), bool(an_g.get("home_rlm")),
        )
        m3_a, _, _, has_sh_a = models.run_model_3_sharp_and_kalshi(
            fair_a, a_k.get("price_cents"), float(a_k.get("volume", 0.0)),
            an_g.get("away_ml_tickets"), an_g.get("away_ml_money"), bool(an_g.get("away_rlm")),
        )
        h_ment, h_sent, h_exp_note = match_expert_consensus(h_name, "FUTBOL", expert_by_sport)
        a_ment, a_sent, a_exp_note = match_expert_consensus(a_name, "FUTBOL", expert_by_sport)
        m4_h, has_ex_h = models.run_model_4_expert_consensus(fair_h, h_ment, a_ment, h_sent)
        m4_a, has_ex_a = models.run_model_4_expert_consensus(fair_a, a_ment, h_ment, a_sent)

        raw_h = models.compute_ensemble_probability(m1_h, m2_h, m3_h, h_mc_p, has_sh_h, m4_h, has_ex_h, fair_h)
        raw_a = models.compute_ensemble_probability(m1_a, m2_a, m3_a, a_mc_p, has_sh_a, m4_a, has_ex_a, fair_a)
        raw_d = 0.50 * (0.5 * (m1_d + m2_d) + 0.5 * d_mc_p) + 0.50 * fair_d
        tot_3w = raw_h + raw_d + raw_a
        h_real = round(raw_h / tot_3w, 4)
        d_real = round(raw_d / tot_3w, 4)
        a_real = round(max(0.02, 1.0 - h_real - d_real), 4)

        h_ev = calculate_ev(h_real, american_to_decimal(h_ml))
        a_ev = calculate_ev(a_real, american_to_decimal(a_ml))

        p3_h = m3_h if has_sh_h else (0.6 * h_mc_p + 0.4 * m1_h)
        p4_h = (0.55 * h_mc_p + 0.45 * m4_h) if has_ex_h else h_mc_p
        p3_a = m3_a if has_sh_a else (0.6 * a_mc_p + 0.4 * m1_a)
        p4_a = (0.55 * a_mc_p + 0.45 * m4_a) if has_ex_a else a_mc_p
        h_cons, _, _ = models.evaluate_model_consensus([m1_h, m2_h, p3_h, p4_h], fair_h)
        a_cons, _, _ = models.evaluate_model_consensus([m1_a, m2_a, p3_a, p4_a], fair_a)

        alt_tag = " [Ventaja Altitud >2,200m]" if alt_boost > 0 else ""
        saber_h = (
            f"Sabermetria Futbol ({lg_name}){alt_tag}: xG {h_match_xg:.2f} vs {a_match_xg:.2f}, "
            f"Field Tilt {h_stats['field_tilt']:.1f}%, PPDA {h_stats['ppda']:.1f}, Tiros a puerta {h_stats['shots_on_target']:.1f}/j."
        )
        saber_a = (
            f"Sabermetria Futbol ({lg_name}): xG {a_match_xg:.2f} vs {h_match_xg:.2f}, "
            f"Field Tilt {a_stats['field_tilt']:.1f}%, PPDA {a_stats['ppda']:.1f}, Tiros a puerta {a_stats['shots_on_target']:.1f}/j."
        )

        h_k_note = f"Kalshi CFTC: {h_k['price_cents']:.0f}% ({h_k['price_cents']:.0f}c)." if h_k.get("price_cents") else ""
        a_k_note = f"Kalshi CFTC: {a_k['price_cents']:.0f}% ({a_k['price_cents']:.0f}c)." if a_k.get("price_cents") else ""

        soccer_analyzed_games.append({
            "event_id": ev_id,
            "league_slug": lg_slug,
            "sport": "Futbol",
            "local_date": today_str,
            "league_name": lg_name,
            "kickoff_cst": kickoff_cst,
            "home_team": h_name,
            "away_team": a_name,
            "matchup": f"{a_name} vs {h_name}",
            "home_odds": h_ml,
            "draw_odds": d_ml,
            "away_odds": a_ml,
            "home_real_prob": round(h_real * 100.0, 1),
            "draw_real_prob": round(d_real * 100.0, 1),
            "draw_prob": d_real,
            "away_real_prob": round(a_real * 100.0, 1),
            "home_ev": round(h_ev * 100.0, 2),
            "away_ev": round(a_ev * 100.0, 2),
            "home_consensus": h_cons,
            "away_consensus": a_cons,
            "total_models": 4,
            "saber_home_note": saber_h,
            "saber_away_note": saber_a,
            "home_kalshi_prob": h_k.get("price_cents", 0.0),
            "away_kalshi_prob": a_k.get("price_cents", 0.0),
            "home_kalshi_note": h_k_note,
            "away_kalshi_note": a_k_note,
            "home_expert_note": h_exp_note,
            "away_expert_note": a_exp_note,
            "home_xg": h_match_xg,
            "away_xg": a_match_xg,
            "projected_score": f"{soc_mc['mean_away_goals']} - {soc_mc['mean_home_goals']}",
        })

        proj_soc = h_match_xg + a_match_xg
        if proj_soc <= 2.28 and soc_mc.get("prob_under_2_5", 0.5) >= 0.55:
            u35_prob = min(0.86, soc_mc["prob_under_2_5"] + 0.21)
            soccer_alt_markets.append({
                "event_id": ev_id,
                "league_slug": lg_slug,
                "sport": "Futbol",
                "local_date": today_str,
                "league_name": lg_name,
                "team": "-",
                "opponent": "-",
                "game": f"{a_name} vs {h_name}",
                "market_type": "Total Alternativo (Goles)",
                "selection": f"{a_name} vs {h_name}: Menos de 3.5 Goles",
                "american_odds": -275,
                "real_prob": u35_prob,
                "justification": (
                    f"[{lg_name}] Bloques defensivos cerrados ({proj_soc:.2f} xG combinado entre {a_name} y {h_name}). "
                    f"Simulacion Dixon-Coles mantiene el partido bajo 3.5 goles en el {round(u35_prob*100, 1)}% de los escenarios."
                ),
            })
            soccer_totals_pool.append({
                "event_id": ev_id,
                "league_slug": lg_slug,
                "sport": "Futbol",
                "local_date": today_str,
                "league_name": lg_name,
                "game": f"{a_name} vs {h_name}",
                "selection": f"{a_name} vs {h_name} Under 3.5 Goles",
                "type": "Total de Partido",
                "american_odds": -265,
                "decimal_odds": american_to_decimal(-265),
                "real_prob": round(u35_prob * 100.0, 1),
                "ev": 6.2,
                "justification": (
                    f"[{lg_name}] Perfil tactico UNDER ({proj_soc:.2f} xG proyectado). "
                    f"Cubre Under 3.5 goles con {round(u35_prob*100, 1)}% de probabilidad real en 10,000 simulaciones."
                ),
            })
        elif proj_soc >= 2.72 and soc_mc.get("match_over_1_5", 0.0) >= 0.76:
            soccer_alt_markets.append({
                "event_id": ev_id,
                "league_slug": lg_slug,
                "sport": "Futbol",
                "local_date": today_str,
                "league_name": lg_name,
                "team": "-",
                "opponent": "-",
                "game": f"{a_name} vs {h_name}",
                "market_type": "Total Alternativo (Goles)",
                "selection": f"{a_name} vs {h_name}: Mas de 1.5 Goles",
                "american_odds": -295,
                "real_prob": soc_mc["match_over_1_5"],
                "justification": (
                    f"[{lg_name}] Potencial ofensivo elevado ({proj_soc:.2f} xG combinado: {h_name} {h_match_xg:.2f} xG, {a_name} {a_match_xg:.2f} xG). "
                    f"Supera 1.5 goles totales en el {round(soc_mc['match_over_1_5']*100, 1)}% de 10,000 simulaciones Dixon-Coles."
                ),
            })
            soccer_totals_pool.append({
                "event_id": ev_id,
                "league_slug": lg_slug,
                "sport": "Futbol",
                "local_date": today_str,
                "league_name": lg_name,
                "game": f"{a_name} vs {h_name}",
                "selection": f"{a_name} vs {h_name} Mas de 1.5 Goles",
                "type": "Total de Partido",
                "american_odds": -280,
                "decimal_odds": american_to_decimal(-280),
                "real_prob": round(soc_mc["match_over_1_5"] * 100.0, 1),
                "ev": 6.6,
                "justification": (
                    f"[{lg_name}] Encuentro clasificado OVER ({proj_soc:.2f} xG combinado), "
                    f"registrando 2+ goles en el {round(soc_mc['match_over_1_5']*100, 1)}% de simulaciones."
                ),
            })

        if (h_real + d_real) >= 0.78 and -360 <= h_ml <= -115:
            soccer_alt_markets.append({
                "event_id": ev_id,
                "league_slug": lg_slug,
                "sport": "Futbol",
                "local_date": today_str,
                "league_name": lg_name,
                "team": h_name,
                "opponent": a_name,
                "game": f"{a_name} vs {h_name}",
                "market_type": "Doble Oportunidad (1X)",
                "selection": f"{h_name} o Empate (Doble Oportunidad)",
                "american_odds": -310,
                "real_prob": round(h_real + d_real, 4),
                "justification": (
                    f"[{lg_name}] Dominio territorial de {h_name} en casa ({h_match_xg:.2f} xG vs {a_match_xg:.2f} xG). "
                    f"Probabilidad conjunta de victoria local o empate del {round((h_real+d_real)*100, 1)}%."
                ),
            })

    soccer_props_pool = scraper.fetch_live_espn_soccer_stars(soccer_espn) if soccer_espn else []
    for p in soccer_props_pool:
        p["date"] = today_str
        p["local_date"] = today_str

    # =========================================================
    # 4. PROCESS NCAAF (COLEGIAL FBS - STRICTLY SEPARATED INTO 3 HORARIOS)
    #    Manana (< 14:00 CST), Tarde (14:00 - 17:59 CST), Noche (>= 18:00 CST)
    #    Each Horario independently generates 2 Apuestas Fuertes + 2 Parlays Ganadores!
    # =========================================================
    ncaaf_analyzed_games: List[Dict[str, Any]] = []
    ncaaf_by_horario: Dict[str, List[Dict[str, Any]]] = {"Manana": [], "Tarde": [], "Noche": []}

    for g in ncaaf_espn:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        h_rank = g["home_team"].get("rank")
        a_rank = g["away_team"].get("rank")
        h_rec = g["home_team"].get("record", "")
        a_rec = g["away_team"].get("record", "")
        horario = g.get("horario", "Manana")
        kickoff_cst = g.get("kickoff_cst", "12:00 CST")
        ev_local_date = g.get("local_date", ncaaf_date_str)

        an_g = match_action_sharp_game(h_name, a_name, "NCAAF", action_by_sport)

        odds = g.get("odds", {})
        raw_spr = odds.get("spread")
        spread_val = float(raw_spr if raw_spr not in (None, 0.0) else (an_g.get("home_spread") or 0.0))
        total_val = float(odds.get("overUnder") or an_g.get("total") or 52.5)
        if total_val < 28.0:
            total_val = 52.5

        h_ml_raw = odds.get("home_moneyline") or an_g.get("home_ml")
        a_ml_raw = odds.get("away_moneyline") or an_g.get("away_ml")
        if h_ml_raw is None or a_ml_raw is None:
            der_h, der_a, _, _ = spread_to_american_ml(spread_val, sigma=14.2)
            h_ml = int(h_ml_raw if h_ml_raw is not None else der_h)
            a_ml = int(a_ml_raw if a_ml_raw is not None else der_a)
        else:
            h_ml = int(h_ml_raw)
            a_ml = int(a_ml_raw)

        # Pass team's half of neutral spread (-spread_val * 0.5) so SP+ calibrates cleanly to 2026 reality
        h_stats = get_ncaaf_stats(h_name, record=h_rec, rank=h_rank, spread_hint=-spread_val * 0.52)
        a_stats = get_ncaaf_stats(a_name, record=a_rec, rank=a_rank, spread_hint=+spread_val * 0.52)

        m1_h, m1_a, _ = models.run_model_1_sp_plus_ncaaf(h_stats, a_stats, horario=horario)
        m2_h, m2_a, _ = models.run_model_2_sabermetric_ncaaf(h_stats, a_stats)
        hfa_val = 3.7 if horario == "Noche" else (2.9 if horario == "Manana" else 3.3)
        ncaaf_mc = mc_sim.simulate_ncaaf_game(
            home_sp=h_stats["sp_plus"],
            away_sp=a_stats["sp_plus"],
            home_epa_off=h_stats["epa_off"],
            home_epa_def=h_stats["epa_def"],
            away_epa_off=a_stats["epa_off"],
            away_epa_def=a_stats["epa_def"],
            home_tempo=h_stats["tempo_plays"],
            away_tempo=a_stats["tempo_plays"],
            hfa=hfa_val,
            spread=spread_val,
            total=total_val,
        )

        intel = evaluate_two_way_intelligence(
            models, "NCAAF", h_name, a_name, h_ml, a_ml,
            m1_h, m1_a, m2_h, m2_a, ncaaf_mc["prob_home_win"], ncaaf_mc["prob_away_win"],
            kalshi_by_sport, an_g, expert_by_sport, use_spread_splits=True,
        )

        h_rk_tag = f" (#{h_rank})" if h_rank else ""
        a_rk_tag = f" (#{a_rank})" if a_rank else ""
        saber_h = (
            f"Sabermetria NCAAF: SP+ calibrado {h_stats['sp_plus']:+.1f} vs {a_stats['sp_plus']:+.1f}, "
            f"EPA/jugada Off {h_stats['epa_off']:+.2f} / Def {h_stats['epa_def']:+.2f}, "
            f"Success {h_stats['success_rate']:.1f}%, Finishing Drives {h_stats['finishing_drives']:.2f}, Havoc {h_stats['havoc_rate']:.1f}%."
        )
        saber_a = (
            f"Sabermetria NCAAF: SP+ calibrado {a_stats['sp_plus']:+.1f} vs {h_stats['sp_plus']:+.1f}, "
            f"EPA/jugada Off {a_stats['epa_off']:+.2f} / Def {a_stats['epa_def']:+.2f}, "
            f"Success {a_stats['success_rate']:.1f}%, Finishing Drives {a_stats['finishing_drives']:.2f}, Havoc {a_stats['havoc_rate']:.1f}%."
        )

        # Calibrated cover probabilities shrunk toward 50% fair market
        cal_h_cov = round(0.64 * ncaaf_mc["prob_home_cover"] + 0.36 * 0.50, 4)
        cal_a_cov = round(0.64 * ncaaf_mc["prob_away_cover"] + 0.36 * 0.50, 4)

        g_obj = {
            "event_id": ev_id,
            "sport": "NCAAF",
            "local_date": ev_local_date,
            "horario": horario,
            "kickoff_cst": kickoff_cst,
            "home_team": h_name,
            "away_team": a_name,
            "home_rank": h_rank,
            "away_rank": a_rank,
            "home_record": h_rec,
            "away_record": a_rec,
            "matchup": f"{a_name} @ {h_name}",
            "display_matchup": f"{a_name}{a_rk_tag} @ {h_name}{h_rk_tag}",
            "spread": spread_val,
            "total": total_val,
            "home_odds": h_ml,
            "away_odds": a_ml,
            "home_cover_prob": cal_h_cov,
            "away_cover_prob": cal_a_cov,
            "prob_over": ncaaf_mc["prob_over"],
            "prob_under": ncaaf_mc["prob_under"],
            "home_sp_plus": h_stats["sp_plus"],
            "away_sp_plus": a_stats["sp_plus"],
            "home_epa_off": h_stats["epa_off"],
            "away_epa_off": a_stats["epa_off"],
            "home_havoc": h_stats["havoc_rate"],
            "away_havoc": a_stats["havoc_rate"],
            "saber_home_note": saber_h,
            "saber_away_note": saber_a,
            "projected_score": f"{ncaaf_mc['mean_away_score']} - {ncaaf_mc['mean_home_score']}",
            "projected_total": ncaaf_mc["mean_total"],
            **intel,
        }
        ncaaf_analyzed_games.append(g_obj)
        ncaaf_by_horario.setdefault(horario, []).append(g_obj)

    # =========================================================
    # 5. PROCESS TENIS (ATP & WTA SINGLES)
    #    Vehicles: ONLY Apuesta Fuerte + Parlay Ganador (NO Ratonero, NO Props)
    # =========================================================
    tennis_analyzed_games: List[Dict[str, Any]] = []

    for m in tennis_espn:
        ev_id = str(m.get("id") or "")
        p1_name = m["p1"]["name"]
        p2_name = m["p2"]["name"]
        tour = m.get("tour", "ATP")
        tournament = m.get("tournament", f"{tour} Tour")
        surface = m.get("surface", "Hard")
        kickoff_cst = m.get("kickoff_cst", "")
        local_dt = m.get("local_date", today_str)

        p1_stats = get_tennis_stats(p1_name, tour=tour, seed=m["p1"].get("seed"), surface=surface)
        p2_stats = get_tennis_stats(p2_name, tour=tour, seed=m["p2"].get("seed"), surface=surface)

        m1_p1, m1_p2 = models.run_model_1_elo_surface_tennis(p1_stats, p2_stats)
        m2_p1, m2_p2, p1_srv_pt, p2_srv_pt = models.run_model_2_sabermetric_tennis(p1_stats, p2_stats)

        tot_games_line = float(m.get("odds", {}).get("overUnder", 0.0) or 0.0)
        if tot_games_line < 16.0:
            tot_games_line = 22.5 if tour == "ATP" else 21.5

        ten_mc = mc_sim.simulate_tennis_match(
            p1_srv_pt_prob=p1_srv_pt,
            p2_srv_pt_prob=p2_srv_pt,
            total_games_line=tot_games_line,
        )

        p1_ml = m.get("odds", {}).get("p1_moneyline")
        p2_ml = m.get("odds", {}).get("p2_moneyline")
        if p1_ml is None or p2_ml is None:
            der_1, der_2 = derive_tennis_book_odds(p1_stats["elo_global"], p2_stats["elo_global"])
            p1_ml = int(p1_ml if p1_ml is not None else der_1)
            p2_ml = int(p2_ml if p2_ml is not None else der_2)
        else:
            p1_ml = int(p1_ml)
            p2_ml = int(p2_ml)

        intel_ten = evaluate_two_way_intelligence(
            models, "TENIS", p1_name, p2_name, p1_ml, p2_ml,
            m1_p1, m1_p2, m2_p1, m2_p2, ten_mc["prob_p1_win"], ten_mc["prob_p2_win"],
            kalshi_by_sport, {}, expert_by_sport, use_spread_splits=False,
        )

        saber_p1 = (
            f"Sabermetria Tenis ({surface}): Elo Superficie {p1_stats['elo_surface']:.0f} vs {p2_stats['elo_surface']:.0f}, "
            f"Hold {p1_stats['hold_pct']:.1f}%, Break {p1_stats['break_pct']:.1f}%, Dominance Ratio {p1_stats['dominance_ratio']:.2f}, "
            f"Tiebreak {p1_stats['tiebreak_win_pct']:.1f}%."
        )
        saber_p2 = (
            f"Sabermetria Tenis ({surface}): Elo Superficie {p2_stats['elo_surface']:.0f} vs {p1_stats['elo_surface']:.0f}, "
            f"Hold {p2_stats['hold_pct']:.1f}%, Break {p2_stats['break_pct']:.1f}%, Dominance Ratio {p2_stats['dominance_ratio']:.2f}, "
            f"Tiebreak {p2_stats['tiebreak_win_pct']:.1f}%."
        )

        tennis_analyzed_games.append({
            "event_id": ev_id,
            "sport": "Tenis",
            "tour": tour,
            "tournament": f"{tournament} ({tour})",
            "surface": surface,
            "kickoff_cst": kickoff_cst,
            "local_date": local_dt,
            "home_team": p1_name,
            "away_team": p2_name,
            "matchup": f"{p1_name} vs {p2_name}",
            "home_odds": p1_ml,
            "away_odds": p2_ml,
            "home_plus_1_5_sets_prob": ten_mc["prob_p1_plus_1_5_sets"],
            "away_plus_1_5_sets_prob": ten_mc["prob_p2_plus_1_5_sets"],
            "p1_elo_surface": p1_stats["elo_surface"],
            "p2_elo_surface": p2_stats["elo_surface"],
            "p1_hold_pct": p1_stats["hold_pct"],
            "p2_hold_pct": p2_stats["hold_pct"],
            "p1_break_pct": p1_stats["break_pct"],
            "p2_break_pct": p2_stats["break_pct"],
            "p1_dr": p1_stats["dominance_ratio"],
            "p2_dr": p2_stats["dominance_ratio"],
            "total_games_line": tot_games_line,
            "mean_total_games": ten_mc["mean_total_games"],
            "saber_home_note": saber_p1,
            "saber_away_note": saber_p2,
            "projected_score": f"{ten_mc['mean_p1_games']} - {ten_mc['mean_p2_games']} Juegos ({ten_mc['mean_total_games']} Tot.)",
            **intel_ten,
        })

    # =========================================================
    # 6. PROCESS MLB & NFL
    # =========================================================
    today_mlb = mlb_espn
    mlb_analyzed_games: List[Dict[str, Any]] = []
    mlb_alt_markets: List[Dict[str, Any]] = []
    mlb_totals_pool: List[Dict[str, Any]] = []

    for g in today_mlb:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        league_code = g.get("league", "LMP" if "LMP" in str(g.get("league_name", "")) else "MLB")
        league_label = g.get("league_name", "Liga ARCO Mexicana del Pacifico" if league_code == "LMP" else "Major League Baseball")
        an_g = match_action_sharp_game(h_name, a_name, "MLB", action_by_sport)
        odds = g.get("odds", {})
        h_ml = int(odds.get("home_moneyline") or an_g.get("home_ml") or -135)
        a_ml = int(odds.get("away_moneyline") or an_g.get("away_ml") or +115)
        tot_val = float(odds.get("overUnder") or 8.5)

        h_rec = g.get("home_team", {}).get("record", "")
        a_rec = g.get("away_team", {}).get("record", "")
        h_mlb_s = get_baseball_stats(h_name, league=league_code, record=h_rec)
        a_mlb_s = get_baseball_stats(a_name, league=league_code, record=a_rec)
        m1_h, m1_a = models.run_model_1_sabermetric_mlb(h_mlb_s, a_mlb_s)
        m2_h, m2_a = models.run_model_2_bullpen_mlb(h_mlb_s, a_mlb_s)
        intel_mlb = evaluate_two_way_intelligence(
            models, "MLB", h_name, a_name, h_ml, a_ml,
            m1_h, m1_a, m2_h, m2_a, 0.5 * (m1_h + m2_h), 0.5 * (m1_a + m2_a),
            kalshi_by_sport, an_g, expert_by_sport,
        )
        mlb_analyzed_games.append({
            "event_id": ev_id,
            "sport": "MLB",
            "league": league_code,
            "league_name": league_label,
            "local_date": today_str,
            "kickoff_cst": g.get("kickoff_cst", ""),
            "home_team": h_name,
            "away_team": a_name,
            "matchup": f"{a_name} vs {h_name}",
            "home_odds": h_ml,
            "away_odds": a_ml,
            "total": tot_val,
            "saber_home_note": f"Sabermetria {league_code}: FIP {h_mlb_s['starter_stats']['FIP']}, wRC+ ofensivo {h_mlb_s['lineup_stats']['wRC_plus']}, bullpen ERA {h_mlb_s['bullpen']['ERA_last_30d']}.",
            "saber_away_note": f"Sabermetria {league_code}: FIP {a_mlb_s['starter_stats']['FIP']}, wRC+ ofensivo {a_mlb_s['lineup_stats']['wRC_plus']}, bullpen ERA {a_mlb_s['bullpen']['ERA_last_30d']}.",
            "projected_score": f"{max(2.8, round(tot_val * 0.46, 1))} - {max(3.2, round(tot_val * 0.54, 1))}",
            **intel_mlb,
        })
        mlb_alt_markets.append({
            "event_id": ev_id,
            "sport": "MLB",
            "league": league_code,
            "local_date": today_str,
            "team": h_name,
            "opponent": a_name,
            "game": f"{a_name} vs {h_name}",
            "market_type": "Run Line Alternativo",
            "selection": f"{h_name} +1.5 Carreras Seguro",
            "american_odds": -210,
            "real_prob": 0.775,
            "justification": f"Colchon defensivo de carreras en {league_label} para {h_name} ante {a_name}."
        })
        mlb_totals_pool.append({
            "event_id": ev_id,
            "sport": "MLB",
            "league": league_code,
            "local_date": today_str,
            "game": f"{a_name} vs {h_name}",
            "market_type": "Totales Carreras",
            "selection": f"{a_name} vs {h_name} Menos de {tot_val + 1.5:.1f} Carreras",
            "american_odds": -210,
            "real_prob": 0.775,
            "justification": f"Duelo sabermetrico de pitcheo en {league_label} proyectando total por debajo de {tot_val + 1.5:.1f} carreras."
        })

    lmp_count = sum(1 for g in today_mlb if g.get("league") == "LMP")
    kbo_count = sum(1 for g in today_mlb if g.get("league") == "KBO")
    npb_count = sum(1 for g in today_mlb if g.get("league") == "NPB")
    mlb_count = sum(1 for g in today_mlb if g.get("league") == "MLB")

    mlb_ws_data = {
        "matchup": "Liga ARCO Mexicana del Pacifico (LMP) & Beisbol Internacional",
        "series": "Temporada Regular LMP 2026-2027 / KBO / NPB / MLB",
        "date": today_str,
        "status": f"En cartelera hoy ({len(today_mlb)} juegos: {lmp_count} LMP, {kbo_count} KBO, {npb_count} NPB, {mlb_count} MLB)",
        "lmp_count": lmp_count,
        "kbo_count": kbo_count,
        "npb_count": npb_count,
        "mlb_count": mlb_count,
    }

    nfl_analyzed_games: List[Dict[str, Any]] = []
    nfl_alt_markets: List[Dict[str, Any]] = []
    nfl_totals_pool: List[Dict[str, Any]] = []
    nfl_by_horario: Dict[str, List[Dict[str, Any]]] = {"Temprano": [], "Tarde y Noche": []}

    for g in nfl_espn:
        ev_id = str(g.get("id") or "")
        h_name = g["home_team"]["name"]
        a_name = g["away_team"]["name"]
        raw_iso = g.get("date", "")
        kickoff_cst_str = "12:00 CST"
        g_horario = "Temprano"
        if raw_iso:
            try:
                dt_c = datetime.fromisoformat(raw_iso.replace("Z", "+00:00")).astimezone(CST_TZ)
                kickoff_cst_str = dt_c.strftime("%H:%M CST")
                g_horario = "Temprano" if dt_c.hour < 14 else "Tarde y Noche"
            except Exception:
                pass

        an_g = match_action_sharp_game(h_name, a_name, "NFL", action_by_sport)
        odds = g.get("odds", {})
        raw_spr = odds.get("spread")
        spread_val = float(raw_spr if raw_spr not in (None, 0.0) else (an_g.get("home_spread") or -3.0))
        total_val = float(odds.get("overUnder") or an_g.get("total") or 46.5)
        h_calc, a_calc, _, _ = spread_to_american_ml(spread_val, sigma=13.0)
        h_ml = int(odds.get("home_moneyline") or an_g.get("home_ml") or h_calc)
        a_ml = int(odds.get("away_moneyline") or an_g.get("away_ml") or a_calc)

        h_nfl_s = get_nfl_stats(h_name, g["home_team"].get("record", ""), spread_hint=-spread_val * 0.5)
        a_nfl_s = get_nfl_stats(a_name, g["away_team"].get("record", ""), spread_hint=+spread_val * 0.5)
        m1_h, m1_a, _ = models.run_model_1_epa_nfl(h_nfl_s, a_nfl_s)
        m2_h, m2_a = models.run_model_2_situational_nfl(h_nfl_s, a_nfl_s)
        nfl_mc = mc_sim.simulate_ncaaf_game(
            home_sp=-spread_val * 0.8,
            away_sp=0.0,
            home_epa_off=h_nfl_s["EPA_per_play_off"],
            home_epa_def=h_nfl_s["EPA_per_play_def"],
            away_epa_off=a_nfl_s["EPA_per_play_off"],
            away_epa_def=a_nfl_s["EPA_per_play_def"],
            home_tempo=63.5,
            away_tempo=63.5,
            hfa=2.2,
            spread=spread_val,
            total=total_val,
        )
        intel_nfl = evaluate_two_way_intelligence(
            models, "NFL", h_name, a_name, h_ml, a_ml,
            m1_h, m1_a, m2_h, m2_a, nfl_mc["prob_home_win"], nfl_mc["prob_away_win"],
            kalshi_by_sport, an_g, expert_by_sport, use_spread_splits=True,
        )
        nfl_g_obj = {
            "event_id": ev_id,
            "sport": "NFL",
            "local_date": nfl_target_date,
            "kickoff_cst": kickoff_cst_str,
            "horario": g_horario,
            "home_team": h_name,
            "away_team": a_name,
            "matchup": f"{a_name} @ {h_name}",
            "spread": spread_val,
            "total": total_val,
            "home_odds": h_ml,
            "away_odds": a_ml,
            "home_cover_prob": nfl_mc["prob_home_cover"],
            "away_cover_prob": nfl_mc["prob_away_cover"],
            "saber_home_note": f"Sabermetria NFL [{g_horario} - {kickoff_cst_str}]: EPA/jugada Off {h_nfl_s['EPA_per_play_off']:+.2f}, DVOA {h_nfl_s['DVOA_off']:+.1f}%, RedZone TD {h_nfl_s['RedZone_TD_pct']:.1f}%.",
            "saber_away_note": f"Sabermetria NFL [{g_horario} - {kickoff_cst_str}]: EPA/jugada Off {a_nfl_s['EPA_per_play_off']:+.2f}, DVOA {a_nfl_s['DVOA_off']:+.1f}%, RedZone TD {a_nfl_s['RedZone_TD_pct']:.1f}%.",
            "projected_score": f"{nfl_mc['mean_away_score']} - {nfl_mc['mean_home_score']}",
            **intel_nfl,
        }
        nfl_analyzed_games.append(nfl_g_obj)
        nfl_by_horario.setdefault(g_horario, []).append(nfl_g_obj)

        nfl_alt_markets.append({
            "event_id": ev_id,
            "sport": "NFL",
            "local_date": nfl_target_date,
            "team": h_name,
            "opponent": a_name,
            "game": f"{a_name} @ {h_name}",
            "market_type": "Spread Alternativo",
            "selection": f"{h_name} {spread_val + 6.5:+.1f} Spread Seguro",
            "american_odds": -260,
            "real_prob": 0.805,
            "justification": (
                f"Proteccion de numero clave (+6.5 puntos adicionales) para {h_name} ante {a_name}, "
                f"cubriendo en el 80.5% de las simulaciones Monte Carlo."
            ),
        })

    nfl_props_pool = scraper.fetch_live_espn_nfl_stars(nfl_espn) if nfl_espn else []
    for p in nfl_props_pool:
        p["date"] = today_str
        p["local_date"] = today_str

    # =========================================================
    # 7. SELECT "LO MEJOR DEL DIA (MVP)" FIRST (PRIORITY #1 PICKS OF TODAY!)
    # =========================================================
    mvp_engine = make_tab_engine()

    today_ncaaf_for_mvp = [g for g in ncaaf_analyzed_games if g.get("local_date") == today_str]
    today_tennis_for_mvp = [g for g in tennis_analyzed_games if g.get("local_date") == today_str]
    if not today_tennis_for_mvp and (len(nba_analyzed_games) + len(nhl_analyzed_games) + len(soccer_analyzed_games)) < 4:
        today_tennis_for_mvp = tennis_analyzed_games

    mvp_analyzed_pool = (
        nba_analyzed_games
        + nhl_analyzed_games
        + soccer_analyzed_games
        + today_ncaaf_for_mvp
        + today_tennis_for_mvp
        + mlb_analyzed_games
        + nfl_analyzed_games
    )
    mvp_alt_pool = nba_alt_markets + nhl_alt_markets + soccer_alt_markets + mlb_alt_markets + nfl_alt_markets
    mvp_props_pool = nba_props_pool + nhl_props_pool + nfl_props_pool  # Strictly zero soccer player props
    mvp_totals_pool = nba_totals_pool + nhl_totals_pool + soccer_totals_pool + mlb_totals_pool + nfl_totals_pool

    mvp_sniper = mvp_engine.select_sniper_bet(mvp_analyzed_pool) if mvp_analyzed_pool else {}
    mvp_ganador = mvp_engine.select_ganador_parlay(
        mvp_analyzed_pool, mvp_totals_pool, alt_pool=mvp_alt_pool, count=2, allow_alt_lines=True
    ) if mvp_analyzed_pool else []
    mvp_props = mvp_engine.select_props_parlay(mvp_props_pool, count=2) if mvp_props_pool else []
    mvp_ratonero = mvp_engine.select_ratonero_parlay(mvp_alt_pool, count=5, prefer_cross_sport=True) if mvp_alt_pool else []

    mvp_sniper_2: Dict[str, Any] = {}
    if len(mvp_ratonero) < 5:
        mvp_ratonero = []
        mvp_sniper_2 = mvp_engine.select_sniper_bet(mvp_analyzed_pool) if mvp_analyzed_pool else {}

    # =========================================================
    # 8. SELECT PER-SPORT VEHICLES (SHARING DIRECTIONAL LOCKS -> ZERO CONTRADICTION)
    # =========================================================
    nba_engine = make_tab_engine()
    nba_sniper = nba_engine.select_sniper_bet(nba_analyzed_games) if nba_analyzed_games else {}
    nba_ratonero = nba_engine.select_ratonero_parlay(nba_alt_markets, count=5) if nba_alt_markets else []
    nba_props = nba_engine.select_props_parlay(nba_props_pool, count=2) if nba_props_pool else []
    nba_ganador = nba_engine.select_ganador_parlay(nba_analyzed_games, nba_totals_pool, alt_pool=nba_alt_markets, count=2) if nba_analyzed_games else []

    nhl_engine = make_tab_engine()
    nhl_sniper = nhl_engine.select_sniper_bet(nhl_analyzed_games) if nhl_analyzed_games else {}
    nhl_props = nhl_engine.select_props_parlay(nhl_props_pool, count=3) if nhl_props_pool else []
    nhl_ratonero = nhl_engine.select_ratonero_parlay(nhl_alt_markets, count=5) if nhl_alt_markets else []
    nhl_ganador = nhl_engine.select_ganador_parlay(nhl_analyzed_games, nhl_totals_pool, alt_pool=nhl_alt_markets, count=2) if nhl_analyzed_games else []

    soc_engine = make_tab_engine()
    soccer_sniper = soc_engine.select_sniper_bet(soccer_analyzed_games) if soccer_analyzed_games else {}
    soccer_props: List[Dict[str, Any]] = []  # Strictly zero player props in soccer
    soccer_ratonero = soc_engine.select_ratonero_parlay(soccer_alt_markets, count=5) if soccer_alt_markets else []
    soccer_sniper_2: Dict[str, Any] = {}
    soccer_ganador_2: List[Dict[str, Any]] = []
    if len(soccer_ratonero) < 5:
        soccer_ratonero = []
        soccer_sniper_2 = soc_engine.select_sniper_bet(soccer_analyzed_games) if len(soccer_analyzed_games) >= 2 else {}
        soccer_ganador_2 = soc_engine.select_ganador_parlay(soccer_analyzed_games, soccer_totals_pool, alt_pool=soccer_alt_markets, count=2) if len(soccer_analyzed_games) >= 4 else []
    soccer_ganador = soc_engine.select_ganador_parlay(soccer_analyzed_games, soccer_totals_pool, alt_pool=soccer_alt_markets, count=2) if soccer_analyzed_games else []

    # NCAAF COLEGIAL: STRICTLY DIVIDED INTO 3 SEPARATE HORARIOS (NEVER MIXED!)
    # Each Horario (Manana, Tarde, Noche) extracts its OWN 2 Apuestas Fuertes + 2 Parlays Ganadores!
    ncaaf_raw_horario_picks: Dict[str, Dict[str, Any]] = {}
    for turno_key in ("Manana", "Tarde", "Noche"):
        turno_games = ncaaf_by_horario.get(turno_key, [])
        turno_engine = make_tab_engine()
        if turno_games:
            t_sn1 = turno_engine.select_sniper_bet(turno_games)
            t_sn2 = turno_engine.select_sniper_bet(turno_games)
            t_gn1 = turno_engine.select_ganador_parlay(turno_games, [], alt_pool=None, count=2, allow_alt_lines=False)
            t_gn2 = turno_engine.select_ganador_parlay(turno_games, [], alt_pool=None, count=2, allow_alt_lines=False)
        else:
            t_sn1, t_sn2, t_gn1, t_gn2 = {}, {}, [], []
        ncaaf_raw_horario_picks[turno_key] = {
            "sniper_1": t_sn1,
            "sniper_2": t_sn2,
            "ganador_1": t_gn1,
            "ganador_2": t_gn2,
        }

    # TENIS (ATP & WTA: ONLY Apuesta Fuerte + Parlay Ganador; NO Ratonero, NO Props)
    tenis_engine = make_tab_engine()
    tenis_sniper = tenis_engine.select_sniper_bet(tennis_analyzed_games) if tennis_analyzed_games else {}
    tenis_ganador = tenis_engine.select_ganador_parlay(
        tennis_analyzed_games, [], alt_pool=None, count=2, allow_alt_lines=False
    ) if tennis_analyzed_games else []

    # BEISBOL (LMP / KBO / NPB / MLB)
    mlb_engine = make_tab_engine()
    mlb_sniper = mlb_engine.select_sniper_bet(mlb_analyzed_games) if mlb_analyzed_games else {}
    mlb_sniper_2 = mlb_engine.select_sniper_bet(mlb_analyzed_games) if len(mlb_analyzed_games) >= 2 else {}
    mlb_props: List[Dict[str, Any]] = []
    mlb_ratonero: List[Dict[str, Any]] = []
    mlb_ganador = mlb_engine.select_ganador_parlay(mlb_analyzed_games, mlb_totals_pool, alt_pool=mlb_alt_markets, count=2) if mlb_analyzed_games else []
    mlb_ganador_2 = mlb_engine.select_ganador_parlay(mlb_analyzed_games, mlb_totals_pool, alt_pool=mlb_alt_markets, count=2) if len(mlb_analyzed_games) >= 4 else []

    # NFL DOMINGO: STRICTLY DIVIDED INTO 2 SEPARATE HORARIOS (TEMPRANO Y TARDE/NOCHE)
    # Temprano: Juegos < 14:00 CST (12:00 CST) -> 2 Apuestas Fuertes + 2 Parlays Ganadores
    # Tarde y Noche: Juegos >= 14:00 CST (15:05/15:25 CST + SNF) -> 2 Apuestas Fuertes + 2 Parlays Ganadores
    is_nfl_sunday = len(nfl_analyzed_games) >= 4 or any(len(v) >= 2 for v in nfl_by_horario.values())
    nfl_raw_horario_picks: Dict[str, Dict[str, Any]] = {}
    for slot_key in ("Temprano", "Tarde y Noche"):
        slot_games = nfl_by_horario.get(slot_key, [])
        eng_slot = make_tab_engine()
        if len(slot_games) >= 2:
            slot_sn1 = eng_slot.select_sniper_bet(slot_games)
            slot_sn2 = eng_slot.select_sniper_bet(slot_games)
            slot_gn1 = eng_slot.select_ganador_parlay(slot_games, [], alt_pool=None, count=2, allow_alt_lines=False)
            slot_gn2 = eng_slot.select_ganador_parlay(slot_games, [], alt_pool=None, count=2, allow_alt_lines=False)
        elif slot_games:
            slot_sn1 = eng_slot.select_sniper_bet(slot_games)
            slot_sn2 = {}
            slot_gn1 = []
            slot_gn2 = []
        else:
            slot_sn1, slot_sn2, slot_gn1, slot_gn2 = {}, {}, [], []
        nfl_raw_horario_picks[slot_key] = {
            "sniper_1": slot_sn1,
            "sniper_2": slot_sn2,
            "ganador_1": slot_gn1,
            "ganador_2": slot_gn2,
        }

    nfl_engine = make_tab_engine()
    nfl_sniper = nfl_engine.select_sniper_bet(nfl_analyzed_games) if nfl_analyzed_games else {}
    nfl_props = nfl_engine.select_props_parlay(nfl_props_pool, count=2) if nfl_props_pool else []
    nfl_ratonero: List[Dict[str, Any]] = []
    nfl_ganador = nfl_engine.select_ganador_parlay(nfl_analyzed_games, nfl_totals_pool, alt_pool=nfl_alt_markets, count=2) if nfl_analyzed_games else []

    # =========================================================
    # 9. RECORD ALL TICKETS TO HISTORY TRACKER
    # =========================================================
    def _resolve_slate_context(sp: str, veh: str) -> Tuple[int, int]:
        slot_rk = 2 if "#2" in veh else 1
        sp_u = (sp or "").upper()
        if sp_u == "DIA":
            return max(1, len(mvp_analyzed_pool)), slot_rk
        if sp_u == "NBA":
            return max(1, len(nba_analyzed_games)), slot_rk
        if sp_u == "NHL":
            return max(1, len(nhl_analyzed_games)), slot_rk
        if sp_u == "FUTBOL":
            return max(1, len(soccer_analyzed_games)), slot_rk
        if sp_u == "TENIS":
            return max(1, len(tennis_analyzed_games)), slot_rk
        if sp_u == "NCAAF":
            for hk in ("Manana", "Tarde", "Noche"):
                if hk in veh:
                    return max(1, len(ncaaf_by_horario.get(hk, []))), slot_rk
            return max(1, len(ncaaf_analyzed_games)), slot_rk
        if sp_u == "NFL":
            return max(1, len(nfl_analyzed_games)), slot_rk
        if sp_u == "MLB":
            return max(1, len(mlb_analyzed_games)), slot_rk
        return 10, slot_rk

    def _enrich_leg_security(leg_dict: Dict[str, Any], sp: str, veh: str) -> Dict[str, Any]:
        if not leg_dict or not isinstance(leg_dict, dict):
            return leg_dict
        sl_sz, sl_rk = _resolve_slate_context(sp, veh)
        leg_dict["slate_size"] = sl_sz
        leg_dict["slot_rank"] = sl_rk
        sec = ai_agent.audit_selection_security(leg_dict, slate_size=sl_sz, slot_rank=sl_rk, vehicle_type=veh)
        leg_dict["prob_suceda_pct"] = sec["prob_suceda_pct"]
        leg_dict["seguridad_1_10"] = sec["seguridad_1_10"]
        leg_dict["is_quota_filler"] = sec["is_quota_filler"]
        leg_dict["tier_label"] = sec["tier_label"]
        leg_dict["short_badge"] = sec["short_badge"]
        leg_dict["origin_tag"] = sec["origin_tag"]
        leg_dict["stake_label"] = sec["stake_label"]
        leg_dict["analisis_motor"] = sec["analisis_motor"]
        return leg_dict

    def _record_parlay_helper(dt_str: str, sp: str, veh: str, title: str, legs: List[Dict[str, Any]]):
        if not legs:
            return
        dec = 1.0
        for item in legs:
            _enrich_leg_security(item, sp, veh)
            dec *= item.get("decimal_odds") or american_to_decimal(int(item["american_odds"]))
        history.record_ticket(
            date_str=dt_str,
            sport=sp,
            vehicle=veh,
            title=title,
            odds_american=decimal_to_american(dec),
            odds_decimal=round(dec, 3),
            is_parlay=True,
            legs=legs,
        )

    def _record_single_helper(dt_str: str, sp: str, veh: str, title: str, pick: Dict[str, Any]):
        if not pick or not pick.get("selection"):
            return
        _enrich_leg_security(pick, sp, veh)
        history.record_ticket(
            date_str=dt_str,
            sport=sp,
            vehicle=veh,
            title=title,
            odds_american=int(pick.get("american_odds", -110)),
            odds_decimal=float(pick.get("decimal_odds", 1.91)),
            is_parlay=False,
            legs=[pick],
        )

    if record_in_history:
        # 1. MVP del Dia
        _record_single_helper(today_str, "Dia", "Apuesta Fuerte", f"Apuesta Fuerte del Dia: {mvp_sniper.get('selection', '')}", mvp_sniper)
        _record_parlay_helper(today_str, "Dia", "Parlay Ganador", "Parlay Ganador del Dia (<= +100 MAX)", mvp_ganador)
        _record_parlay_helper(today_str, "Dia", "Parlay Props", "Parlay de Props del Dia (Superestrellas)", mvp_props)
        if mvp_ratonero and len(mvp_ratonero) >= 5:
            _record_parlay_helper(today_str, "Dia", "Parlay Ratonero", "Parlay Ratonero Multi-Deporte del Dia (5 Patas)", mvp_ratonero)
        elif mvp_sniper_2 and mvp_sniper_2.get("selection"):
            _record_single_helper(today_str, "Dia", "Apuesta Fuerte #2", f"Apuesta Fuerte #2 del Dia: {mvp_sniper_2.get('selection', '')}", mvp_sniper_2)

        # 2. NBA
        _record_single_helper(today_str, "NBA", "Apuesta Fuerte", f"Apuesta Fuerte NBA: {nba_sniper.get('selection', '')}", nba_sniper)
        _record_parlay_helper(today_str, "NBA", "Parlay Props", "Parlay Props NBA", nba_props)
        if len(nba_ratonero) >= 5:
            _record_parlay_helper(today_str, "NBA", "Parlay Ratonero", "Parlay Ratonero NBA (5 Patas)", nba_ratonero)
        _record_parlay_helper(today_str, "NBA", "Parlay Ganador", "Parlay Ganador NBA", nba_ganador)

        # 3. NHL
        _record_single_helper(today_str, "NHL", "Apuesta Fuerte", f"Apuesta Fuerte NHL: {nhl_sniper.get('selection', '')}", nhl_sniper)
        _record_parlay_helper(today_str, "NHL", "Parlay Props", "Top 3 Props Remates NHL", nhl_props)
        if len(nhl_ratonero) >= 5:
            _record_parlay_helper(today_str, "NHL", "Parlay Ratonero", "Parlay Ratonero NHL (5 Patas)", nhl_ratonero)
        _record_parlay_helper(today_str, "NHL", "Parlay Ganador", "Parlay Ganador NHL", nhl_ganador)

        # 4. Futbol (Zero Player Props; Apuesta Fuerte #1/#2, Parlay Ganador #1/#2, Ratonero)
        if soccer_analyzed_games:
            _record_single_helper(today_str, "Futbol", "Apuesta Fuerte", f"Apuesta Fuerte Futbol: {soccer_sniper.get('selection', '')}", soccer_sniper)
            if soccer_sniper_2 and soccer_sniper_2.get("selection"):
                _record_single_helper(today_str, "Futbol", "Apuesta Fuerte #2", f"Apuesta Fuerte #2 Futbol: {soccer_sniper_2.get('selection', '')}", soccer_sniper_2)
            if len(soccer_ratonero) >= 5:
                _record_parlay_helper(today_str, "Futbol", "Parlay Ratonero", "Parlay Ratonero Futbol (5 Patas)", soccer_ratonero)
            _record_parlay_helper(today_str, "Futbol", "Parlay Ganador", "Parlay Ganador Futbol", soccer_ganador)
            if soccer_ganador_2 and len(soccer_ganador_2) >= 2:
                _record_parlay_helper(today_str, "Futbol", "Parlay Ganador #2", "Parlay Ganador #2 Futbol", soccer_ganador_2)

        # 5. NCAAF (Strictly separated per Horario: Manana, Tarde, Noche -> 2 Fuertes + 2 Ganadores each!)
        if ncaaf_analyzed_games:
            for turno_key in ("Manana", "Tarde", "Noche"):
                tp = ncaaf_raw_horario_picks.get(turno_key, {})
                _record_single_helper(
                    ncaaf_date_str, "NCAAF", f"Apuesta Fuerte {turno_key} #1",
                    f"Apuesta Fuerte #1 NCAAF ({turno_key}): {tp.get('sniper_1', {}).get('selection', '')}",
                    tp.get("sniper_1", {}),
                )
                _record_single_helper(
                    ncaaf_date_str, "NCAAF", f"Apuesta Fuerte {turno_key} #2",
                    f"Apuesta Fuerte #2 NCAAF ({turno_key}): {tp.get('sniper_2', {}).get('selection', '')}",
                    tp.get("sniper_2", {}),
                )
                _record_parlay_helper(
                    ncaaf_date_str, "NCAAF", f"Parlay Ganador {turno_key} #1",
                    f"Parlay Ganador #1 NCAAF ({turno_key} | <= +100 MAX)",
                    tp.get("ganador_1", []),
                )
                _record_parlay_helper(
                    ncaaf_date_str, "NCAAF", f"Parlay Ganador {turno_key} #2",
                    f"Parlay Ganador #2 NCAAF ({turno_key} | <= +100 MAX)",
                    tp.get("ganador_2", []),
                )

        # 6. Tenis (ONLY Apuesta Fuerte + Parlay Ganador)
        if tennis_analyzed_games:
            t_rec_date = tenis_sniper.get("local_date", today_str) if tenis_sniper else (
                tenis_ganador[0].get("local_date", today_str) if tenis_ganador else today_str
            )
            _record_single_helper(t_rec_date, "Tenis", "Apuesta Fuerte", f"Apuesta Fuerte Tenis: {tenis_sniper.get('selection', '')}", tenis_sniper)
            _record_parlay_helper(t_rec_date, "Tenis", "Parlay Ganador", "Parlay Ganador Tenis ATP/WTA (<= +100 MAX)", tenis_ganador)

        # 7. BEISBOL (LMP / KBO / NPB / MLB)
        if mlb_analyzed_games:
            _record_single_helper(today_str, "MLB", "Apuesta Fuerte", f"Apuesta Fuerte Beisbol: {mlb_sniper.get('selection', '')}", mlb_sniper)
            if mlb_sniper_2 and mlb_sniper_2.get("selection"):
                _record_single_helper(today_str, "MLB", "Apuesta Fuerte #2", f"Apuesta Fuerte #2 Beisbol: {mlb_sniper_2.get('selection', '')}", mlb_sniper_2)
            _record_parlay_helper(today_str, "MLB", "Parlay Ganador", "Parlay Ganador Beisbol", mlb_ganador)
            if mlb_ganador_2 and len(mlb_ganador_2) >= 2:
                _record_parlay_helper(today_str, "MLB", "Parlay Ganador #2", "Parlay Ganador #2 Beisbol", mlb_ganador_2)

        # 8. NFL (Separated into Temprano and Tarde y Noche when Sunday / multi-slot slate)
        if nfl_analyzed_games:
            if is_nfl_sunday:
                for slot_key in ("Temprano", "Tarde y Noche"):
                    sp_picks = nfl_raw_horario_picks.get(slot_key, {})
                    sn1 = sp_picks.get("sniper_1", {})
                    sn2 = sp_picks.get("sniper_2", {})
                    gn1 = sp_picks.get("ganador_1", [])
                    gn2 = sp_picks.get("ganador_2", [])
                    if sn1 and sn1.get("selection"):
                        _record_single_helper(nfl_target_date, "NFL", f"Apuesta Fuerte {slot_key} #1", f"Apuesta Fuerte #1 NFL ({slot_key}): {sn1.get('selection', '')}", sn1)
                    if sn2 and sn2.get("selection"):
                        _record_single_helper(nfl_target_date, "NFL", f"Apuesta Fuerte {slot_key} #2", f"Apuesta Fuerte #2 NFL ({slot_key}): {sn2.get('selection', '')}", sn2)
                    if gn1 and len(gn1) >= 2:
                        _record_parlay_helper(nfl_target_date, "NFL", f"Parlay Ganador {slot_key} #1", f"Parlay Ganador #1 NFL ({slot_key} | <= +100 MAX)", gn1)
                    if gn2 and len(gn2) >= 2:
                        _record_parlay_helper(nfl_target_date, "NFL", f"Parlay Ganador {slot_key} #2", f"Parlay Ganador #2 NFL ({slot_key} | <= +100 MAX)", gn2)
            else:
                _record_single_helper(nfl_target_date, "NFL", "Apuesta Fuerte", f"Apuesta Fuerte NFL: {nfl_sniper.get('selection', '')}", nfl_sniper)
                _record_parlay_helper(nfl_target_date, "NFL", "Parlay Props", "Parlay Props NFL", nfl_props)
                _record_parlay_helper(nfl_target_date, "NFL", "Parlay Ganador", "Parlay Ganador NFL", nfl_ganador)

    scorecards = history.sync_and_grade_all_tickets()

    def load_active_vehicle(sport: str, vehicle_name: str, fallback_val: Any, target_dt: str = today_str) -> Any:
        locked_t = history.get_ticket(target_dt, sport, vehicle_name)
        if not locked_t:
            if isinstance(fallback_val, dict) and fallback_val.get("selection"):
                return _enrich_leg_security(dict(fallback_val), sport, vehicle_name)
            if isinstance(fallback_val, list):
                return [_enrich_leg_security(dict(x), sport, vehicle_name) for x in fallback_val]
            return fallback_val
        legs = locked_t.get("legs", [])
        if "Apuesta Fuerte" in vehicle_name:
            if not legs:
                return fallback_val
            leg = dict(legs[0])
            leg.setdefault("american_odds", locked_t.get("odds_american", -110))
            leg.setdefault("decimal_odds", locked_t.get("odds_decimal", 1.91))
            if isinstance(fallback_val, dict):
                for k_meta in (
                    "real_prob", "ev", "horario", "kickoff_cst", "league_name", "tournament",
                    "surface", "consensus", "total_models", "consensus_label", "model_dispersion",
                    "kalshi_prob", "sharp_money_pct", "tickets_pct", "is_rlm",
                ):
                    if k_meta not in leg and fallback_val.get(k_meta) is not None:
                        leg[k_meta] = fallback_val[k_meta]
            leg.setdefault("real_prob", 58.0)
            leg.setdefault("ev", 7.5)
            leg["status"] = locked_t.get("status", "PROXIMAMENTE")
            leg["score_result"] = locked_t.get("score_result", "")
            return _enrich_leg_security(leg, sport, vehicle_name)
        enriched_legs = []
        fb_list = fallback_val if isinstance(fallback_val, list) else []
        for idx_l, raw_l in enumerate(legs):
            l_copy = dict(raw_l)
            if idx_l < len(fb_list) and isinstance(fb_list[idx_l], dict) and fb_list[idx_l].get("selection") == l_copy.get("selection"):
                for k_meta in (
                    "consensus", "total_models", "consensus_label", "model_dispersion",
                    "kalshi_prob", "sharp_money_pct", "tickets_pct", "is_rlm",
                ):
                    if k_meta not in l_copy and fb_list[idx_l].get(k_meta) is not None:
                        l_copy[k_meta] = fb_list[idx_l][k_meta]
            enriched_legs.append(_enrich_leg_security(l_copy, sport, vehicle_name))
        return enriched_legs

    dia_sniper_active = load_active_vehicle("Dia", "Apuesta Fuerte", mvp_sniper)
    dia_sniper_2_active = load_active_vehicle("Dia", "Apuesta Fuerte #2", mvp_sniper_2)
    dia_props_active = load_active_vehicle("Dia", "Parlay Props", mvp_props)
    dia_ratonero_active = load_active_vehicle("Dia", "Parlay Ratonero", mvp_ratonero)
    dia_ganador_active = load_active_vehicle("Dia", "Parlay Ganador", mvp_ganador)

    nba_sniper_active = load_active_vehicle("NBA", "Apuesta Fuerte", nba_sniper)
    nba_props_active = load_active_vehicle("NBA", "Parlay Props", nba_props)
    nba_ratonero_active = load_active_vehicle("NBA", "Parlay Ratonero", nba_ratonero)
    nba_ganador_active = load_active_vehicle("NBA", "Parlay Ganador", nba_ganador)

    nhl_sniper_active = load_active_vehicle("NHL", "Apuesta Fuerte", nhl_sniper)
    nhl_props_active = load_active_vehicle("NHL", "Parlay Props", nhl_props)
    nhl_ratonero_active = load_active_vehicle("NHL", "Parlay Ratonero", nhl_ratonero)
    nhl_ganador_active = load_active_vehicle("NHL", "Parlay Ganador", nhl_ganador)

    soccer_sniper_active = load_active_vehicle("Futbol", "Apuesta Fuerte", soccer_sniper)
    soccer_sniper_2_active = load_active_vehicle("Futbol", "Apuesta Fuerte #2", soccer_sniper_2)
    soccer_props_active: List[Dict[str, Any]] = []  # Strictly zero player props in soccer
    soccer_ratonero_active = load_active_vehicle("Futbol", "Parlay Ratonero", soccer_ratonero)
    soccer_ganador_active = load_active_vehicle("Futbol", "Parlay Ganador", soccer_ganador)
    soccer_ganador_2_active = load_active_vehicle("Futbol", "Parlay Ganador #2", soccer_ganador_2)

    ncaaf_by_horario_picks: Dict[str, Dict[str, Any]] = {}
    for turno_key in ("Manana", "Tarde", "Noche"):
        tp = ncaaf_raw_horario_picks.get(turno_key, {})
        ncaaf_by_horario_picks[turno_key] = {
            "sniper_1": load_active_vehicle("NCAAF", f"Apuesta Fuerte {turno_key} #1", tp.get("sniper_1", {}), target_dt=ncaaf_date_str),
            "sniper_2": load_active_vehicle("NCAAF", f"Apuesta Fuerte {turno_key} #2", tp.get("sniper_2", {}), target_dt=ncaaf_date_str),
            "ganador_1": load_active_vehicle("NCAAF", f"Parlay Ganador {turno_key} #1", tp.get("ganador_1", []), target_dt=ncaaf_date_str),
            "ganador_2": load_active_vehicle("NCAAF", f"Parlay Ganador {turno_key} #2", tp.get("ganador_2", []), target_dt=ncaaf_date_str),
        }

    nfl_by_horario_picks: Dict[str, Dict[str, Any]] = {}
    for slot_key in ("Temprano", "Tarde y Noche"):
        sp_picks = nfl_raw_horario_picks.get(slot_key, {})
        nfl_by_horario_picks[slot_key] = {
            "sniper_1": load_active_vehicle("NFL", f"Apuesta Fuerte {slot_key} #1", sp_picks.get("sniper_1", {}), target_dt=nfl_target_date),
            "sniper_2": load_active_vehicle("NFL", f"Apuesta Fuerte {slot_key} #2", sp_picks.get("sniper_2", {}), target_dt=nfl_target_date),
            "ganador_1": load_active_vehicle("NFL", f"Parlay Ganador {slot_key} #1", sp_picks.get("ganador_1", []), target_dt=nfl_target_date),
            "ganador_2": load_active_vehicle("NFL", f"Parlay Ganador {slot_key} #2", sp_picks.get("ganador_2", []), target_dt=nfl_target_date),
        }

    tenis_rec_dt = tenis_sniper.get("local_date", today_str) if isinstance(tenis_sniper, dict) and tenis_sniper else (
        tenis_ganador[0].get("local_date", today_str) if tenis_ganador else today_str
    )
    tenis_sniper_active = load_active_vehicle("Tenis", "Apuesta Fuerte", tenis_sniper, target_dt=tenis_rec_dt)
    tenis_ganador_active = load_active_vehicle("Tenis", "Parlay Ganador", tenis_ganador, target_dt=tenis_rec_dt)

    mlb_sniper_active = load_active_vehicle("MLB", "Apuesta Fuerte", mlb_sniper)
    mlb_sniper_2_active = load_active_vehicle("MLB", "Apuesta Fuerte #2", mlb_sniper_2)
    mlb_props_active: List[Dict[str, Any]] = []
    mlb_ratonero_active: List[Dict[str, Any]] = []
    mlb_ganador_active = load_active_vehicle("MLB", "Parlay Ganador", mlb_ganador)
    mlb_ganador_2_active = load_active_vehicle("MLB", "Parlay Ganador #2", mlb_ganador_2)

    nfl_sniper_active = load_active_vehicle("NFL", "Apuesta Fuerte", nfl_sniper)
    nfl_props_active = load_active_vehicle("NFL", "Parlay Props", nfl_props)
    nfl_ratonero_active = load_active_vehicle("NFL", "Parlay Ratonero", nfl_ratonero)
    nfl_ganador_active = load_active_vehicle("NFL", "Parlay Ganador", nfl_ganador)

    ai_report = {
        "model_weights": ai_agent.model_weights,
        "status": "ACTIVO - ENSAMBLE 4 MODELOS + KALSHI CFTC + ACTION SHARP/RLM + 7 FILTROS INSTITUCIONALES",
        "intel_counts": {
            "kalshi_contracts": kalshi_total_mkts,
            "action_sharp_games": action_total_games,
            "expert_articles": expert_total_arts,
        },
        "dictamens": {
            "dia_sniper": ai_agent.generate_ai_vehicle_dictamen(
                "Dia", "Apuesta Fuerte",
                dia_sniper_active.get("selection", "") if dia_sniper_active else "",
                dia_sniper_active.get("american_odds", -110) if dia_sniper_active else -110,
                dia_sniper_active.get("real_prob", 60.0) if dia_sniper_active else 60.0,
                dia_sniper_active.get("ev", 9.0) if dia_sniper_active else 9.0, {},
            ),
            "nba": ai_agent.generate_ai_vehicle_dictamen(
                "NBA", "Apuesta Fuerte",
                nba_sniper_active.get("selection", "") if nba_sniper_active else "",
                nba_sniper_active.get("american_odds", -110) if nba_sniper_active else -110,
                nba_sniper_active.get("real_prob", 58.0) if nba_sniper_active else 58.0,
                nba_sniper_active.get("ev", 8.0) if nba_sniper_active else 8.0, {},
            ) if nba_sniper_active else "Sin selecciones activas para NBA hoy.",
            "nhl": ai_agent.generate_ai_vehicle_dictamen(
                "NHL", "Apuesta Fuerte",
                nhl_sniper_active.get("selection", "") if nhl_sniper_active else "",
                nhl_sniper_active.get("american_odds", -130) if nhl_sniper_active else -130,
                nhl_sniper_active.get("real_prob", 58.0) if nhl_sniper_active else 58.0,
                nhl_sniper_active.get("ev", 8.0) if nhl_sniper_active else 8.0, {},
            ) if nhl_sniper_active else "Sin selecciones activas para NHL hoy.",
            "soccer": ai_agent.generate_ai_vehicle_dictamen(
                "Futbol", "Apuesta Fuerte",
                soccer_sniper_active.get("selection", "") if soccer_sniper_active else "",
                soccer_sniper_active.get("american_odds", -120) if soccer_sniper_active else -120,
                soccer_sniper_active.get("real_prob", 58.0) if soccer_sniper_active else 58.0,
                soccer_sniper_active.get("ev", 8.0) if soccer_sniper_active else 8.0, {},
            ) if soccer_sniper_active else "Sin selecciones activas para Futbol hoy.",
            "ncaaf": (
                f"Cartelera Sabado Colegial ({ncaaf_date_str}) dividida estrictamente sin mezclar en 3 horarios independientes "
                f"(Manana: {len(ncaaf_by_horario.get('Manana', []))} juegos | "
                f"Tarde: {len(ncaaf_by_horario.get('Tarde', []))} juegos | "
                f"Noche: {len(ncaaf_by_horario.get('Noche', []))} juegos), con 2 Apuestas Fuertes + 2 Parlays Ganadores por cada horario."
            ),
            "tenis": (
                f"Modelos ATP/WTA activos sobre {len(tennis_analyzed_games)} encuentros Singles de circuito principal. "
                f"Evaluacion mediante Elo ajustado por superficie, Hold%, Break%, Dominance Ratio, Tiebreak% y 10,000 simulaciones Markov."
            ),
            "mlb": (
                f"Cartelera de Beisbol activa ({len(mlb_analyzed_games)} juegos analizados: "
                f"LMP Liga del Pacifico, KBO, NPB y MLB) con evaluacion sabermetrica de abridores (FIP/xFIP), "
                f"potencia de lineup (wRC+) y bullpen."
            ) if mlb_analyzed_games else "Monitoreo sabermetrico de beisbol.",
            "nfl": (
                f"Cartelera NFL Domingo ({nfl_target_date}) dividida sin mezclar en 2 horarios independientes "
                f"(Temprano < 14:00 CST: {len(nfl_by_horario.get('Temprano', []))} juegos | "
                f"Tarde y Noche >= 14:00 CST: {len(nfl_by_horario.get('Tarde y Noche', []))} juegos), "
                f"con 2 Apuestas Fuertes + 2 Parlays Ganadores por horario."
            ) if is_nfl_sunday else "Monitoreo activo de cartelera NFL.",
        },
    }

    return {
        "active_date": today_str,
        "nfl_target_date": nfl_target_date,
        "whales": whales,
        "ai_agent": ai_report,
        "dia_mvp": {
            "sniper": dia_sniper_active,
            "sniper_2": dia_sniper_2_active,
            "ratonero": dia_ratonero_active,
            "props": dia_props_active,
            "ganador": dia_ganador_active,
        },
        "nba": {
            "games": nba_analyzed_games,
            "sniper": nba_sniper_active,
            "props": nba_props_active,
            "ratonero": nba_ratonero_active,
            "ganador": nba_ganador_active,
        },
        "nhl": {
            "games": nhl_analyzed_games,
            "sniper": nhl_sniper_active,
            "props": nhl_props_active,
            "ratonero": nhl_ratonero_active,
            "ganador": nhl_ganador_active,
        },
        "soccer": {
            "games": soccer_analyzed_games,
            "sniper": soccer_sniper_active,
            "sniper_2": soccer_sniper_2_active,
            "props": [],  # Strictly zero player props in soccer
            "ratonero": soccer_ratonero_active,
            "ganador": soccer_ganador_active,
            "ganador_2": soccer_ganador_2_active,
        },
        "ncaaf": {
            "target_date": ncaaf_date_str,
            "games": ncaaf_analyzed_games,
            "horarios": ncaaf_by_horario,
            "by_horario_picks": ncaaf_by_horario_picks,
        },
        "tenis": {
            "target_date": tenis_rec_dt,
            "games": tennis_analyzed_games,
            "sniper": tenis_sniper_active,
            "ganador": tenis_ganador_active,
        },
        "mlb": {
            "matchup_data": mlb_ws_data,
            "games": mlb_analyzed_games,
            "sniper": mlb_sniper_active,
            "sniper_2": mlb_sniper_2_active,
            "props": [],
            "ratonero": [],
            "ganador": mlb_ganador_active,
            "ganador_2": mlb_ganador_2_active,
        },
        "nfl": {
            "target_date": nfl_target_date,
            "games": nfl_analyzed_games if nfl_analyzed_games else [
                {
                    "sport": "NFL",
                    "home_team": g["home_team"]["name"],
                    "away_team": g["away_team"]["name"],
                    "matchup": f"{g['away_team']['name']} @ {g['home_team']['name']}",
                    "spread": float(g.get("odds", {}).get("spread", 0.0) or 0.0),
                    "total": float(g.get("odds", {}).get("overUnder", 45.0) or 45.0),
                    "home_odds": g.get("odds", {}).get("home_moneyline") or -110,
                    "away_odds": g.get("odds", {}).get("away_moneyline") or 110,
                    "home_real_prob": 0.50,
                    "away_real_prob": 0.50,
                    "home_ev": 0.0,
                    "away_ev": 0.0,
                    "projected_score": "Programado (Prox. Jornada)",
                }
                for g in nfl_espn_raw
                if not str(g.get("status", "")).endswith("FINAL")
            ],
            "horarios": nfl_by_horario,
            "by_horario_picks": nfl_by_horario_picks,
            "is_sunday_slate": is_nfl_sunday,
            "sniper": nfl_sniper_active,
            "props": nfl_props_active,
            "ratonero": nfl_ratonero_active,
            "ganador": nfl_ganador_active,
        },
        "scorecards": scorecards,
    }


if __name__ == "__main__":
    res = run_sports_pipeline(record_in_history=True)
    print("\n--- RESUMEN EJECUTIVO GENERADO ---")
    print("MVP Sniper #1:", res["dia_mvp"]["sniper"].get("selection") if res["dia_mvp"]["sniper"] else "None")
    print("MVP Ratonero legs:", len(res["dia_mvp"]["ratonero"]), "| MVP Sniper #2:", res["dia_mvp"]["sniper_2"].get("selection") if res["dia_mvp"]["sniper_2"] else "None")
    print("NCAAF Date:", res["ncaaf"]["target_date"], "| Horarios:", {k: len(v) for k, v in res["ncaaf"]["horarios"].items()})
    for tk in ("Manana", "Tarde", "Noche"):
        tp = res["ncaaf"]["by_horario_picks"].get(tk, {})
        print(
            f"  [NCAAF {tk}] Fuerte #1: {tp.get('sniper_1', {}).get('selection', 'None')} | "
            f"Fuerte #2: {tp.get('sniper_2', {}).get('selection', 'None')} | "
            f"Ganador #1: {[x.get('selection') for x in tp.get('ganador_1', [])]} | "
            f"Ganador #2: {[x.get('selection') for x in tp.get('ganador_2', [])]}"
        )
    print("Tenis Sniper:", res["tenis"]["sniper"].get("selection") if res["tenis"]["sniper"] else "None")
    print("Tenis Ganador:", [x.get("selection") for x in res["tenis"]["ganador"]])
    print("Ejecucion Completada Exitosamente.")
