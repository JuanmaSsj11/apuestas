"""
History and Bet Tracking Engine with Ticket and Multi-Leg Parlay Support
Tracks:
- Singles (Apuesta Fuerte)
- Parlays (Parlay Ratonero, Parlay Props, Parlay Ganador) with individual leg evaluation
- Sport-specific performance (NBA, NHL, Futbol, NCAAF, Tenis, MLB, NFL)
- Vehicle-specific performance (Apuesta Fuerte, Props, Ratonero, Ganador)
- Global/Day MVP performance
Enforces strict dual-team + exact-date + sport-isolated matching so unplayed or
same-mascot games NEVER collide across dates or opponents.
Strictly zero emojis.
"""
import os
import json
import logging
import requests
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HistoryTracker")

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bet_history.json")
CST_TZ = timezone(timedelta(hours=-6))

SOCCER_SLUGS = [
    "eng.1", "esp.1", "ita.1", "ger.1", "fra.1",
    "eng.fa", "eng.league_cup", "esp.copa_del_rey", "ita.coppa_italia",
    "ger.dfb_pokal", "fra.coupe_de_france",
    "uefa.champions", "uefa.europa", "uefa.europa.conf", "mex.1"
]

NCAAF_GENERIC_MASCOTS = {
    "wildcats", "tigers", "bulls", "bulldogs", "eagles", "bears", "panthers",
    "cougars", "trojans", "aggies", "rebels", "hawks", "lions", "knights",
    "spartans", "falcons", "huskies", "broncos", "cardinals", "cowboys",
    "wolfpack", "owls", "seminoles", "hurricanes", "cavaliers", "mountaineers",
    "bobcats", "gamecocks", "volunteers", "commodores", "razorbacks", "sooners",
    "longhorns", "jayhawks", "cyclones", "badgers", "wolverines", "buckeyes",
    "hoosiers", "boilermakers", "terrapins", "gophers", "cornhuskers", "ducks",
    "beavers", "bruins", "buffaloes", "utes", "aztecs", "mustangs", "blazers",
    "pirates", "monarchs", "flames", "golden", "fighting", "crimson", "scarlet",
    "red", "blue", "green", "mean", "black", "orange", "wave", "tide", "irish",
    "illini", "nittany", "demon", "deacons", "yellow", "jackets", "horned",
    "frogs", "redhawks", "chippewas", "rockets", "zips", "bulls", "thundering",
    "herd", "hilltoppers", " Chanticleers", "mountaineers", "warriors", "rainbow"
}

SOCCER_GENERIC_WORDS = {
    "fc", "cf", "sc", "ac", "as", "cd", "ud", "rc", "rcd", "sv", "vfb", "vfl",
    "fsv", "rb", "club", "real", "atletico", "athletic", "sporting", "united",
    "city", "town", "wanderers", "rovers", "albion", "hotspur", "villa",
    "saint", "st", "san", "de", "del", "la", "las", "los", "el", "unam", "uanl",
    "calcio", "1909", "1899", "04", "05", "98"
}

TEAM_ALIASES = {
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton",
    "inter milan": "internazionale",
    "atletico madrid": "atletico de madrid",
    "gladbach": "borussia monchengladbach",
    "bayern": "bayern munich",
    "leverkusen": "bayer leverkusen",
    "dortmund": "borussia dortmund",
    "stade rennais": "rennes",
    "rennes": "stade rennais",
    "uconn": "connecticut",
    "ole miss": "mississippi",
    "usc": "southern california",
    "ucf": "central florida",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "lsu": "louisiana state",
    "utsa": "tx san antonio",
    "utep": "texas el paso",
    "unlv": "nevada las vegas",
    "fiu": "florida international",
    "fau": "florida atlantic",
    "uab": "alabama birmingham",
    "usf": "south florida",
    "nc state": "north carolina state",
    "pitt": "pittsburgh",
    "cal": "california",
    "umass": "massachusetts",
}

NCAAF_DISAMBIGUATION_SUFFIXES = (
    "state", "tech", "southern", "northern", "eastern", "western", "central",
    "atlantic", "international", "christian", "methodist", "a m", "oh", "fl"
)


def _fold_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_phrase(text: str) -> str:
    if not text:
        return ""
    s = _fold_accents(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _has_word_phrase(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return bool(re.search(pattern, haystack))


def _extract_ncaaf_school(team_norm: str) -> str:
    """
    Strips trailing generic NCAAF mascot words to isolate the school/university name
    (e.g. 'oregon ducks' -> 'oregon', 'oregon state beavers' -> 'oregon state',
    'florida state seminoles' -> 'florida state', 'south florida bulls' -> 'south florida').
    """
    tokens = team_norm.split()
    while len(tokens) > 1 and tokens[-1] in NCAAF_GENERIC_MASCOTS:
        tokens.pop()
    return " ".join(tokens)


def team_matches_text(team_name: str, text: str, sport: str = "") -> bool:
    """
    Strictly checks if team_name/player_name appears in text as a whole-word identity.
    Prevents substring collisions ('villa' inside 'villarreal'), mascot collisions
    ('Wildcats'/'Tigers'/'Bulls'), and school suffix collisions ('Oregon' vs 'Oregon State').
    """
    t_norm = _norm_phrase(team_name)
    txt_norm = _norm_phrase(text)
    if not t_norm or not txt_norm:
        return False

    sp = (sport or "").upper()

    # 1. Exact full display name match with word boundaries
    if _has_word_phrase(t_norm, txt_norm):
        if sp == "NCAAF":
            # Ensure 'oregon' doesn't match 'oregon state' if team_name was just 'Oregon'
            for suf in ("state", "tech", "oh"):
                if not t_norm.endswith(f" {suf}") and _has_word_phrase(f"{t_norm} {suf}", txt_norm):
                    return False
        return True

    # 2. Sport-specific strict identity rules
    if sp == "NCAAF":
        school = _extract_ncaaf_school(t_norm)
        if not school or len(school) < 2:
            return False
        if _has_word_phrase(school, txt_norm):
            # Guard against 'oregon' matching 'oregon state', 'miami' matching 'miami oh', 'michigan' matching 'michigan state'
            for suf in NCAAF_DISAMBIGUATION_SUFFIXES:
                if suf not in school.split():
                    if _has_word_phrase(f"{school} {suf}", txt_norm) or _has_word_phrase(f"{suf} {school}", txt_norm):
                        return False
            return True
        return False

    if sp == "TENIS":
        tokens = t_norm.split()
        if len(tokens) >= 2:
            last_name = tokens[-1]
            first_name = tokens[0]
            # Require full name OR (unique surname >= 5 chars AND first initial/name not contradicted)
            if len(last_name) >= 5 and _has_word_phrase(last_name, txt_norm):
                if _has_word_phrase(first_name, txt_norm) or not any(
                    other_first != first_name and _has_word_phrase(f"{other_first} {last_name}", txt_norm)
                    for other_first in txt_norm.split()
                ):
                    return True
        return False

    if sp in ("FUTBOL", "SOCCER"):
        # Strip generic prefixes/suffixes like FC, CF, Real, Club, SV, etc., keeping core club identity
        sig_tokens = [w for w in t_norm.split() if w not in SOCCER_GENERIC_WORDS]
        # Special two-word clubs where generic words are part of identity (Aston Villa, Manchester City, Manchester United, Real Madrid, Real Sociedad, Real Betis, Atletico Madrid, Athletic Club, Cruz Azul)
        two_word_identities = [
            "aston villa", "manchester city", "manchester united", "real madrid",
            "real sociedad", "real betis", "real valladolid", "atletico madrid",
            "atletico de madrid", "atletico de san luis", "atletico san luis",
            "athletic club", "cruz azul", "tigres uanl", "pumas unam", "fc juarez",
            "rayo vallecano", "celta vigo", "las palmas", "ac milan", "as roma",
            "inter milan", "union berlin", "werder bremen", "st pauli", "holstein kiel",
            "paris saint germain", "as monaco", "le havre", "stade rennais",
            "crystal palace", "west ham", "leicester city", "ipswich town",
            "nottingham forest", "newcastle united"
        ]
        for tw in two_word_identities:
            if _has_word_phrase(tw, t_norm):
                if _has_word_phrase(tw, txt_norm):
                    return True
                # Alias check (e.g. 'stade rennais' <-> 'rennes', 'pumas unam' <-> 'pumas', 'fc juarez' <-> 'juarez')
                if tw == "stade rennais" and _has_word_phrase("rennes", txt_norm):
                    return True
                if tw == "pumas unam" and _has_word_phrase("pumas", txt_norm):
                    return True
                if tw == "tigres uanl" and _has_word_phrase("tigres", txt_norm):
                    return True
                if tw == "fc juarez" and _has_word_phrase("juarez", txt_norm):
                    return True
                if tw in ("atletico de san luis", "atletico san luis") and _has_word_phrase("san luis", txt_norm):
                    return True
                if tw == "le havre" and _has_word_phrase("le havre", txt_norm):
                    return True
                return False

        core_phrase = " ".join(sig_tokens)
        if core_phrase and len(core_phrase) >= 4 and _has_word_phrase(core_phrase, txt_norm):
            return True
        if len(sig_tokens) == 1 and len(sig_tokens[0]) >= 4 and _has_word_phrase(sig_tokens[0], txt_norm):
            return True
        for k_alias, v_alias in TEAM_ALIASES.items():
            if _has_word_phrase(k_alias, t_norm) and _has_word_phrase(v_alias, txt_norm):
                return True
            if _has_word_phrase(v_alias, t_norm) and _has_word_phrase(k_alias, txt_norm):
                return True
        return False

    # Pro US leagues (NBA, NHL, NFL, MLB): sport is already strictly isolated!
    # Check multi-word franchise nicknames first ('trail blazers', 'golden knights', 'maple leafs', 'blue jackets', 'red wings', 'white sox', 'red sox', 'blue jays')
    multi_nicks = [
        "trail blazers", "golden knights", "maple leafs", "blue jackets",
        "red wings", "white sox", "red sox", "blue jays", "timberwolves"
    ]
    for mn in multi_nicks:
        if _has_word_phrase(mn, t_norm):
            return _has_word_phrase(mn, txt_norm)

    tokens = t_norm.split()
    if tokens:
        nick = tokens[-1]
        if len(nick) >= 4 and _has_word_phrase(nick, txt_norm):
            return True
    return False


def both_teams_match_game_text(home_team: str, away_team: str, game_text: str, sport: str) -> bool:
    """
    Requires BOTH participants of a live/final ESPN event to match the leg's game string.
    Eliminates 100% of single-team cross-opponent collisions.
    """
    if not home_team or not away_team or not game_text:
        return False
    if _norm_phrase(home_team) == _norm_phrase(away_team):
        return False
    return (
        team_matches_text(home_team, game_text, sport)
        and team_matches_text(away_team, game_text, sport)
    )


def _iso_to_cst_date(iso_str: str, fallback_ymd: str) -> str:
    if iso_str:
        try:
            dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt_utc.astimezone(CST_TZ).strftime("%Y-%m-%d")
        except Exception:
            pass
    if fallback_ymd and len(fallback_ymd) == 8 and fallback_ymd.isdigit():
        return f"{fallback_ymd[:4]}-{fallback_ymd[4:6]}-{fallback_ymd[6:8]}"
    return datetime.now(CST_TZ).strftime("%Y-%m-%d")


class HistoryTracker:
    def __init__(self, filepath: str = HISTORY_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self._boxscore_cache: Dict[str, Dict[str, Any]] = {}
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if "tickets" in content:
                        return content
                    return {"tickets": []}
            except Exception as e:
                logger.error(f"Error loading history file: {e}")
        return {"tickets": []}

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving history file: {e}")

    @staticmethod
    def _build_ticket_id(date_str: str, sport: str, vehicle: str) -> str:
        return (
            f"{date_str}_{sport}_{vehicle}"
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .lower()
        )

    def record_ticket(
        self,
        date_str: str,
        sport: str,
        vehicle: str,
        title: str,
        odds_american: int,
        odds_decimal: float,
        legs: List[Dict[str, Any]],
        is_parlay: bool = True
    ):
        """
        Records a betting ticket (Single or Parlay) with one or multiple legs.
        Ensures strict uniqueness by date, sport, and vehicle to prevent duplicates.
        """
        ticket_id = self._build_ticket_id(date_str, sport, vehicle)
        existing = next((t for t in self.data["tickets"] if t.get("id") == ticket_id), None)
        if existing:
            logger.info(f"Ticket {ticket_id} is already locked in history. Preserving official selections.")
            return

        new_ticket = {
            "id": ticket_id,
            "date": date_str,
            "sport": sport,
            "vehicle": vehicle,
            "title": title,
            "is_parlay": is_parlay,
            "odds_american": odds_american,
            "odds_decimal": odds_decimal,
            "status": "PROXIMAMENTE",
            "score_result": "Pendiente de inicio",
            "units_pnl": 0.0,
            "settled_at": None,
            "legs": legs
        }
        self.data["tickets"].append(new_ticket)
        self._save()

    def get_ticket(self, date_str: str, sport: str, vehicle: str) -> Optional[Dict[str, Any]]:
        ticket_id = self._build_ticket_id(date_str, sport, vehicle)
        return next((t for t in self.data["tickets"] if t.get("id") == ticket_id), None)

    def get_tickets_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        return [t for t in self.data.get("tickets", []) if t.get("date") == date_str]

    def fetch_live_game_states(self, extra_dates: Optional[Set[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Concurrently fetches live game scores, clocks, and states from ESPN across:
        NBA, NHL, NFL, NCAAF (FBS), Tenis (ATP & WTA Singles), MLB, and 15 Futbol competitions.
        Keys every game uniquely by '{sport}:{game_date}:{event_id}' so multi-date schedules never overwrite.
        """
        now_cst = datetime.now(CST_TZ)
        today_ymd = now_cst.strftime("%Y%m%d")
        tomorrow_ymd = (now_cst + timedelta(days=1)).strftime("%Y%m%d")
        active_dates: Set[str] = {today_ymd, tomorrow_ymd}

        if extra_dates:
            for d_raw in extra_dates:
                d_clean = str(d_raw).replace("-", "")
                if len(d_clean) == 8 and d_clean.isdigit():
                    active_dates.add(d_clean)

        for t in self.data.get("tickets", []):
            if t.get("status") not in ("GANADA", "PERDIDA") or not t.get("settled_at"):
                for d_cand in [t.get("date")] + [l.get("local_date") or l.get("date") for l in t.get("legs", [])]:
                    if d_cand:
                        d_clean = str(d_cand).replace("-", "")
                        if len(d_clean) == 8 and d_clean.isdigit():
                            active_dates.add(d_clean)

        # Build concurrent task list: (kind, sport, league_slug, d_ymd, url)
        tasks = []
        for d_ymd in sorted(active_dates):
            tasks.append(("team", "NBA", "nba", d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_ymd}"))
            tasks.append(("team", "NHL", "nhl", d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={d_ymd}"))
            tasks.append(("team", "MLB", "mlb", d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={d_ymd}"))
            tasks.append(("team", "NFL", "nfl", d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={d_ymd}"))
            tasks.append(("team", "NCAAF", "college-football", d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={d_ymd}&groups=80&limit=100"))
            for lg in SOCCER_SLUGS:
                tasks.append(("team", "Futbol", lg, d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={d_ymd}"))
            for tour in ("atp", "wta"):
                tasks.append(("tennis", "Tenis", tour, d_ymd, f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={d_ymd}"))

        live_games: Dict[str, Dict[str, Any]] = {}

        def _fetch_one(task_tuple: Tuple[str, str, str, str, str]) -> List[Tuple[str, Dict[str, Any]]]:
            kind, sport, lg_slug, d_ymd, url = task_tuple
            out: List[Tuple[str, Dict[str, Any]]] = []
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                if resp.status_code != 200:
                    return out
                payload = resp.json()
                if kind == "team":
                    for ev in payload.get("events", []):
                        ev_id = str(ev.get("id") or "")
                        st_type = ev.get("status", {}).get("type", {})
                        state = st_type.get("state", "pre")
                        completed = bool(st_type.get("completed", False))
                        detail = st_type.get("detail", "Programado")
                        comps = ev.get("competitions", [{}])[0].get("competitors", [])
                        if len(comps) < 2:
                            continue
                        home = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
                        away = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
                        h_name = home.get("team", {}).get("displayName", "")
                        a_name = away.get("team", {}).get("displayName", "")
                        if not h_name or not a_name:
                            continue
                        h_score = int(float(home.get("score") or 0))
                        a_score = int(float(away.get("score") or 0))
                        g_date = _iso_to_cst_date(ev.get("date", ""), d_ymd)
                        if sport == "Futbol":
                            score_str = f"{h_name} {h_score} - {a_score} {a_name}"
                        else:
                            score_str = f"{a_name} {a_score} - {h_name} {h_score}"
                        g_key = f"{sport}:{g_date}:{ev_id}".lower()
                        out.append((g_key, {
                            "event_id": ev_id,
                            "sport": sport,
                            "league_slug": lg_slug,
                            "game_date": g_date,
                            "query_ymd": d_ymd,
                            "home_team": h_name,
                            "away_team": a_name,
                            "home_score": h_score,
                            "away_score": a_score,
                            "total": h_score + a_score,
                            "state": state,
                            "completed": completed,
                            "detail": detail,
                            "score_str": score_str,
                            "winner": h_name if h_score > a_score else (a_name if a_score > h_score else "tie")
                        }))
                else:
                    for tev in payload.get("events", []):
                        for grp in tev.get("groupings", []):
                            slug = grp.get("grouping", {}).get("slug", "")
                            if "singles" not in slug.lower():
                                continue
                            for mc in grp.get("competitions", []):
                                comps = mc.get("competitors", [])
                                if len(comps) < 2:
                                    continue
                                p1_c, p2_c = comps[0], comps[1]
                                p1_name = p1_c.get("athlete", {}).get("displayName", "")
                                p2_name = p2_c.get("athlete", {}).get("displayName", "")
                                if not p1_name or not p2_name or "TBD" in (p1_name, p2_name):
                                    continue
                                mc_id = str(mc.get("id") or "")
                                st_type = mc.get("status", {}).get("type", {})
                                state = st_type.get("state", "pre")
                                completed = bool(st_type.get("completed", False))
                                detail = st_type.get("detail", "Programado")
                                g_date = _iso_to_cst_date(mc.get("date", ""), d_ymd)

                                ls1 = p1_c.get("linescores", []) or []
                                ls2 = p2_c.get("linescores", []) or []
                                p1_sets = sum(1 for s in ls1 if s.get("winner"))
                                p2_sets = sum(1 for s in ls2 if s.get("winner"))
                                if p1_sets == 0 and p2_sets == 0 and ls1 and ls2:
                                    for idx_s in range(min(len(ls1), len(ls2))):
                                        v1 = int(float(ls1[idx_s].get("value", 0)))
                                        v2 = int(float(ls2[idx_s].get("value", 0)))
                                        if v1 > v2 and v1 >= 6:
                                            p1_sets += 1
                                        elif v2 > v1 and v2 >= 6:
                                            p2_sets += 1

                                p1_games = sum(int(float(s.get("value", 0))) for s in ls1)
                                p2_games = sum(int(float(s.get("value", 0))) for s in ls2)
                                set_pairs = [
                                    f"{int(float(ls1[k].get('value', 0)))}-{int(float(ls2[k].get('value', 0)))}"
                                    for k in range(min(len(ls1), len(ls2)))
                                ]
                                sets_detail = f" ({', '.join(set_pairs)})" if set_pairs else ""
                                winner_name = "tie"
                                if p1_c.get("winner") or p1_sets > p2_sets:
                                    winner_name = p1_name
                                elif p2_c.get("winner") or p2_sets > p1_sets:
                                    winner_name = p2_name

                                g_key = f"tenis:{g_date}:{mc_id}".lower()
                                out.append((g_key, {
                                    "event_id": mc_id,
                                    "sport": "Tenis",
                                    "league_slug": lg_slug,
                                    "game_date": g_date,
                                    "query_ymd": d_ymd,
                                    "home_team": p1_name,
                                    "away_team": p2_name,
                                    "home_score": p1_sets,
                                    "away_score": p2_sets,
                                    "home_games": p1_games,
                                    "away_games": p2_games,
                                    "total": p1_games + p2_games,
                                    "state": state,
                                    "completed": completed,
                                    "detail": detail,
                                    "score_str": f"{p1_name} {p1_sets} - {p2_sets} {p2_name}{sets_detail}",
                                    "winner": winner_name
                                }))
            except Exception as e:
                logger.debug(f"Live state fetch warning ({sport} {d_ymd}): {e}")
            return out

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_fetch_one, t) for t in tasks]
            for fut in as_completed(futures):
                for gk, gv in fut.result():
                    live_games[gk] = gv

        return live_games

    def fetch_event_summary(self, sport: str, event_id: str, league_slug: str = "") -> Dict[str, Any]:
        """
        Fetches and caches the official ESPN event summary/boxscore for player props evaluation.
        Supports all 15 soccer competitions via league_slug with multi-league fallback.
        """
        if not event_id:
            return {}
        cache_key = f"{sport}:{event_id}"
        if cache_key in self._boxscore_cache:
            return self._boxscore_cache[cache_key]

        sport_clean = (sport or "").upper()
        paths_to_try = []
        if sport_clean == "NBA":
            paths_to_try = ["basketball/nba"]
        elif sport_clean == "NHL":
            paths_to_try = ["hockey/nhl"]
        elif sport_clean == "NFL":
            paths_to_try = ["football/nfl"]
        elif sport_clean == "NCAAF":
            paths_to_try = ["football/college-football"]
        elif sport_clean == "MLB":
            paths_to_try = ["baseball/mlb"]
        else:
            if league_slug and league_slug not in ("nba", "nhl", "nfl", "mlb"):
                paths_to_try.append(f"soccer/{league_slug}")
            for sl in ("mex.1", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions", "uefa.europa"):
                p_cand = f"soccer/{sl}"
                if p_cand not in paths_to_try:
                    paths_to_try.append(p_cand)

        for path in paths_to_try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/summary?event={event_id}"
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("boxscore") or data.get("rosters") or data.get("header"):
                        self._boxscore_cache[cache_key] = data
                        return data
            except Exception as e:
                logger.debug(f"Summary fetch warning ({path} {event_id}): {e}")

        return {}

    @staticmethod
    def _athlete_matches(p_target: str, ath_display: str) -> bool:
        p_norm = _norm_phrase(p_target)
        a_norm = _norm_phrase(ath_display)
        if not p_norm or not a_norm:
            return False
        if p_norm == a_norm or _has_word_phrase(p_norm, a_norm) or _has_word_phrase(a_norm, p_norm):
            return True
        p_toks = p_norm.split()
        a_toks = a_norm.split()
        if len(p_toks) >= 2 and len(a_toks) >= 2:
            # Same surname AND matching first initial (prevents Davion Mitchell vs Donovan Mitchell, Brady vs Matthew Tkachuk)
            if p_toks[-1] == a_toks[-1] and len(p_toks[-1]) >= 4 and p_toks[0][0] == a_toks[0][0]:
                return True
        return False

    def evaluate_player_prop(
        self,
        leg: Dict[str, Any],
        matched_game: Dict[str, Any],
        is_officially_final: bool,
        is_live_in_play: bool
    ) -> Tuple[str, str]:
        """
        Extracts real athlete stats from ESPN boxscore/rosters and evaluates player props across
        NBA, NHL, NFL, MLB, and Futbol with strict athlete identity matching.
        """
        sp = (matched_game.get("sport") or leg.get("sport") or "NBA").upper()
        ev_id = matched_game.get("event_id", "")
        lg_slug = matched_game.get("league_slug", "")
        p_name = (leg.get("player") or "").strip()
        sel = (leg.get("selection") or "").strip()
        sel_lower = _fold_accents(sel).lower()

        if not p_name and ":" in sel:
            p_name = sel.split(":")[0].strip()

        summary = self.fetch_event_summary(sp, ev_id, league_slug=lg_slug)
        if not summary:
            if is_officially_final:
                return ("PERDIDA", f"Final: {matched_game.get('score_str', 'Final')} (Sin registro en boxscore)")
            return ("EN JUEGO" if is_live_in_play else "PROXIMAMENTE", "Programado")

        box = summary.get("boxscore", {})
        players_data = box.get("players", [])
        found = False
        stat_val = 0.0
        stat_unit = "PTS"
        target_val = 1.0

        if sp == "NBA":
            pts_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*puntos", sel_lower)
            reb_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*rebotes", sel_lower)
            ast_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*asistencias", sel_lower)

            col_label, stat_unit = "PTS", "PTS"
            if pts_match:
                target_val, col_label, stat_unit = float(pts_match[0]), "PTS", "PTS"
            elif reb_match:
                target_val, col_label, stat_unit = float(reb_match[0]), "REB", "REB"
            elif ast_match:
                target_val, col_label, stat_unit = float(ast_match[0]), "AST", "AST"
            else:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", sel_lower)
                target_val = float(nums[0]) if nums else 20.0

            for t in players_data:
                for s in t.get("statistics", []):
                    labels = s.get("labels", [])
                    if col_label in labels:
                        idx = labels.index(col_label)
                        for ath in s.get("athletes", []):
                            ath_name = ath.get("athlete", {}).get("displayName", "")
                            if self._athlete_matches(p_name, ath_name):
                                stats_arr = ath.get("stats", [])
                                if idx < len(stats_arr):
                                    try:
                                        stat_val = float(stats_arr[idx])
                                        found = True
                                        break
                                    except (ValueError, TypeError):
                                        pass
                if found:
                    break

        elif sp == "NHL":
            shots_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:remates|tiros|disparos)", sel_lower)
            goals_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*goles?", sel_lower)
            col_label, stat_unit = "S", "Remates"
            if shots_match:
                target_val, col_label, stat_unit = float(shots_match[0]), "S", "Remates"
            elif goals_match:
                target_val, col_label, stat_unit = float(goals_match[0]), "G", "Goles"
            else:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", sel_lower)
                target_val = float(nums[0]) if nums else 2.5

            for t in players_data:
                for s in t.get("statistics", []):
                    labels = s.get("labels", [])
                    # ESPN NHL uses 'S' or 'SOG' for shots on goal
                    eff_col = col_label if col_label in labels else ("SOG" if col_label == "S" and "SOG" in labels else None)
                    if eff_col:
                        idx = labels.index(eff_col)
                        for ath in s.get("athletes", []):
                            ath_name = ath.get("athlete", {}).get("displayName", "")
                            if self._athlete_matches(p_name, ath_name):
                                stats_arr = ath.get("stats", [])
                                if idx < len(stats_arr):
                                    try:
                                        stat_val = float(stats_arr[idx])
                                        found = True
                                        break
                                    except (ValueError, TypeError):
                                        pass
                if found:
                    break

        elif sp == "NFL":
            rush_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*yardas\s*terrestres", sel_lower)
            pass_td_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*pases?\s*de\s*td", sel_lower)
            rec_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*yardas\s*(?:por\s*recepcion|de\s*recepcion|recibidas)", sel_lower)

            cat_name, col_label, stat_unit = "rushing", "YDS", "Yds Terrestres"
            if rush_match:
                target_val, cat_name, col_label, stat_unit = float(rush_match[0]), "rushing", "YDS", "Yds Terrestres"
            elif pass_td_match:
                target_val, cat_name, col_label, stat_unit = float(pass_td_match[0]), "passing", "TD", "Pases de TD"
            elif rec_match:
                target_val, cat_name, col_label, stat_unit = float(rec_match[0]), "receiving", "YDS", "Yds Recepcion"
            else:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", sel_lower)
                target_val = float(nums[0]) if nums else 40.0

            for t in players_data:
                for s in t.get("statistics", []):
                    if s.get("name") == cat_name:
                        labels = s.get("labels", [])
                        if col_label in labels:
                            idx = labels.index(col_label)
                            for ath in s.get("athletes", []):
                                ath_name = ath.get("athlete", {}).get("displayName", "")
                                if self._athlete_matches(p_name, ath_name):
                                    stats_arr = ath.get("stats", [])
                                    if idx < len(stats_arr):
                                        try:
                                            stat_val = float(stats_arr[idx])
                                            found = True
                                            break
                                        except (ValueError, TypeError):
                                            pass
                if found:
                    break

        elif sp in ("FUTBOL", "SOCCER"):
            sot_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:tiros?|remates?|disparos?)\s*a\s*(?:puerta|porteria|arco)", sel_lower)
            goal_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*goles?", sel_lower)
            is_goal_prop = bool(goal_match) or "anotador" in sel_lower or "anytime" in sel_lower
            target_val = float(sot_match[0]) if sot_match else (float(goal_match[0]) if goal_match else 1.0)
            stat_unit = "Goles" if is_goal_prop else "Tiros a Puerta"
            want_keys = ("totalGoals", "G") if is_goal_prop else ("shotsOnTarget", "ST", "SOT")

            # ESPN Soccer summary stores athlete stats inside summary['rosters'][i]['roster'][j]['stats']
            for r_team in summary.get("rosters", []):
                for entry in r_team.get("roster", []):
                    ath_name = entry.get("athlete", {}).get("displayName", "")
                    if self._athlete_matches(p_name, ath_name):
                        played = bool(entry.get("starter") or entry.get("subbedIn") or entry.get("played"))
                        stats_list = entry.get("stats", [])
                        if stats_list:
                            found = True
                        elif played:
                            found = True
                        for st_item in stats_list:
                            if st_item.get("name") in want_keys or st_item.get("abbreviation") in want_keys:
                                try:
                                    stat_val = float(st_item.get("value", 0.0))
                                    found = True
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if found:
                            break
                if found:
                    break

            # Secondary check in boxscore['players'] if rosters empty
            if not found:
                for t in players_data:
                    for s in t.get("statistics", []):
                        labels = s.get("labels", [])
                        col_cands = [c for c in want_keys if c in labels]
                        if col_cands:
                            idx = labels.index(col_cands[0])
                            for ath in s.get("athletes", []):
                                ath_name = ath.get("athlete", {}).get("displayName", "")
                                if self._athlete_matches(p_name, ath_name):
                                    stats_arr = ath.get("stats", [])
                                    if idx < len(stats_arr):
                                        try:
                                            stat_val = float(stats_arr[idx])
                                            found = True
                                            break
                                        except (ValueError, TypeError):
                                            pass
                    if found:
                        break

        elif sp == "MLB":
            stat_unit = "Hits"
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", sel_lower)
            target_val = float(nums[0]) if nums else 1.0
            for t in players_data:
                for s in t.get("statistics", []):
                    labels = s.get("labels", [])
                    if "H" in labels:
                        idx = labels.index("H")
                        for ath in s.get("athletes", []):
                            ath_name = ath.get("athlete", {}).get("displayName", "")
                            if self._athlete_matches(p_name, ath_name):
                                stats_arr = ath.get("stats", [])
                                if idx < len(stats_arr):
                                    try:
                                        stat_val = float(stats_arr[idx])
                                        found = True
                                        break
                                    except (ValueError, TypeError):
                                        pass
                if found:
                    break

        is_injured = False
        for t_inj in summary.get("injuries", []):
            for inj in t_inj.get("injuries", []):
                inj_ath = inj.get("athlete", {}).get("displayName", "")
                if self._athlete_matches(p_name, inj_ath):
                    is_injured = True
                    break

        if not found and is_injured:
            if is_officially_final:
                return ("PERDIDA", f"{p_name}: Inactivo / Lesionado (0 {stat_unit})")
            return ("PROXIMAMENTE", f"{p_name}: Lesionado / En duda")

        if not found and is_officially_final:
            return ("PERDIDA", f"{p_name}: Sin participacion registrada (0 {stat_unit})")

        int_stat = int(stat_val) if float(stat_val).is_integer() else stat_val
        int_target = int(target_val) if float(target_val).is_integer() else target_val

        if stat_val >= target_val:
            return ("GANADA", f"{p_name}: {int_stat} {stat_unit} (Linea: {int_target}+ | {matched_game.get('score_str', '')})")
        else:
            if is_officially_final:
                return ("PERDIDA", f"{p_name}: {int_stat} {stat_unit} (Linea: {int_target}+ - Final: {matched_game.get('score_str', '')})")
            elif is_live_in_play:
                return ("EN JUEGO", f"{p_name}: {int_stat} {stat_unit} en vivo (Linea: {int_target}+)")
            else:
                return ("PROXIMAMENTE", f"Programado (Linea: {int_target}+ {stat_unit})")

    def _infer_leg_sport(self, leg: Dict[str, Any], ticket_sport: str) -> str:
        leg_sport = (leg.get("sport") or "").strip()
        if leg_sport and leg_sport.upper() not in ("DIA", "GLOBAL"):
            return leg_sport.upper()
        if ticket_sport and ticket_sport.upper() not in ("DIA", "GLOBAL"):
            return ticket_sport.upper()

        game_str = (leg.get("game") or "").lower()
        sel_str = (leg.get("selection") or "").lower()
        if "sets" in sel_str or leg.get("tournament") or leg.get("surface"):
            return "TENIS"
        if leg.get("horario"):
            return "NCAAF"
        if "pase de td" in sel_str or "yardas" in sel_str or any(
            k in game_str for k in ["titans", "browns", "patriots", "49ers", "steelers", "saints", "packers", "ravens", "chiefs", "bills", "texans", "vikings", "commanders"]
        ):
            return "NFL"
        if "puntos" in sel_str or "rebotes" in sel_str or "asistencias" in sel_str or any(
            k in game_str for k in ["cavaliers", "nets", "wizards", "bucks", "76ers", "heat", "hornets", "spurs", "mavericks", "warriors", "suns", "nuggets", "pelicans", "rockets", "celtics", "timberwolves", "clippers", "lakers", "knicks", "pacers", "thunder", "grizzlies", "magic", "pistons", "raptors", "trail blazers", "jazz"]
        ):
            return "NBA"
        if "remates a puerta" in sel_str or "puck line" in sel_str or any(
            k in game_str for k in ["flyers", "devils", "bruins", "sharks", "ducks", "maple leafs", "flames", "oilers", "canadiens", "rangers", "senators", "predators", "stars", "kraken", "golden knights", "avalanche", "hurricanes", "lightning", "canucks", "sabres", "capitals", "blackhawks", "islanders", "penguins", "blues", "wild", "jets"]
        ):
            return "NHL"
        if "tiro a puerta" in sel_str or "tiros a puerta" in sel_str or "empate" in sel_str or "doble oportunidad" in sel_str or " vs " in game_str:
            return "FUTBOL"
        return ""

    def _match_leg_to_live_game(
        self,
        leg: Dict[str, Any],
        ticket_date: str,
        leg_sport: str,
        live_games: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Matches a ticket leg strictly to its authentic ESPN game:
        1. Exact event_id + matching sport, OR
        2. Matching sport + BOTH home_team AND away_team match leg['game'] +
           strict date proximity (exact 0-day CST date preferred; 1-day rollover only for UTC midnight edge).
        NEVER matches single teams or distant dates.
        """
        leg_ev_id = str(leg.get("event_id") or "").strip()
        leg_date_str = str(leg.get("local_date") or leg.get("date") or ticket_date or "").strip()
        try:
            leg_dt = datetime.strptime(leg_date_str[:10], "%Y-%m-%d").date()
        except Exception:
            leg_dt = datetime.now(CST_TZ).date()

        game_text = leg.get("game") or ""
        if (not game_text or game_text == "-") and leg.get("team") and leg.get("opponent") and leg.get("team") != "-":
            game_text = f"{leg.get('team')} vs {leg.get('opponent')}"

        exact_id_match = None
        zero_day_matches = []
        one_day_matches = []

        for g_data in live_games.values():
            g_sport = (g_data.get("sport") or "").upper()
            if leg_sport and g_sport != leg_sport:
                continue

            if leg_ev_id and leg_ev_id == str(g_data.get("event_id") or ""):
                exact_id_match = g_data
                break

            try:
                g_dt = datetime.strptime(str(g_data.get("game_date", ""))[:10], "%Y-%m-%d").date()
                day_diff = abs((g_dt - leg_dt).days)
            except Exception:
                day_diff = 99

            if day_diff > 1:
                continue

            if both_teams_match_game_text(g_data["home_team"], g_data["away_team"], game_text, g_sport):
                if day_diff == 0:
                    zero_day_matches.append(g_data)
                elif day_diff == 1:
                    # Only allow 1-day boundary rollover for international Tennis / European Soccer / late UTC
                    if g_sport in ("TENIS", "FUTBOL") or g_data.get("query_ymd") == leg_date_str.replace("-", ""):
                        one_day_matches.append(g_data)

        if exact_id_match:
            return exact_id_match
        if zero_day_matches:
            return zero_day_matches[0]
        if one_day_matches:
            return one_day_matches[0]
        return None

    def sync_and_grade_all_tickets(self, extra_dates: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Synchronizes all unsettled tickets and their individual legs against official ESPN states:
        - Future or pre-game ('pre') legs are ALWAYS marked 'PROXIMAMENTE' (never 'PERDIDA').
        - Live ('in') legs are marked 'EN JUEGO'.
        - Completed ('post') legs are graded strictly from the exact matched game's final score / boxscore.
        """
        live_games = self.fetch_live_game_states(extra_dates=extra_dates)
        today_cst_date = datetime.now(CST_TZ).date()

        for ticket in self.data["tickets"]:
            if ticket.get("settled_at") and ticket.get("status") in ("GANADA", "PERDIDA"):
                continue

            legs = ticket.get("legs", [])
            leg_statuses = []
            ticket_sport = (ticket.get("sport") or "").upper()
            ticket_date = ticket.get("date", "")

            for leg in legs:
                sel_str_raw = leg.get("selection", "")
                sel_str = _fold_accents(sel_str_raw).lower()
                mkt_type = leg.get("market_type", "")
                player_str = (leg.get("player") or "").strip()
                leg_sport = self._infer_leg_sport(leg, ticket_sport)
                leg_date_str = str(leg.get("local_date") or leg.get("date") or ticket_date or "")

                try:
                    leg_dt_obj = datetime.strptime(leg_date_str[:10], "%Y-%m-%d").date()
                except Exception:
                    leg_dt_obj = today_cst_date

                matched_game = self._match_leg_to_live_game(leg, ticket_date, leg_sport, live_games)

                if not matched_game:
                    # Only keep a previously cached Final score if BOTH teams in that score strictly match leg['game']
                    score_res = leg.get("score_result", "")
                    if leg_dt_obj <= today_cst_date and ("final:" in score_res.lower() or "ft:" in score_res.lower()):
                        m = re.search(r"(?:Final|FT):\s*(.+?)\s+(\d+)\s*-\s*(.+?)\s+(\d+)", score_res, re.IGNORECASE)
                        if m:
                            t1, s1, t2, s2 = m.group(1).strip(), int(m.group(2)), m.group(3).strip(), int(m.group(4))
                            if both_teams_match_game_text(t2, t1, leg.get("game", ""), leg_sport):
                                matched_game = {
                                    "sport": leg_sport,
                                    "home_team": t2,
                                    "away_team": t1,
                                    "home_score": s2,
                                    "away_score": s1,
                                    "total": s1 + s2,
                                    "state": "post",
                                    "completed": True,
                                    "detail": "Final",
                                    "score_str": f"{t1} {s1} - {t2} {s2}",
                                    "winner": t2 if s2 > s1 else (t1 if s1 > s2 else "tie")
                                }

                if not matched_game:
                    leg["status"] = "PROXIMAMENTE"
                    leg["score_result"] = f"Programado ({leg_date_str})" if leg_date_str else "Programado"
                    leg_statuses.append("PROXIMAMENTE")
                    continue

                if matched_game.get("event_id") and not leg.get("event_id"):
                    leg["event_id"] = matched_game["event_id"]

                state = matched_game.get("state", "pre")
                completed = bool(matched_game.get("completed", False))
                detail = matched_game.get("detail", "Programado")
                score_str = matched_game.get("score_str", "")
                detail_lower = detail.strip().lower()

                is_officially_final = (
                    state == "post"
                    and (completed or "final" in detail_lower or "full time" in detail_lower or "ft" in detail_lower)
                )
                is_live_in_play = (state == "in" or (not is_officially_final and state != "pre"))

                is_total_market = (
                    "total" in mkt_type.lower()
                    or "puntos totales" in sel_str
                    or "carreras totales" in sel_str
                    or "goles totales" in sel_str
                    or (not bool(player_str) and any(k in sel_str for k in ["under", "over", "menos de", "mas de"]))
                )

                is_player_prop = not is_total_market and (
                    mkt_type in ("Player Prop", "Anotador de Gol (Anytime Goal)", "Remates a Puerta")
                    or "anotador" in mkt_type.lower()
                    or "remate" in mkt_type.lower()
                    or bool(player_str)
                    or (":" in sel_str and any(
                        k in sel_str for k in ["puntos", "remates", "tiro", "tiros", "yardas", "asistencias", "rebotes", "ponches", "hit", "pase de td"]
                    ))
                )

                if state == "pre" and not is_live_in_play:
                    leg["status"] = "PROXIMAMENTE"
                    leg["score_result"] = f"Programado: {detail}"

                elif is_live_in_play:
                    if is_player_prop:
                        p_stat, p_res = self.evaluate_player_prop(leg, matched_game, is_officially_final=False, is_live_in_play=True)
                        # Do not finalize a prop as won/lost mid-game unless already exceeded target; keep ticket live if not hit yet
                        leg["status"] = p_stat
                        leg["score_result"] = p_res
                    else:
                        leg["status"] = "EN JUEGO"
                        leg["score_result"] = f"En Juego ({detail}): {score_str}"

                elif is_officially_final:
                    h_team = matched_game["home_team"]
                    a_team = matched_game["away_team"]
                    h_score = matched_game["home_score"]
                    a_score = matched_game["away_score"]
                    total_score = matched_game["total"]
                    g_sp = matched_game.get("sport", leg_sport)

                    backed_team = leg.get("team", "")
                    backed_is_home = (
                        (backed_team and backed_team != "-" and team_matches_text(h_team, backed_team, g_sp))
                        or team_matches_text(h_team, sel_str_raw, g_sp)
                    )
                    backed_is_away = (
                        (backed_team and backed_team != "-" and team_matches_text(a_team, backed_team, g_sp))
                        or team_matches_text(a_team, sel_str_raw, g_sp)
                    )
                    # Disambiguate if game matchup string is embedded inside selection (e.g. "TeamA @ TeamB Under 230")
                    if backed_team and backed_team != "-" and "_total" not in backed_team:
                        backed_is_home = team_matches_text(h_team, backed_team, g_sp)
                        backed_is_away = team_matches_text(a_team, backed_team, g_sp)

                    # 1. Player Props
                    if is_player_prop:
                        p_stat, p_res = self.evaluate_player_prop(leg, matched_game, is_officially_final=True, is_live_in_play=False)
                        leg["status"] = p_stat
                        leg["score_result"] = p_res

                    # 2. Doble Oportunidad (1X / X2)
                    elif "o empate" in sel_str or "doble oportunidad" in sel_str:
                        if backed_is_home:
                            leg["status"] = "GANADA" if h_score >= a_score else "PERDIDA"
                        elif backed_is_away:
                            leg["status"] = "GANADA" if a_score >= h_score else "PERDIDA"
                        else:
                            leg["status"] = "GANADA" if h_score == a_score else "PERDIDA"

                    # 3. Money Line
                    elif "ml" in sel_str or "money line" in mkt_type.lower() or "a ganar" in sel_str:
                        if backed_is_home:
                            leg["status"] = "GANADA" if h_score > a_score else "PERDIDA"
                        elif backed_is_away:
                            leg["status"] = "GANADA" if a_score > h_score else "PERDIDA"
                        else:
                            leg["status"] = "PERDIDA"

                    # 4. Over / Under & Totales
                    elif is_total_market or any(k in sel_str for k in ["over", "under", "mas de", "menos de"]):
                        # Extract threshold right after Over/Under/Mas de/Menos de to avoid team numbers like '76ers' or '49ers'
                        dir_m = re.search(r"(?:over|under|mas de|menos de)\s*(\d+(?:\.\d+)?)", sel_str)
                        nums = [dir_m.group(1)] if dir_m else re.findall(r"\d+(?:\.\d+)?", sel_str)
                        if nums:
                            thresh = float(nums[-1])
                            is_team_total = (
                                "equipo" in mkt_type.lower()
                                or "team" in mkt_type.lower()
                                or (":" in sel_str and "@" not in sel_str and " vs " not in sel_str and not any(
                                    w in sel_str for w in ["@ ", " vs "]
                                ))
                            )
                            # Note: selections like 'TeamA vs TeamB: Mas de 1.5 Goles' contain ' vs ', so they are match totals, NOT team totals!
                            if is_team_total and ("@" not in sel_str and "vs" not in sel_str):
                                eval_score = h_score if backed_is_home else (a_score if backed_is_away else total_score)
                            else:
                                eval_score = total_score

                            if "over" in sel_str or "mas de" in sel_str:
                                leg["status"] = "GANADA" if eval_score > thresh else ("TABLAS" if eval_score == thresh else "PERDIDA")
                            else:
                                leg["status"] = "GANADA" if eval_score < thresh else ("TABLAS" if eval_score == thresh else "PERDIDA")

                    # 5. Spreads, Set Handicaps (+1.5 Sets), Puck Lines, Run Lines
                    elif "+" in sel_str or "-" in sel_str or "spread" in mkt_type.lower() or "handicap" in mkt_type.lower() or "sets" in sel_str:
                        signed_nums = re.findall(r"[+-]\d+(?:\.\d+)?", sel_str)
                        if signed_nums:
                            spread_val = float(signed_nums[-1])
                            is_tennis_games = (g_sp == "Tenis" and "juego" in sel_str and "set" not in sel_str)
                            val_h = matched_game.get("home_games", h_score) if is_tennis_games else h_score
                            val_a = matched_game.get("away_games", a_score) if is_tennis_games else a_score

                            if backed_is_home:
                                margin = val_h - val_a
                                leg["status"] = "GANADA" if (margin + spread_val) > 0 else ("TABLAS" if (margin + spread_val) == 0 else "PERDIDA")
                            elif backed_is_away:
                                margin = val_a - val_h
                                leg["status"] = "GANADA" if (margin + spread_val) > 0 else ("TABLAS" if (margin + spread_val) == 0 else "PERDIDA")
                            else:
                                leg["status"] = "PERDIDA"

                    if not is_player_prop:
                        leg["score_result"] = f"Final: {score_str}"

                leg.setdefault("status", "PROXIMAMENTE")
                leg_statuses.append(leg["status"])

            # Compute Ticket-Level Status from Legs
            odds_dec = float(ticket.get("odds_decimal", 1.91))
            won_cnt = sum(1 for s in leg_statuses if s == "GANADA")
            push_cnt = sum(1 for s in leg_statuses if s == "TABLAS")
            in_play_cnt = sum(1 for s in leg_statuses if s == "EN JUEGO")
            prog_cnt = sum(1 for s in leg_statuses if s == "PROXIMAMENTE")

            if "PERDIDA" in leg_statuses:
                # Only settle ticket as PERDIDA once that losing leg is genuinely final (PERDIDA is only assigned on is_officially_final)
                ticket["status"] = "PERDIDA"
                ticket["units_pnl"] = -1.0
                ticket["settled_at"] = datetime.now(CST_TZ).strftime("%Y-%m-%d %H:%M:%S")
                ticket["score_result"] = "1 o mas selecciones no cumplidas"
            elif leg_statuses and all(s in ("GANADA", "TABLAS") for s in leg_statuses) and in_play_cnt == 0 and prog_cnt == 0:
                if all(s == "TABLAS" for s in leg_statuses):
                    ticket["status"] = "TABLAS"
                    ticket["units_pnl"] = 0.0
                    ticket["settled_at"] = datetime.now(CST_TZ).strftime("%Y-%m-%d %H:%M:%S")
                    ticket["score_result"] = "Apuesta empatada (Push)"
                else:
                    # Recalculate payout excluding pushed legs if any
                    eff_dec = 1.0
                    for lg_item in legs:
                        if lg_item.get("status") == "GANADA":
                            eff_dec *= float(lg_item.get("decimal_odds") or odds_dec)
                    if not ticket.get("is_parlay"):
                        eff_dec = odds_dec
                    ticket["status"] = "GANADA"
                    ticket["units_pnl"] = round(eff_dec - 1.0, 2)
                    ticket["settled_at"] = datetime.now(CST_TZ).strftime("%Y-%m-%d %H:%M:%S")
                    ticket["score_result"] = "Todas las selecciones cumplidas"
            elif in_play_cnt > 0 or (won_cnt + push_cnt) > 0:
                ticket["status"] = "EN JUEGO"
                ticket["units_pnl"] = 0.0
                ticket["settled_at"] = None
                if in_play_cnt > 0:
                    ticket["score_result"] = f"En disputa ({won_cnt} ganadas, {in_play_cnt} en juego)"
                else:
                    ticket["score_result"] = f"En proceso ({won_cnt} ganadas, {prog_cnt} por iniciar)"
            else:
                ticket["status"] = "PROXIMAMENTE"
                ticket["units_pnl"] = 0.0
                ticket["settled_at"] = None
                ticket["score_result"] = "Partidos programados"

        self._save()
        return self.get_comprehensive_scorecards()

    def repair_corrupted_history(self, purge_preview_future_dates: bool = True) -> Dict[str, Any]:
        """
        Audits and repairs all historical & active tickets in bet_history.json:
        1. Purges old mixed-horario future preview tickets (e.g. 2026-10-31 NCAAF/Tenis created before per-horario split).
        2. Detects every ticket where any leg matched a wrong opponent/date, had false 'Inactivo' from a wrong game,
           or was graded prematurely before game completion.
        3. Resets corrupted settlements and re-grades every affected date directly from official ESPN boxscores.
        """
        today_cst_str = datetime.now(CST_TZ).strftime("%Y-%m-%d")
        kept_tickets = []
        repair_dates: Set[str] = set()
        repaired_count = 0
        purged_count = 0

        for ticket in self.data.get("tickets", []):
            t_date = str(ticket.get("date") or "")
            t_id = str(ticket.get("id") or "")
            t_sport = (ticket.get("sport") or "").upper()

            # Remove old mixed-horario future NCAAF/Tenis preview tickets (> today_cst_str) that lack per-horario vehicle naming
            if purge_preview_future_dates and t_date > today_cst_str:
                veh = ticket.get("vehicle", "")
                if t_sport == "NCAAF" and not any(h in veh for h in ("Manana", "Tarde", "Noche")):
                    purged_count += 1
                    continue
                if t_sport == "TENIS" and ticket.get("settled_at"):
                    purged_count += 1
                    continue

            needs_regrade = False
            if t_date >= today_cst_str:
                needs_regrade = True

            for leg in ticket.get("legs", []):
                # Fix legacy inflated EV display metadata (> 38% EV from pre-calibration static priors)
                try:
                    ev_val = float(leg.get("ev", 0.0) or 0.0)
                    dec_val = float(leg.get("decimal_odds", 1.91) or 1.91)
                    rp_val = float(leg.get("real_prob", 60.0) or 60.0)
                    if ev_val > 38.0:
                        if rp_val > 78.0 and dec_val >= 1.85:
                            rp_val = 65.5
                            leg["real_prob"] = rp_val
                        recalc_ev = round(((rp_val / 100.0) * dec_val - 1.0) * 100.0, 2)
                        leg["ev"] = min(26.5, max(3.5, recalc_ev))
                except Exception:
                    pass

                l_sport = self._infer_leg_sport(leg, t_sport)
                sc_res = str(leg.get("score_result") or "")
                g_txt = leg.get("game", "")

                if "inactivo" in sc_res.lower() or "no disponibles" in sc_res.lower() or "sin participacion" in sc_res.lower():
                    needs_regrade = True

                m_fin = re.search(r"(?:Final|FT):\s*(.+?)\s+\d+\s*-\s*(?:\d+\s+)?(.+?)(?:\s+\d+|\s*\(|$)", sc_res, re.IGNORECASE)
                if m_fin and g_txt and g_txt != "-":
                    team_a = m_fin.group(1).strip()
                    team_b = m_fin.group(2).strip()
                    if not both_teams_match_game_text(team_a, team_b, g_txt, l_sport):
                        needs_regrade = True

                if (leg.get("local_date") or "") > today_cst_str and leg.get("status") in ("GANADA", "PERDIDA"):
                    needs_regrade = True

            if needs_regrade:
                repaired_count += 1
                ticket["settled_at"] = None
                ticket["status"] = "PROXIMAMENTE"
                ticket["units_pnl"] = 0.0
                ticket["score_result"] = "Partidos programados"
                if t_date:
                    repair_dates.add(t_date)
                for leg in ticket.get("legs", []):
                    leg["status"] = "PROXIMAMENTE"
                    leg["score_result"] = "Programado"
                    if leg.get("local_date"):
                        repair_dates.add(str(leg["local_date"]))
                    if leg.get("date"):
                        repair_dates.add(str(leg["date"]))

            kept_tickets.append(ticket)

        self.data["tickets"] = kept_tickets
        self._save()
        scorecards = self.sync_and_grade_all_tickets(extra_dates=repair_dates)
        return {
            "purged_future_mixed_tickets": purged_count,
            "regraded_tickets": repaired_count,
            "queried_dates": sorted(list(repair_dates)),
            "scorecards": scorecards
        }

    def get_scorecard_for_subset(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(tickets)
        won = sum(1 for t in tickets if t.get("status") == "GANADA")
        lost = sum(1 for t in tickets if t.get("status") == "PERDIDA")
        in_play = sum(1 for t in tickets if t.get("status") == "EN JUEGO")
        upcoming = sum(1 for t in tickets if t.get("status") == "PROXIMAMENTE")
        push = sum(1 for t in tickets if t.get("status") == "TABLAS")

        resolved = won + lost
        win_rate = round((won / resolved) * 100, 1) if resolved > 0 else 0.0
        total_pnl = round(sum(t.get("units_pnl", 0.0) for t in tickets), 2)
        roi = round((total_pnl / resolved) * 100, 1) if resolved > 0 else 0.0

        return {
            "total_bets": total,
            "won": won,
            "lost": lost,
            "in_play": in_play,
            "upcoming": upcoming,
            "push": push,
            "win_rate_pct": win_rate,
            "win_rate": win_rate,
            "units_pnl": total_pnl,
            "total_pnl": total_pnl,
            "roi_pct": roi,
            "roi": roi,
            "tickets": tickets
        }

    def get_comprehensive_scorecards(self) -> Dict[str, Any]:
        tickets = self.data["tickets"]
        sports = {
            "general": self.get_scorecard_for_subset(tickets),
            "dia_mvp": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") in ("Global", "Dia")]),
            "nba": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "NBA"]),
            "nhl": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "NHL"]),
            "futbol": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "Futbol"]),
            "ncaaf": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "NCAAF"]),
            "tenis": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "Tenis"]),
            "mlb": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "MLB"]),
            "nfl": self.get_scorecard_for_subset([t for t in tickets if t.get("sport") == "NFL"])
        }
        vehicles = {
            "apuesta_fuerte": self.get_scorecard_for_subset([t for t in tickets if "Apuesta Fuerte" in t.get("vehicle", "")]),
            "parlay_ratonero": self.get_scorecard_for_subset([t for t in tickets if "Ratonero" in t.get("vehicle", "")]),
            "parlay_props": self.get_scorecard_for_subset([t for t in tickets if "Props" in t.get("vehicle", "")]),
            "parlay_ganador": self.get_scorecard_for_subset([t for t in tickets if "Ganador" in t.get("vehicle", "")])
        }
        return {"sports": sports, "vehicles": vehicles}

    def get_available_dates(self) -> List[str]:
        dates = sorted(list(set(t.get("date") for t in self.data.get("tickets", []) if t.get("date"))), reverse=True)
        return dates if dates else [datetime.now(CST_TZ).strftime("%Y-%m-%d")]

    def get_scorecard_for_date(self, date_str: str) -> Dict[str, Any]:
        date_tickets = self.get_tickets_by_date(date_str)
        sc = self.get_scorecard_for_subset(date_tickets)
        sc["tickets"] = date_tickets
        sc["by_vehicle"] = {
            "apuesta_fuerte": self.get_scorecard_for_subset([t for t in date_tickets if "Apuesta Fuerte" in t.get("vehicle", "")]),
            "parlay_ratonero": self.get_scorecard_for_subset([t for t in date_tickets if "Ratonero" in t.get("vehicle", "")]),
            "parlay_props": self.get_scorecard_for_subset([t for t in date_tickets if "Props" in t.get("vehicle", "")]),
            "parlay_ganador": self.get_scorecard_for_subset([t for t in date_tickets if "Ganador" in t.get("vehicle", "")])
        }
        return sc
