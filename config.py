"""
Configuration and constants for Sports Betting Quantitative Pipeline
"""
import os

# API Endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Monte Carlo Settings
MONTE_CARLO_SIMULATIONS = 10000

# Bankroll Settings
BANKROLL_TOTAL = 1000.0  # Base unit for display (or scalable to any unit)
VAULT_PERCENTAGE = 0.90   # 90% Untouchable Vault
ACTIVE_PERCENTAGE = 0.10  # 10% Active Capital

# Vehicle Stake Allocations (% of Total Bankroll)
STAKE_SNIPER_PCT = 0.040   # 4.0% of total bankroll (40% of active capital)
STAKE_GANADOR_PCT = 0.025  # 2.5% of total bankroll (25% of active capital)
STAKE_RATONERO_PCT = 0.020 # 2.0% of total bankroll (20% of active capital)
STAKE_PROPS_PCT = 0.015    # 1.5% of total bankroll (15% of active capital)

# Vehicle Odds Constraints
SNIPER_MIN_AMERICAN = -150
SNIPER_MAX_AMERICAN = 100

RATONERO_MIN_AMERICAN = -450
RATONERO_MAX_AMERICAN = -250

# Expanded Soccer Leagues (5 Big European Leagues + Domestic Cups + UEFA + Liga MX)
SOCCER_LEAGUES_CONFIG = [
    ("eng.1", "Premier League"),
    ("esp.1", "LaLiga"),
    ("ita.1", "Serie A"),
    ("ger.1", "Bundesliga"),
    ("fra.1", "Ligue 1"),
    ("eng.fa", "FA Cup"),
    ("eng.league_cup", "EFL Carabao Cup"),
    ("esp.copa_del_rey", "Copa del Rey"),
    ("ita.coppa_italia", "Coppa Italia"),
    ("ger.dfb_pokal", "DFB-Pokal"),
    ("fra.coupe_de_france", "Coupe de France"),
    ("uefa.champions", "UEFA Champions League"),
    ("uefa.europa", "UEFA Europa League"),
    ("uefa.europa.conf", "UEFA Conference League"),
    ("mex.1", "Liga MX"),
]

# NCAAF Horarios (America/Mexico_City UTC-6)
NCAAF_HORARIOS = ("Manana", "Tarde", "Noche")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
