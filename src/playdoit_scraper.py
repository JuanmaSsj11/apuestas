"""
Playdoit (PD) Scraper and Mexican Sportsbook Odds Calibrator
Handles direct scraping of Playdoit / Altenar endpoints using curl_cffi TLS impersonation,
and provides precise Playdoit vig-adjusted real odds conversion and interactive +EV validation.
"""
import logging
from typing import Dict, List, Any, Optional
from curl_cffi import requests

logger = logging.getLogger("PlaydoitScraper")

class PlaydoitScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.base_url = "https://www.playdoit.mx"
        self.altenar_host = "https://sb2frontend-altenar2.biahosted.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://www.playdoit.mx",
            "Referer": "https://www.playdoit.mx/deportes",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "es-MX,es;q=0.9"
        }
        self.session = requests.Session(impersonate="chrome120")

    def test_playdoit_connection(self) -> Dict[str, Any]:
        """
        Verifies live connectivity to Playdoit.mx bypassing Cloudflare via curl_cffi.
        """
        try:
            r = self.session.get(f"{self.base_url}/deportes", headers=self.headers, timeout=self.timeout)
            return {
                "success": r.status_code == 200,
                "status_code": r.status_code,
                "provider": "Playdoit (Altenar Sportsbook México)",
                "cloudflare_bypassed": r.status_code == 200
            }
        except Exception as e:
            logger.warning(f"Error testing Playdoit connection: {e}")
            return {
                "success": False,
                "status_code": None,
                "error": str(e)
            }

    @staticmethod
    def adjust_odds_to_playdoit_margin(us_american_odds: int, additional_vig: float = 0.02) -> int:
        """
        Adjusts US baseline lines (DraftKings 4.5% vig) to Playdoit (Altenar Mexico ~6.5% vig).
        Typically, Playdoit shades favorites down (e.g. -148 becomes -160)
        and underdogs slightly lower (e.g. +125 becomes +115).
        """
        if us_american_odds is None or us_american_odds == 0:
            return us_american_odds

        # Convert to decimal
        if us_american_odds > 0:
            dec = 1.0 + (us_american_odds / 100.0)
        else:
            dec = 1.0 + (100.0 / abs(us_american_odds))

        # Apply Mexican market vig penalty (~2% haircut on gross payout)
        pd_dec = max(1.05, dec * (1.0 - additional_vig))

        # Convert back to American
        if pd_dec >= 2.0:
            pd_american = int(round((pd_dec - 1.0) * 100))
        else:
            pd_american = int(round(-100.0 / (pd_dec - 1.0)))

        return pd_american

    @staticmethod
    def calculate_playdoit_ev(real_prob_pct: float, pd_american_odds: int) -> Dict[str, Any]:
        """
        Calculates exact +EV in Playdoit given the real model probability
        and the exact American odds shown on Playdoit app.
        Formula: EV = (Prob * DecOdds) - 1.0
        """
        prob = real_prob_pct / 100.0 if real_prob_pct > 1.0 else real_prob_pct
        if pd_american_odds > 0:
            dec_odds = 1.0 + (pd_american_odds / 100.0)
        elif pd_american_odds < 0:
            dec_odds = 1.0 + (100.0 / abs(pd_american_odds))
        else:
            dec_odds = 1.0

        ev = (prob * dec_odds) - 1.0
        ev_pct = round(ev * 100.0, 2)
        is_positive = ev_pct > 0.0

        return {
            "pd_american_odds": pd_american_odds,
            "pd_decimal_odds": round(dec_odds, 2),
            "real_prob_pct": round(prob * 100.0, 1),
            "ev_pct": ev_pct,
            "is_positive": is_positive,
            "status_text": "VALOR POSITIVO [+EV]" if is_positive else "VALOR NEGATIVO [-EV]"
        }
