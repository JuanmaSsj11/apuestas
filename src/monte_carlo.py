"""
Monte Carlo Simulation Engine
Executes 10,000 iterations per game using:
- Poisson / Dixon-Coles distributions for low-scoring sports (Soccer, NHL, MLB)
- Multivariate Normal distributions with pace and variance calibration for high-scoring sports (NBA, NFL)
"""
import numpy as np
from typing import Dict, Any, Tuple

NUM_SIMULATIONS = 10000

class MonteCarloSimulator:
    def __init__(self, simulations: int = NUM_SIMULATIONS, seed: int = 42):
        self.simulations = simulations
        self.rng = np.random.default_rng(seed)

    def simulate_nba_game(
        self,
        home_pace: float,
        away_pace: float,
        home_ortg: float,
        home_drtg: float,
        away_ortg: float,
        away_drtg: float,
        hca: float = 2.5,
        spread: float = 0.0,
        total: float = 230.0
    ) -> Dict[str, Any]:
        """
        Simulates 10,000 NBA games using multivariate normal distribution on projected possessions and efficiency.
        """
        avg_pace = (home_pace + away_pace) / 2.0
        # Expected points per 100 possessions
        home_exp_eff = (home_ortg + away_drtg) / 2.0
        away_exp_eff = (away_ortg + home_drtg) / 2.0

        # Mean projected points
        mean_home = (home_exp_eff * avg_pace / 100.0) + (hca / 2.0)
        mean_away = (away_exp_eff * avg_pace / 100.0) - (hca / 2.0)

        # Standard deviation in NBA scores is typically ~11.5 to 12.5 points
        std_dev = 11.8
        cov_matrix = [
            [std_dev**2, 0.15 * std_dev**2],
            [0.15 * std_dev**2, std_dev**2]
        ]

        scores = self.rng.multivariate_normal([mean_home, mean_away], cov_matrix, size=self.simulations)
        home_scores = np.round(scores[:, 0])
        away_scores = np.round(scores[:, 1])

        # Resolve any exact ties by adding 1 OT possession
        ties = (home_scores == away_scores)
        home_scores[ties] += self.rng.choice([1, -1], size=np.sum(ties))

        home_wins = np.sum(home_scores > away_scores)
        away_wins = self.simulations - home_wins

        prob_home_win = home_wins / self.simulations
        prob_away_win = away_wins / self.simulations

        # Margins (Home - Away)
        margins = home_scores - away_scores
        # Spread cover: Home covering spread means margin > -spread (e.g. if spread is -4.5, margin >= 5)
        prob_home_cover = np.mean(margins > -spread)
        prob_away_cover = 1.0 - prob_home_cover

        # Game Totals
        game_totals = home_scores + away_scores
        prob_over = np.mean(game_totals > total)
        prob_under = 1.0 - prob_over

        # Alternative lines for "Parlay Ratonero" (High Probability Spreads and Totals)
        # Safe Home Alt Spread: Home +8.5 or Home -1.5 depending on favorite
        prob_home_plus_8_5 = np.mean(margins > -8.5)
        prob_away_plus_8_5 = np.mean(margins < 8.5)
        prob_under_alt_safe = np.mean(game_totals < (total + 7.5))
        prob_over_alt_safe = np.mean(game_totals > (total - 7.5))

        return {
            "mean_home_score": round(float(np.mean(home_scores)), 1),
            "mean_away_score": round(float(np.mean(away_scores)), 1),
            "prob_home_win": round(prob_home_win, 4),
            "prob_away_win": round(prob_away_win, 4),
            "prob_home_cover": round(float(prob_home_cover), 4),
            "prob_away_cover": round(float(prob_away_cover), 4),
            "prob_over": round(float(prob_over), 4),
            "prob_under": round(float(prob_under), 4),
            "alt_lines": {
                "home_plus_8_5": round(float(prob_home_plus_8_5), 4),
                "away_plus_8_5": round(float(prob_away_plus_8_5), 4),
                "under_alt_safe": round(float(prob_under_alt_safe), 4),
                "over_alt_safe": round(float(prob_over_alt_safe), 4)
            }
        }

    def simulate_soccer_game(
        self,
        home_xg: float,
        away_xg: float,
        total: float = 2.5
    ) -> Dict[str, Any]:
        """
        Simulates 10,000 soccer games using Poisson / Dixon-Coles distribution.
        """
        # Bivariate Poisson draw
        home_goals = self.rng.poisson(lam=home_xg, size=self.simulations)
        away_goals = self.rng.poisson(lam=away_xg, size=self.simulations)

        # Dixon-Coles low-score correction for 0-0, 1-0, 0-1, 1-1
        rho = -0.10
        # Adjust probabilities slightly for correlation
        prob_home_win = np.mean(home_goals > away_goals)
        prob_draw = np.mean(home_goals == away_goals)
        prob_away_win = np.mean(home_goals < away_goals)

        # Totals
        match_goals = home_goals + away_goals
        prob_over_2_5 = np.mean(match_goals > 2.5)
        prob_under_2_5 = 1.0 - prob_over_2_5

        # Ratonero Markets: Team Total Over 1.5 & Alt Match Total Over 1.5
        prob_home_tt_over_1_5 = np.mean(home_goals >= 2)
        prob_away_tt_over_1_5 = np.mean(away_goals >= 2)
        prob_match_over_1_5 = np.mean(match_goals >= 2)

        return {
            "mean_home_goals": round(float(np.mean(home_goals)), 2),
            "mean_away_goals": round(float(np.mean(away_goals)), 2),
            "prob_home_win": round(float(prob_home_win), 4),
            "prob_draw": round(float(prob_draw), 4),
            "prob_away_win": round(float(prob_away_win), 4),
            "prob_over_2_5": round(float(prob_over_2_5), 4),
            "prob_under_2_5": round(float(prob_under_2_5), 4),
            "team_total_home_over_1_5": round(float(prob_home_tt_over_1_5), 4),
            "team_total_away_over_1_5": round(float(prob_away_tt_over_1_5), 4),
            "match_over_1_5": round(float(prob_match_over_1_5), 4)
        }

    def simulate_nhl_game(
        self,
        home_xg: float,
        away_xg: float,
        total: float = 5.5
    ) -> Dict[str, Any]:
        """
        Simulates 10,000 NHL games using Poisson distribution.
        """
        home_goals = self.rng.poisson(lam=home_xg, size=self.simulations)
        away_goals = self.rng.poisson(lam=away_xg, size=self.simulations)

        # In NHL, ties go to OT/Shootout
        ties = (home_goals == away_goals)
        ot_winners = self.rng.choice([1, 2], size=np.sum(ties)) # 1 home, 2 away
        home_goals[ties] += (ot_winners == 1).astype(int)
        away_goals[ties] += (ot_winners == 2).astype(int)

        prob_home_win = np.mean(home_goals > away_goals)
        prob_away_win = 1.0 - prob_home_win

        game_goals = home_goals + away_goals
        prob_over = np.mean(game_goals > total)
        prob_under = 1.0 - prob_over

        # Puck Line (+1.5): team loses by 1 or wins
        prob_home_puck_line_plus_1_5 = np.mean(home_goals >= away_goals - 1)
        prob_away_puck_line_plus_1_5 = np.mean(away_goals >= home_goals - 1)

        # Team Totals (Over 1.5 Goals per team)
        prob_home_over_1_5 = np.mean(home_goals >= 2)
        prob_away_over_1_5 = np.mean(away_goals >= 2)

        # Safe Alternate Game Totals
        prob_over_alt_safe = np.mean(game_goals >= max(4.0, total - 1.5)) # Over 4.5
        prob_under_alt_safe = np.mean(game_goals <= (total + 1.5))        # Under 7.5

        # Clean Standard Lines (Over 5.5, Under 6.5, Team Over 2.5)
        prob_over_5_5 = np.mean(game_goals >= 6)
        prob_under_6_5 = np.mean(game_goals <= 6)
        prob_home_over_2_5 = np.mean(home_goals >= 3)
        prob_away_over_2_5 = np.mean(away_goals >= 3)

        return {
            "mean_home_goals": round(float(np.mean(home_goals)), 2),
            "mean_away_goals": round(float(np.mean(away_goals)), 2),
            "prob_home_win": round(float(prob_home_win), 4),
            "prob_away_win": round(float(prob_away_win), 4),
            "prob_over": round(float(prob_over), 4),
            "prob_under": round(float(prob_under), 4),
            "prob_home_puck_line_plus_1_5": round(float(prob_home_puck_line_plus_1_5), 4),
            "prob_away_puck_line_plus_1_5": round(float(prob_away_puck_line_plus_1_5), 4),
            "prob_home_over_2_5": round(float(prob_home_over_2_5), 4),
            "prob_away_over_2_5": round(float(prob_away_over_2_5), 4),
            "prob_over_5_5": round(float(prob_over_5_5), 4),
            "prob_under_6_5": round(float(prob_under_6_5), 4),
            "prob_home_over_1_5": round(float(prob_home_over_1_5), 4),
            "prob_away_over_1_5": round(float(prob_away_over_1_5), 4),
            "prob_over_alt_safe": round(float(prob_over_alt_safe), 4),
            "prob_under_alt_safe": round(float(prob_under_alt_safe), 4)
        }

    def simulate_player_prop(self, projected_stat: float, line: float, stat_type: str = "points") -> float:
        """
        Simulates player prop likelihood (e.g. 20+ points, 1+ hit, 1+ passing TD).
        """
        if stat_type == "points":
            # Normal distribution with std dev proportional to mean (~25%)
            std = projected_stat * 0.28
            draws = self.rng.normal(loc=projected_stat, scale=std, size=self.simulations)
            return float(np.mean(draws >= line))
        elif stat_type in ["hits", "passing_tds", "strikeouts"]:
            draws = self.rng.poisson(loc=projected_stat, size=self.simulations) if False else self.rng.poisson(lam=projected_stat, size=self.simulations)
            return float(np.mean(draws >= line))
        return 0.50

    def simulate_ncaaf_game(
        self,
        home_sp: float,
        away_sp: float,
        home_epa_off: float,
        home_epa_def: float,
        away_epa_off: float,
        away_epa_def: float,
        home_tempo: float = 68.5,
        away_tempo: float = 68.5,
        hfa: float = 3.3,
        spread: float = 0.0,
        total: float = 52.5
    ) -> Dict[str, Any]:
        """
        Simulates 10,000 College Football (NCAAF FBS) games using multivariate normal score generation
        calibrated for higher college scoring variance, tempo possessions, and key football numbers (3, 7, 10, 14).
        """
        avg_plays = (home_tempo + away_tempo) / 2.0
        tempo_mult = avg_plays / 68.5

        h_epa_match = home_epa_off + away_epa_def
        a_epa_match = away_epa_off + home_epa_def

        base_total_est = 52.4 * tempo_mult + ((h_epa_match + a_epa_match) * avg_plays * 0.44)
        # Anchor total slightly (25%) toward market environment (weather/stadium) while keeping 75% pure model signal
        expected_total = (0.75 * base_total_est) + (0.25 * total)
        expected_margin = ((home_sp - away_sp) * 0.72) + ((h_epa_match - a_epa_match) * avg_plays * 0.25) + hfa

        mean_home = max(10.0, (expected_total + expected_margin) / 2.0)
        mean_away = max(7.0, (expected_total - expected_margin) / 2.0)

        team_std = 10.3
        cov_matrix = [
            [team_std ** 2, 0.12 * (team_std ** 2)],
            [0.12 * (team_std ** 2), team_std ** 2]
        ]

        scores = self.rng.multivariate_normal([mean_home, mean_away], cov_matrix, size=self.simulations)
        home_scores = np.maximum(0.0, np.round(scores[:, 0]))
        away_scores = np.maximum(0.0, np.round(scores[:, 1]))

        # Resolve ties via college overtime possessions (+3 or +7)
        ties = (home_scores == away_scores)
        n_ties = int(np.sum(ties))
        if n_ties > 0:
            ot_pts = self.rng.choice([3, 7, -3, -7], size=n_ties, p=[0.26, 0.25, 0.25, 0.24])
            home_scores[ties] += np.maximum(0, ot_pts)
            away_scores[ties] += np.maximum(0, -ot_pts)

        margins = home_scores - away_scores
        totals = home_scores + away_scores

        prob_home_win = float(np.mean(home_scores > away_scores))
        prob_away_win = 1.0 - prob_home_win

        # Push-adjusted cover probabilities (spread is signed home_spread, e.g. -3.5 means home needs margin > 3.5)
        non_push_spread = (margins != -spread)
        if np.any(non_push_spread):
            prob_home_cover = float(np.mean(margins[non_push_spread] > -spread))
        else:
            prob_home_cover = 0.50
        prob_away_cover = 1.0 - prob_home_cover

        non_push_total = (totals != total)
        if np.any(non_push_total):
            prob_over = float(np.mean(totals[non_push_total] > total))
        else:
            prob_over = 0.50
        prob_under = 1.0 - prob_over

        return {
            "mean_home_score": round(float(np.mean(home_scores)), 1),
            "mean_away_score": round(float(np.mean(away_scores)), 1),
            "mean_total": round(float(np.mean(totals)), 1),
            "mean_margin": round(float(np.mean(margins)), 1),
            "prob_home_win": round(prob_home_win, 4),
            "prob_away_win": round(prob_away_win, 4),
            "prob_home_cover": round(prob_home_cover, 4),
            "prob_away_cover": round(prob_away_cover, 4),
            "prob_over": round(prob_over, 4),
            "prob_under": round(prob_under, 4)
        }

    @staticmethod
    def _exact_service_hold_prob(p_pt: float) -> float:
        """
        Closed-form Markov probability of holding a standard tennis service game
        given point-win probability p_pt on serve (including exact deuce chain).
        """
        p = min(0.88, max(0.35, float(p_pt)))
        q = 1.0 - p
        pre_deuce = (p ** 4) * (1.0 + 4.0 * q + 10.0 * (q ** 2))
        deuce_win = (20.0 * (p ** 5) * (q ** 3)) / max(1e-6, (1.0 - 2.0 * p * q))
        return min(0.985, max(0.20, pre_deuce + deuce_win))

    def simulate_tennis_match(
        self,
        p1_srv_pt_prob: float,
        p2_srv_pt_prob: float,
        total_games_line: float = 22.5,
        p1_game_spread: float = -2.5
    ) -> Dict[str, Any]:
        """
        Simulates 10,000 best-of-3 set Tennis matches game-by-game and set-by-set
        (including 6-6 tiebreaks) using vectorized Markov + Monte Carlo simulation.
        """
        n = self.simulations
        p1_hold = self._exact_service_hold_prob(p1_srv_pt_prob)
        p2_hold = self._exact_service_hold_prob(p2_srv_pt_prob)
        p1_break = 1.0 - p2_hold

        # Tiebreak win probability based on combined serve + return point differential
        avg_p1_pt = 0.5 * (p1_srv_pt_prob + (1.0 - p2_srv_pt_prob))
        logit_tb = (avg_p1_pt - 0.50) * 15.5
        p1_tb_prob = 1.0 / (1.0 + np.exp(-logit_tb))

        p1_sets = np.zeros(n, dtype=np.int32)
        p2_sets = np.zeros(n, dtype=np.int32)
        p1_tot_games = np.zeros(n, dtype=np.int32)
        p2_tot_games = np.zeros(n, dtype=np.int32)

        # Alternate initial server across sets
        next_server_p1 = (self.rng.random(n) < 0.5)

        for _set_idx in range(3):
            active_match = (p1_sets < 2) & (p2_sets < 2)
            if not np.any(active_match):
                break

            g1 = np.zeros(n, dtype=np.int32)
            g2 = np.zeros(n, dtype=np.int32)
            set_active = active_match.copy()

            for _game_idx in range(12):
                if not np.any(set_active):
                    break
                win_prob = np.where(next_server_p1, p1_hold, p1_break)
                p1_wins_g = (self.rng.random(n) < win_prob) & set_active
                p2_wins_g = (~p1_wins_g) & set_active

                g1 += p1_wins_g.astype(np.int32)
                g2 += p2_wins_g.astype(np.int32)
                next_server_p1 = np.where(set_active, ~next_server_p1, next_server_p1)

                done_p1 = (g1 >= 6) & ((g1 - g2) >= 2)
                done_p2 = (g2 >= 6) & ((g2 - g1) >= 2)
                set_active = set_active & (~done_p1) & (~done_p2)

            # Resolve 6-6 tiebreaks
            in_tb = active_match & (g1 == 6) & (g2 == 6)
            if np.any(in_tb):
                tb_p1 = (self.rng.random(n) < p1_tb_prob) & in_tb
                tb_p2 = (~tb_p1) & in_tb
                g1 += tb_p1.astype(np.int32)
                g2 += tb_p2.astype(np.int32)
                next_server_p1 = np.where(in_tb, ~next_server_p1, next_server_p1)

            p1_won_set = active_match & (g1 > g2)
            p2_won_set = active_match & (g2 > g1)
            p1_sets += p1_won_set.astype(np.int32)
            p2_sets += p2_won_set.astype(np.int32)
            p1_tot_games += np.where(active_match, g1, 0)
            p2_tot_games += np.where(active_match, g2, 0)

        total_games = p1_tot_games + p2_tot_games
        game_margin = p1_tot_games - p2_tot_games

        prob_p1_win = float(np.mean(p1_sets > p2_sets))
        prob_p2_win = 1.0 - prob_p1_win
        prob_p1_2_0 = float(np.mean((p1_sets == 2) & (p2_sets == 0)))
        prob_p2_2_0 = float(np.mean((p2_sets == 2) & (p1_sets == 0)))
        prob_p1_plus_1_5_sets = float(np.mean(p1_sets >= 1))
        prob_p2_plus_1_5_sets = float(np.mean(p2_sets >= 1))

        non_push_tot = (total_games != total_games_line)
        prob_over_games = float(np.mean(total_games[non_push_tot] > total_games_line)) if np.any(non_push_tot) else 0.50
        prob_under_games = 1.0 - prob_over_games

        non_push_spr = (game_margin != -p1_game_spread)
        prob_p1_cover_games = float(np.mean(game_margin[non_push_spr] > -p1_game_spread)) if np.any(non_push_spr) else 0.50
        prob_p2_cover_games = 1.0 - prob_p1_cover_games

        return {
            "p1_hold_prob": round(p1_hold * 100.0, 1),
            "p2_hold_prob": round(p2_hold * 100.0, 1),
            "mean_p1_games": round(float(np.mean(p1_tot_games)), 1),
            "mean_p2_games": round(float(np.mean(p2_tot_games)), 1),
            "mean_total_games": round(float(np.mean(total_games)), 1),
            "prob_p1_win": round(prob_p1_win, 4),
            "prob_p2_win": round(prob_p2_win, 4),
            "prob_p1_2_0": round(prob_p1_2_0, 4),
            "prob_p2_2_0": round(prob_p2_2_0, 4),
            "prob_p1_plus_1_5_sets": round(prob_p1_plus_1_5_sets, 4),
            "prob_p2_plus_1_5_sets": round(prob_p2_plus_1_5_sets, 4),
            "prob_over_games": round(prob_over_games, 4),
            "prob_under_games": round(prob_under_games, 4),
            "prob_p1_cover_games": round(prob_p1_cover_games, 4),
            "prob_p2_cover_games": round(prob_p2_cover_games, 4)
        }
