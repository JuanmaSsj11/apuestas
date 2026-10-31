"""
Bankroll and Capital Management Engine
Enforces strict institutional capital preservation:
- 90% Bóveda Intocable (Untouchable Vault, 100% protected)
- 10% Capital Activo (Maximum at risk per betting cycle)
- Rigorous stake sizing across the 4 vehicles ensuring sum(stakes) <= Capital Activo
"""
from typing import Dict, Any

class BankrollManager:
    def __init__(self, total_bankroll: float = 1000.0):
        self.total_bankroll = total_bankroll
        self.vault_percentage = 0.90
        self.active_percentage = 0.10

        self.vault_capital = self.total_bankroll * self.vault_percentage
        self.active_capital = self.total_bankroll * self.active_percentage

        # Allocation shares of the active capital (sum to 1.0)
        self.allocations = {
            "sniper": 0.40,    # 4.0% of total bankroll
            "ganador": 0.25,   # 2.5% of total bankroll
            "ratonero": 0.20,  # 2.0% of total bankroll
            "props": 0.15      # 1.5% of total bankroll
        }

    def get_vehicle_stakes(self) -> Dict[str, Any]:
        """
        Calculates exact currency and percentage stakes for each vehicle.
        """
        sniper_stake = self.active_capital * self.allocations["sniper"]
        ganador_stake = self.active_capital * self.allocations["ganador"]
        ratonero_stake = self.active_capital * self.allocations["ratonero"]
        props_stake = self.active_capital * self.allocations["props"]
        total_staked = sniper_stake + ganador_stake + ratonero_stake + props_stake

        return {
            "total_bankroll": self.total_bankroll,
            "vault_amount": round(self.vault_capital, 2),
            "vault_pct": 90.0,
            "active_amount": round(self.active_capital, 2),
            "active_pct": 10.0,
            "stakes": {
                "sniper": {
                    "amount": round(sniper_stake, 2),
                    "pct_bankroll": round(self.allocations["sniper"] * self.active_percentage * 100, 1),
                    "pct_active": round(self.allocations["sniper"] * 100, 1)
                },
                "ganador": {
                    "amount": round(ganador_stake, 2),
                    "pct_bankroll": round(self.allocations["ganador"] * self.active_percentage * 100, 1),
                    "pct_active": round(self.allocations["ganador"] * 100, 1)
                },
                "ratonero": {
                    "amount": round(ratonero_stake, 2),
                    "pct_bankroll": round(self.allocations["ratonero"] * self.active_percentage * 100, 1),
                    "pct_active": round(self.allocations["ratonero"] * 100, 1)
                },
                "props": {
                    "amount": round(props_stake, 2),
                    "pct_bankroll": round(self.allocations["props"] * self.active_percentage * 100, 1),
                    "pct_active": round(self.allocations["props"] * 100, 1)
                }
            },
            "total_staked": round(total_staked, 2)
        }
