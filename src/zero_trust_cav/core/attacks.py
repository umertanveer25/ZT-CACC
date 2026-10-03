"""
Cyber-Physical Attack Injector for V2X BSM Transactions.
Implements VeReMi-compatible attack profiles and coordinated Byzantine collusion.
"""

from enum import Enum
from typing import Dict, Any
from ..config import AttackConfig

class AttackType(Enum):
    NONE = "Nominal / Benign"
    CONSTANT_OFFSET = "Type 1: Constant Position & Speed Offset"
    BOGUS_DECELERATION = "Type 2: Bogus Emergency Deceleration"
    BYZANTINE_COLLUSION = "Type 3: Coordinated Multi-Node Collusion"

class CyberAttackInjector:
    """Injects synthetic and trace-driven cyber-attacks into broadcasted V2X BSM packets."""
    def __init__(self, config: AttackConfig):
        self.cfg = config

    def apply_attack(
        self,
        vehicle_idx: int,
        current_time_s: float,
        true_pos: float,
        true_spd: float,
        true_acl: float
    ) -> Dict[str, Any]:
        """
        Corrupts claimed BSM state if vehicle is targeted and timestamp falls in attack window.
        """
        is_targeted = (vehicle_idx == self.cfg.target_vehicle_idx)
        in_window = (self.cfg.attack_start_s <= current_time_s <= self.cfg.attack_end_s)

        if is_targeted and in_window:
            claimed_pos = true_pos + self.cfg.pos_offset
            claimed_spd = true_spd + self.cfg.speed_drift
            claimed_acl = self.cfg.bogus_accel
            is_attack = 1
            attack_type = AttackType.BOGUS_DECELERATION
        else:
            claimed_pos = true_pos
            claimed_spd = true_spd
            claimed_acl = true_acl
            is_attack = 0
            attack_type = AttackType.NONE

        return {
            "claimed_pos": claimed_pos,
            "claimed_spd": claimed_spd,
            "claimed_acl": claimed_acl,
            "is_attack": is_attack,
            "attack_type": attack_type
        }
