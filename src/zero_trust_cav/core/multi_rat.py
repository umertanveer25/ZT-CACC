"""
Heterogeneous Multi-RAT V2X Broker inspired by the Sto-CAV architecture.
Dynamically handles low-latency packet routing across Optical VLC, ITS-G5, and LTE-V2X.
"""

from enum import Enum
from typing import Tuple
from ..config import MultiRATConfig

class RATType(Enum):
    VLC = "Optical VLC (Primary)"
    ITS_G5 = "ITS-G5 / 802.11p (Secondary)"
    LTE_V2X = "LTE-V2X / Sidelink (Fallback)"
    FAILED = "Link Failure / Jammed"

class MultiRATBroker:
    """Manages dynamic RAT selection, SINR monitoring, and failover state machines."""
    def __init__(self, config: MultiRATConfig):
        self.cfg = config

    def get_active_rat(self, current_time_s: float) -> Tuple[RATType, float]:
        """
        Determines the optimal active Radio Access Technology and associated latency.
        
        Failover Hierarchy:
          1. VLC: Ultra-low latency (1.8 ms) - disabled during Optical Glare
          2. ITS-G5: Low latency (8.4 ms) - disabled during RF Jamming
          3. LTE-V2X: Robust cellular fallback (14.2 ms)
        """
        is_glare = (self.cfg.glare_start_s <= current_time_s <= self.cfg.glare_end_s)
        is_jammed = (self.cfg.jamming_start_s <= current_time_s <= self.cfg.jamming_end_s)

        if not is_glare:
            return RATType.VLC, self.cfg.vlc_delay_ms / 1000.0
        elif not is_jammed:
            return RATType.ITS_G5, self.cfg.its_g5_delay_ms / 1000.0
        else:
            return RATType.LTE_V2X, self.cfg.lte_v2x_delay_ms / 1000.0
