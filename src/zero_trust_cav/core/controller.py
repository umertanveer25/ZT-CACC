"""
Resilient Cooperative Adaptive Cruise Control (CACC) Law and Dynamic Trust Manager.
"""

import numpy as np
from ..config import CACCConfig, DetectionConfig

class TrustManager:
    """Manages continuous dynamic trust score evaluation with asymmetric decay and recovery."""
    def __init__(self, detection_config: DetectionConfig, initial_trust: float = 1.0):
        self.cfg = detection_config
        self.trust: float = initial_trust

    def update(self, anomaly_probability: float) -> float:
        """
        Asymmetric trust update law:
        Rapid exponential-like penalty upon threat detection; slow conservative recovery.
        """
        if anomaly_probability >= self.cfg.threat_threshold:
            # Degrade trust aggressively
            self.trust = max(0.0, self.trust - self.cfg.alpha_decay * anomaly_probability)
        else:
            # Recover trust conservatively
            self.trust = min(1.0, self.trust + self.cfg.beta_recovery * (1.0 - anomaly_probability))
        return self.trust

class ResilientCACCController:
    """
    Trust-weighted CACC controller with smooth autonomous fallback.
    u_i(t) = kp * e_i(t) + kd * dot(e_i)(t) + T_i(t) * ka * a_{i-1, claim}(t - tau_comm)
    """
    def __init__(self, cacc_config: CACCConfig, detection_config: DetectionConfig):
        self.cacc_cfg = cacc_config
        self.trust_mgr = TrustManager(detection_config)

    def compute_control(
        self,
        pos_self: float,
        spd_self: float,
        pos_lead: float,
        spd_lead: float,
        claimed_lead_accel: float,
        anomaly_probability: float,
        vehicle_length: float = 4.5,
        standstill_gap: float = 5.0,
        nominal_ht: float = 0.60
    ) -> tuple[float, float]:
        """
        Computes the resilient commanded acceleration and updated trust score.
        """
        trust = self.trust_mgr.update(anomaly_probability)
        
        # Adaptive headway interpolation: scale smoothly from 0.6s to 1.2s if trust collapses
        effective_ht = nominal_ht * trust + self.cacc_cfg.acc_fallback_ht * (1.0 - trust)
        desired_spacing = standstill_gap + effective_ht * spd_self
        
        # Spacing and velocity error calculation
        spacing_error = pos_lead - pos_self - vehicle_length - desired_spacing
        velocity_error = spd_lead - spd_self
        
        # Resilient control law
        feedforward = trust * self.cacc_cfg.ka * claimed_lead_accel
        feedback = self.cacc_cfg.kp * spacing_error + self.cacc_cfg.kd * velocity_error
        commanded_accel = feedback + feedforward
        
        return commanded_accel, trust
