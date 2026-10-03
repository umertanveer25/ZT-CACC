"""
Analytical and Numerical String Stability Analyzer.
Calculates the closed-loop H_inf transfer function and verification deadlines.
"""

from typing import Tuple
import numpy as np

class StringStabilityAnalyzer:
    """Frequency-domain string stability evaluator for CACC platoons."""
    def __init__(
        self,
        kp: float = 0.5,
        kd: float = 1.5,
        ka: float = 1.0,
        tau_actuator: float = 0.10,
        ht: float = 0.60
    ):
        self.kp = kp
        self.kd = kd
        self.ka = ka
        self.tau_a = tau_actuator
        self.ht = ht

    def compute_max_allowable_delay(self, trust_score: float = 1.0) -> float:
        """
        Calculates the theoretical upper bound on total communication + verification latency:
        tau_max = (ht / (ka * T)) - tau_a
        """
        if trust_score <= 1e-4:
            return 999.0  # In pure ACC mode, communication delay is decoupled
        tau_max = (self.ht / (self.ka * trust_score)) - self.tau_a
        return float(tau_max)

    def evaluate_transfer_function(
        self,
        omega_array: np.ndarray,
        tau_total: float,
        trust_score: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluates |Gamma(j*omega)| = |A_i(j*omega) / A_{i-1}(j*omega)| across frequencies.
        Returns the magnitude array and H_inf peak norm.
        """
        s = 1j * omega_array
        
        # Numerator: ka * T * s^2 * e^(-tau*s) + kd * s + kp
        num = self.ka * trust_score * (s**2) * np.exp(-s * tau_total) + self.kd * s + self.kp
        # Denominator: tau_a * s^3 + s^2 + (kd + kp * ht) * s + kp
        den = self.tau_a * (s**3) + (s**2) + (self.kd + self.kp * self.ht) * s + self.kp
        
        gamma = num / den
        mag = np.abs(gamma)
        h_inf_norm = float(np.max(mag))
        
        return mag, h_inf_norm
