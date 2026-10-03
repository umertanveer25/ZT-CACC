import unittest
import numpy as np
from src.zero_trust_cav.simulation.stability import StringStabilityAnalyzer

class TestStringStability(unittest.TestCase):
    def test_max_allowable_delay(self):
        analyzer = StringStabilityAnalyzer(kp=0.5, kd=1.5, ka=1.0, tau_actuator=0.10, ht=0.60)
        tau_max = analyzer.compute_max_allowable_delay(trust_score=1.0)
        # tau_max = 0.60 / (1.0 * 1.0) - 0.10 = 0.50 s (500 ms)
        self.assertAlmostEqual(tau_max, 0.50, places=3)

    def test_h_infinity_transfer_function(self):
        analyzer = StringStabilityAnalyzer(kp=0.5, kd=1.5, ka=1.0, tau_actuator=0.10, ht=0.60)
        omega = np.linspace(0.01, 10.0, 500)
        
        # Nominal CACC: delay = 14.2 ms (within 500 ms limit) -> peak magnitude bounded
        _, h_inf_cacc = analyzer.evaluate_transfer_function(omega, tau_total=0.0142, trust_score=1.0)
        self.assertLessEqual(h_inf_cacc, 1.20)

        # Degraded ACC Mode (trust=0.0) -> strictly attenuating
        _, h_inf_acc = analyzer.evaluate_transfer_function(omega, tau_total=0.0, trust_score=0.0)
        self.assertLessEqual(h_inf_acc, 1.001)

if __name__ == "__main__":
    unittest.main()
