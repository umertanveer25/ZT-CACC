import unittest
from src.zero_trust_cav.config import CACCConfig, DetectionConfig
from src.zero_trust_cav.core.controller import ResilientCACCController, TrustManager

class TestCACCController(unittest.TestCase):
    def test_trust_manager_decay_and_recovery(self):
        det_cfg = DetectionConfig(alpha_decay=0.35, beta_recovery=0.05, threat_threshold=0.50)
        tm = TrustManager(det_cfg, initial_trust=1.0)
        
        # Severe threat -> should drop trust
        t1 = tm.update(anomaly_probability=0.95)
        self.assertLess(t1, 1.0)
        self.assertAlmostEqual(t1, 1.0 - 0.35 * 0.95, places=3)
        
        # Benign packet -> should recover trust slightly
        t2 = tm.update(anomaly_probability=0.01)
        self.assertGreater(t2, t1)

    def test_cacc_controller_output(self):
        cacc_cfg = CACCConfig(kp=1.0, kd=2.0, ka=1.0)
        det_cfg = DetectionConfig()
        controller = ResilientCACCController(cacc_cfg, det_cfg)
        
        # Nominal equilibrium state
        u, trust = controller.compute_control(
            pos_self=980.0,
            spd_self=25.0,
            pos_lead=1004.5,
            spd_lead=25.0,
            claimed_lead_accel=0.0,
            anomaly_probability=0.01,
            vehicle_length=4.5,
            standstill_gap=5.0,
            nominal_ht=0.60
        )
        self.assertAlmostEqual(u, 0.0, delta=0.05)
        self.assertEqual(trust, 1.0)

if __name__ == "__main__":
    unittest.main()
