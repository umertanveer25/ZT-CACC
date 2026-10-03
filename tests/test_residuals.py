import unittest
import numpy as np
from src.zero_trust_cav.config import DetectionConfig
from src.zero_trust_cav.detection.residuals import ResidualExtractor

class TestResidualExtractor(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectionConfig(jerk_bound_limit=5.5)
        self.extractor = ResidualExtractor(self.cfg)

    def test_single_residual_benign(self):
        res = self.extractor.compute_single_residual(
            claimed_pos=100.0,
            claimed_spd=25.0,
            claimed_acl=0.50,
            radar_pos=100.1,
            radar_spd=24.95,
            last_claimed_acl=0.48, # jerk = 0.02 / 0.01 = 2.0 < 5.5
            dt=0.01
        )
        self.assertAlmostEqual(res["delta_pos_radar_v2x"], 0.1, places=3)
        self.assertAlmostEqual(res["delta_spd_radar_v2x"], 0.05, places=3)
        self.assertEqual(res["kinematic_jerk_bound"], 0.0)

    def test_single_residual_attack(self):
        res = self.extractor.compute_single_residual(
            claimed_pos=100.0,
            claimed_spd=25.0,
            claimed_acl=-6.0,
            radar_pos=90.0,
            radar_spd=20.0,
            last_claimed_acl=1.0, # jerk = 7.0 / 0.01 = 700.0 > 5.5
            dt=0.01
        )
        self.assertAlmostEqual(res["delta_pos_radar_v2x"], 10.0, places=3)
        self.assertAlmostEqual(res["delta_spd_radar_v2x"], 5.0, places=3)
        self.assertEqual(res["kinematic_jerk_bound"], 1.0)

if __name__ == "__main__":
    unittest.main()
