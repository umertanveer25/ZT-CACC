import unittest
from src.zero_trust_cav.config import MultiRATConfig
from src.zero_trust_cav.core.multi_rat import MultiRATBroker, RATType

class TestMultiRATBroker(unittest.TestCase):
    def test_multi_rat_switching(self):
        cfg = MultiRATConfig(
            vlc_delay_ms=1.8,
            its_g5_delay_ms=8.4,
            lte_v2x_delay_ms=14.2,
            glare_start_s=10.0,
            glare_end_s=20.0,
            jamming_start_s=15.0,
            jamming_end_s=25.0
        )
        broker = MultiRATBroker(cfg)
        
        # t = 5.0s: Nominal -> VLC (1.8 ms)
        rat, delay = broker.get_active_rat(5.0)
        self.assertEqual(rat, RATType.VLC)
        self.assertAlmostEqual(delay, 0.0018, places=5)
        
        # t = 12.0s: Glare active, no RF jamming -> ITS-G5 (8.4 ms)
        rat, delay = broker.get_active_rat(12.0)
        self.assertEqual(rat, RATType.ITS_G5)
        self.assertAlmostEqual(delay, 0.0084, places=5)
        
        # t = 18.0s: Glare + Jamming active -> LTE-V2X (14.2 ms)
        rat, delay = broker.get_active_rat(18.0)
        self.assertEqual(rat, RATType.LTE_V2X)
        self.assertAlmostEqual(delay, 0.0142, places=5)

if __name__ == "__main__":
    unittest.main()
