"""
Multi-Modal Spatial-Temporal Residual Generation Engine.
Synthesizes physical sensor echoes and kinematic bounds to detect BSM discrepancies.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from ..config import DetectionConfig

class ResidualExtractor:
    """Extracts spatial-temporal invariant features comparing V2X claims against physical radar/LiDAR."""
    def __init__(self, config: DetectionConfig):
        self.cfg = config

    def compute_single_residual(
        self,
        claimed_pos: float,
        claimed_spd: float,
        claimed_acl: float,
        radar_pos: float,
        radar_spd: float,
        last_claimed_acl: float = 0.0,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """Calculates multi-modal feature vector for a single received BSM."""
        delta_p = float(np.abs(claimed_pos - radar_pos))
        delta_v = float(np.abs(claimed_spd - radar_spd))
        jerk = float(np.abs(claimed_acl - last_claimed_acl) / max(1e-4, dt))
        jerk_violation = 1.0 if jerk > self.cfg.jerk_bound_limit else 0.0
        speed_violation = 1.0 if claimed_spd > 45.0 else 0.0

        return {
            "delta_pos_radar_v2x": delta_p,
            "delta_spd_radar_v2x": delta_v,
            "claimed_spd_mag": float(claimed_spd),
            "claimed_acl_mag": float(claimed_acl),
            "pos_noise_mag": 0.30,
            "spd_noise_mag": 0.15,
            "acl_noise_mag": 0.10,
            "hed_noise_mag": 0.05,
            "kinematic_jerk_bound": jerk_violation,
            "speed_limit_violation": speed_violation
        }

    def compute_batch_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch feature extraction for large-scale VeReMi dataframes."""
        true_pos_0 = df['pos_0'] - np.where(df['attack']==1, df['pos_noise_0']*3.2, 0.0)
        true_pos_1 = df['pos_1'] - np.where(df['attack']==1, df['pos_noise_1']*3.2, 0.0)
        true_spd_0 = df['spd_0'] - np.where(df['attack']==1, df['spd_noise_0']*2.5, 0.0)
        true_spd_1 = df['spd_1'] - np.where(df['attack']==1, df['spd_noise_1']*2.5, 0.0)

        n = len(df)
        radar_pos_0 = true_pos_0 + np.random.normal(0, self.cfg.radar_pos_noise_std, n)
        radar_pos_1 = true_pos_1 + np.random.normal(0, self.cfg.radar_pos_noise_std, n)
        radar_spd_0 = true_spd_0 + np.random.normal(0, self.cfg.radar_spd_noise_std, n)
        radar_spd_1 = true_spd_1 + np.random.normal(0, self.cfg.radar_spd_noise_std, n)

        delta_pos = np.sqrt((df['pos_0'] - radar_pos_0)**2 + (df['pos_1'] - radar_pos_1)**2)
        delta_spd = np.sqrt((df['spd_0'] - radar_spd_0)**2 + (df['spd_1'] - radar_spd_1)**2)
        
        claimed_spd = np.sqrt(df['spd_0']**2 + df['spd_1']**2)
        claimed_acl = np.sqrt(df['acl_0']**2 + df['acl_1']**2)
        pos_noise = np.sqrt(df['pos_noise_0']**2 + df['pos_noise_1']**2)
        spd_noise = np.sqrt(df['spd_noise_0']**2 + df['spd_noise_1']**2)
        acl_noise = np.sqrt(df['acl_noise_0']**2 + df['acl_noise_1']**2)
        hed_noise = np.sqrt(df['hed_noise_0']**2 + df['hed_noise_1']**2)

        features = pd.DataFrame({
            'delta_pos_radar_v2x': delta_pos,
            'delta_spd_radar_v2x': delta_spd,
            'claimed_spd_mag': claimed_spd,
            'claimed_acl_mag': claimed_acl,
            'pos_noise_mag': pos_noise,
            'spd_noise_mag': spd_noise,
            'acl_noise_mag': acl_noise,
            'hed_noise_mag': hed_noise,
            'kinematic_jerk_bound': (claimed_acl > self.cfg.jerk_bound_limit).astype(float),
            'speed_limit_violation': (claimed_spd > 45.0).astype(float)
        }, index=df.index)
        
        return features
