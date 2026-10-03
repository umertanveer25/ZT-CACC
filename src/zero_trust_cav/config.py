"""
Centralized Configuration Module using Python Dataclasses and YAML serialization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass
class SimulationConfig:
    dt: float = 0.01                     # Simulation timestep (seconds)
    total_time: float = 40.0             # Duration of simulation (seconds)
    num_vehicles: int = 8                # Platoon size
    v_target: float = 25.0               # Nominal cruise speed (m/s)
    d0: float = 5.0                      # Standstill gap (m)
    ht: float = 0.60                     # Time headway (s)
    vehicle_length: float = 4.5          # Length of vehicle (m)
    random_seed: int = 42

@dataclass
class MultiRATConfig:
    # Delays in milliseconds
    vlc_delay_ms: float = 1.80
    its_g5_delay_ms: float = 8.40
    lte_v2x_delay_ms: float = 14.20
    # Failure intervals (seconds)
    glare_start_s: float = 12.0
    glare_end_s: float = 24.0
    jamming_start_s: float = 18.0
    jamming_end_s: float = 28.0

@dataclass
class CACCConfig:
    kp: float = 1.0                      # Spacing proportional gain
    kd: float = 2.0                      # Velocity damping gain
    ka: float = 1.0                      # Feedforward acceleration gain
    tau_actuator: float = 0.10           # Driveline actuator time constant (s)
    a_max: float = 3.0                   # Max acceleration (m/s^2)
    a_min: float = -6.0                  # Max braking deceleration (m/s^2)
    acc_fallback_ht: float = 1.20        # Safe headway for pure ACC fallback (s)

@dataclass
class DetectionConfig:
    alpha_decay: float = 0.35            # Trust degradation rate on attack detection
    beta_recovery: float = 0.05          # Trust recovery rate under valid packets
    threat_threshold: float = 0.50       # Anomaly probability threshold for attack classification
    radar_pos_noise_std: float = 0.30    # Radar position error standard deviation (m)
    radar_spd_noise_std: float = 0.15    # Radar Doppler speed error standard deviation (m/s)
    jerk_bound_limit: float = 5.5        # Max plausible jerk threshold (m/s^3)

@dataclass
class AttackConfig:
    target_vehicle_idx: int = 2          # Vehicle index under cyber-attack
    attack_start_s: float = 15.0         # Attack initiation timestamp
    attack_end_s: float = 26.0           # Attack termination timestamp
    bogus_accel: float = -6.0            # Claimed bogus emergency deceleration (m/s^2)
    speed_drift: float = 12.0            # Claimed speed offset (m/s)
    pos_offset: float = 8.0              # Claimed position bias (m)

@dataclass
class AppConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    multi_rat: MultiRATConfig = field(default_factory=MultiRATConfig)
    cacc: CACCConfig = field(default_factory=CACCConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls(
            sim=SimulationConfig(**data.get('sim', {})),
            multi_rat=MultiRATConfig(**data.get('multi_rat', {})),
            cacc=CACCConfig(**data.get('cacc', {})),
            detection=DetectionConfig(**data.get('detection', {})),
            attack=AttackConfig(**data.get('attack', {}))
        )

    def to_yaml(self, path: str) -> None:
        data = {
            'sim': self.sim.__dict__,
            'multi_rat': self.multi_rat.__dict__,
            'cacc': self.cacc.__dict__,
            'detection': self.detection.__dict__,
            'attack': self.attack.__dict__
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
