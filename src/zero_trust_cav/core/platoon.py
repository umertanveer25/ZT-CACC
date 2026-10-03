"""
Vehicle Kinematics and Platoon State Dynamics.
Models third-order driveline actuator lag and kinematic integration.
"""

from typing import List, Dict
import numpy as np
from ..config import SimulationConfig, CACCConfig

class Vehicle:
    """Individual vehicle state representation with third-order actuator lag."""
    def __init__(self, vehicle_id: int, initial_pos: float, initial_spd: float, tau_actuator: float = 0.10):
        self.id = vehicle_id
        self.p = initial_pos          # Position (m)
        self.v = initial_spd          # Speed (m/s)
        self.a = 0.0                  # Actual acceleration (m/s^2)
        self.u = 0.0                  # Commanded acceleration (m/s^2)
        self.tau = tau_actuator       # Driveline actuator time constant (s)

    def step(self, commanded_accel: float, dt: float, a_min: float = -6.0, a_max: float = 3.0):
        """Integrates vehicle states using third-order dynamic driveline model."""
        self.u = float(np.clip(commanded_accel, a_min, a_max))
        # Actuator lag: dot(a) = -(1/tau)*a + (1/tau)*u
        dot_a = (-self.a + self.u) / self.tau
        self.a += dot_a * dt
        self.v += self.a * dt
        self.p += self.v * dt

class Platoon:
    """Homogeneous/Heterogeneous Platoon Container managing kinematic state histories."""
    def __init__(self, sim_config: SimulationConfig, cacc_config: CACCConfig):
        self.sim_cfg = sim_config
        self.cacc_cfg = cacc_config
        self.vehicles: List[Vehicle] = []
        self._initialize_platoon()

    def _initialize_platoon(self):
        """Initializes platoon positions based on nominal constant time headway."""
        nominal_gap = self.sim_cfg.d0 + self.sim_cfg.ht * self.sim_cfg.v_target
        spacing = self.sim_cfg.vehicle_length + nominal_gap
        
        for i in range(self.sim_cfg.num_vehicles):
            p0 = 1000.0 - i * spacing
            v0 = self.sim_cfg.v_target
            self.vehicles.append(Vehicle(i, p0, v0, self.cacc_cfg.tau_actuator))

    def get_spacings(self) -> np.ndarray:
        """Returns instantaneous physical bumper-to-bumper gaps between adjacent vehicles."""
        gaps = []
        for i in range(1, len(self.vehicles)):
            gap = self.vehicles[i-1].p - self.vehicles[i].p - self.sim_cfg.vehicle_length
            gaps.append(gap)
        return np.array(gaps)

    def get_positions(self) -> np.ndarray:
        return np.array([v.p for v in self.vehicles])

    def get_velocities(self) -> np.ndarray:
        return np.array([v.v for v in self.vehicles])

    def get_accelerations(self) -> np.ndarray:
        return np.array([v.a for v in self.vehicles])
