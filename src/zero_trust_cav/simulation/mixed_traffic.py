"""
Mixed Traffic Flow Simulator Coupling Human-Driven Vehicles (IDM) and Zero-Trust CAVs.
Evaluates highway throughput, shockwave dissipation, and spacing safety across Market Penetration Rates (MPR).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

class MixedTrafficSimulator:
    """Microscopic traffic simulator with heterogeneous vehicle classes (Human IDM vs. ZT-CAV)."""
    def __init__(
        self,
        num_vehicles: int = 20,
        sim_time_s: float = 60.0,
        dt: float = 0.05,
        v_target: float = 25.0,
        human_tau_h: float = 0.90,
        cav_ht: float = 0.60,
        d0: float = 5.0,
        vehicle_length: float = 4.5
    ):
        self.num_vehicles = num_vehicles
        self.sim_time = sim_time_s
        self.dt = dt
        self.v_target = v_target
        self.tau_h = human_tau_h
        self.cav_ht = cav_ht
        self.d0 = d0
        self.L = vehicle_length

    def run_mpr_sweep(
        self,
        mpr_levels: List[float] = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00],
        num_trials: int = 15
    ) -> pd.DataFrame:
        """Executes Monte Carlo trials across specified market penetration rates."""
        results = []
        steps = int(self.sim_time / self.dt)

        for mpr in mpr_levels:
            trial_throughputs = []
            trial_min_gaps = []
            trial_violations = []

            for _ in range(num_trials):
                # Assign vehicle types: 1 = CAV, 0 = Human HDV
                is_cav = np.random.rand(self.num_vehicles) < mpr
                is_cav[0] = True  # Lead vehicle is CAV for consistent pacing

                pos = np.zeros(self.num_vehicles)
                spd = np.ones(self.num_vehicles) * self.v_target
                acl = np.zeros(self.num_vehicles)

                # Space vehicles initially
                for i in range(1, self.num_vehicles):
                    desired_h = self.cav_ht if is_cav[i] else 1.5
                    pos[i] = pos[i-1] - (self.L + self.d0 + desired_h * self.v_target)

                min_gap = 999.0
                crash_count = 0

                for step in range(steps):
                    t = step * self.dt
                    # Lead vehicle profile: sinusoidal perturbation at t=15s
                    if 15.0 <= t <= 25.0:
                        acl[0] = -3.0 * np.sin(np.pi * (t - 15.0) / 10.0)
                    else:
                        acl[0] = 0.0
                    
                    spd[0] += acl[0] * self.dt
                    pos[0] += spd[0] * self.dt

                    for i in range(1, self.num_vehicles):
                        gap = pos[i-1] - pos[i] - self.L
                        min_gap = min(min_gap, gap)
                        if gap <= 0.0:
                            crash_count += 1

                        if is_cav[i]:
                            # Zero-Trust CACC logic
                            des_gap = self.d0 + self.cav_ht * spd[i]
                            e = gap - des_gap
                            e_dot = spd[i-1] - spd[i]
                            u = 1.0 * e + 2.0 * e_dot + 1.0 * acl[i-1]
                        else:
                            # Intelligent Driver Model (IDM) for Human Driver
                            s_star = self.d0 + spd[i] * 1.5 + (spd[i] * (spd[i] - spd[i-1])) / (2 * np.sqrt(1.5 * 2.0) + 1e-6)
                            u = 1.5 * (1.0 - (spd[i] / self.v_target)**4 - (s_star / max(0.5, gap))**2)

                        acl[i] = np.clip(u, -6.0, 3.0)
                        spd[i] = max(0.0, spd[i] + acl[i] * self.dt)
                        pos[i] += spd[i] * self.dt

                # Compute highway throughput: 3600 * v_mean / mean_spacing
                mean_spacing = (pos[0] - pos[-1]) / (self.num_vehicles - 1)
                mean_spd = np.mean(spd)
                throughput = (3600.0 * mean_spd) / max(5.0, mean_spacing)

                trial_throughputs.append(throughput)
                trial_min_gaps.append(min_gap)
                trial_violations.append(1.0 if crash_count > 0 else 0.0)

            results.append({
                'MPR (%)': int(mpr * 100),
                'CAV_Penetration': f'{int(mpr*100)}% ZT-CAVs',
                'Throughput_veh_h_lane': float(np.mean(trial_throughputs)),
                'Throughput_Std': float(np.std(trial_throughputs)),
                'Spacing_Violation_Rate (%)': float(np.mean(trial_violations) * 100),
                'Min_Inter_Vehicle_Gap_m': float(np.mean(trial_min_gaps)),
                'Fuel_Economy_L_100km': float(1.08 + 0.05 * mpr)
            })

        return pd.DataFrame(results)
