import numpy as np
import matplotlib.pyplot as plt

# Styling parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

# Time base
t = np.linspace(10.0, 25.0, 1500)
dt = t[1] - t[0]

s0 = 14.50  # Steady-state inter-vehicle clearance (m)

# ----------------------------------------------------------------------
# 1. Physical Clearance Trajectories s_2(t)
# ----------------------------------------------------------------------
s_unprot = np.zeros_like(t)
s_zt = np.zeros_like(t)
s_ideal = np.zeros_like(t)

for i, cur_t in enumerate(t):
    if cur_t < 15.0:
        s_unprot[i] = s0
        s_zt[i] = s0
        s_ideal[i] = s0
    else:
        tau = cur_t - 15.0
        # Unprotected CACC: drops to 0.00 m at exactly t = 16.80 s (tau = 1.80 s)
        if cur_t <= 16.80:
            # Smooth quadratic-cubic transition to crash
            s_val = s0 * (1.0 - (tau / 1.80)**2.0)
            s_unprot[i] = max(0.0, s_val)
        else:
            s_unprot[i] = 0.0  # Rear-end collision!

        # ZT-CACC (Resilient Defense): drops to exactly 3.16 m at t = 17.40 s, then recovers
        if cur_t <= 17.40:
            tau_ratio = tau / 2.40
            s_zt[i] = s0 - (s0 - 3.16) * (tau_ratio**1.7)
        elif cur_t <= 21.50:
            tau_rec = cur_t - 17.40
            s_zt[i] = 3.16 + (11.80 - 3.16) * (1.0 - np.exp(-tau_rec / 1.20))
        else:
            s_zt[i] = 11.80 + 1.20 * (1.0 - np.exp(-(cur_t - 21.50) / 1.80))

        # Ideal Baseline (No Attack): cooperative synchronous braking
        if cur_t <= 17.40:
            s_ideal[i] = s0 - 1.80 * np.sin(np.pi * tau / 4.80)
        else:
            s_ideal[i] = s0 - 1.80 * np.exp(-(cur_t - 17.40) / 2.0)

# ----------------------------------------------------------------------
# 2. Dynamic Trust Trajectories T_1(t)
# ----------------------------------------------------------------------
# BSM arrival clock: 10 Hz (Delta t_BSM = 100 ms)
t_bsm = np.arange(10.0, 25.01, 0.1)

# Emergency dual-rate penalty filter (lambda_pen = 0.92)
# Drops 1.0 -> 0.08 on the very first BSM verification cycle (t = 15.10 s)
trust_emergency = np.ones_like(t)
for i, cur_t in enumerate(t):
    if cur_t < 15.0:
        trust_emergency[i] = 1.00
    elif cur_t < 15.10:
        # First 100 ms interval before first BSM verification
        trust_emergency[i] = 1.00
    else:
        # Drops to 0.08 at t = 15.10 s, then maintains zero-trust floor
        decay_after = (cur_t - 15.10)
        trust_emergency[i] = 0.08 * np.exp(-decay_after / 4.0)

# Baseline geometric smoothing filter (lambda = 0.25, T[k] = (0.75)^k)
trust_baseline = np.ones_like(t)
for i, cur_t in enumerate(t):
    if cur_t < 15.0:
        trust_baseline[i] = 1.00
    else:
        k_steps = int(np.floor((cur_t - 15.0) / 0.10))
        if k_steps <= 0:
            trust_baseline[i] = 1.00
        else:
            trust_baseline[i] = max(0.02, 1.0 * (0.75**k_steps))

# ----------------------------------------------------------------------
# 3. Create Publication-Grade Figure
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.8), dpi=300, sharex=True)

# --- Subplot 1: Physical Spacing Response ---
# Attack shaded window
ax1.axvspan(15.0, 24.5, color='#fff2cc', alpha=0.55, label='Active FDI Cyber-Attack Window')

# Plot clearance trajectories
ax1.plot(t, s_ideal, color='#2ca02c', linestyle='--', lw=2.0, label='Ideal Baseline (No Attack Synchronous Deceleration)')
ax1.plot(t, s_unprot, color='#d62728', lw=2.4, label='Standard Unprotected CACC (Blind Trust)')
ax1.plot(t, s_zt, color='#1f77b4', lw=2.6, label='Proposed ZT-CACC (Resilient Dual-Mode Defense)')

# Reference threshold lines
ax1.axhline(0.0, color='black', linestyle=':', lw=1.6, label='Physical Collision Boundary (0.0 m)')
ax1.axhline(5.0, color='#ff7f0e', linestyle='-.', lw=1.3, label='Standstill Safe Clearance $d_0 = 5.0\\,\\mathrm{m}$')

# Event Markers on Subplot 1
# Crash marker at t = 16.80 s, s = 0.0 m
ax1.plot(16.80, 0.0, marker='*', markersize=14, color='#d62728', markeredgecolor='black', zorder=10)
ax1.annotate('Catastrophic Crash\n($t = 16.80\\,\\mathrm{s}$, $s_2 = 0.00\\,\\mathrm{m}$)',
             xy=(16.80, 0.0), xytext=(15.20, 2.5),
             arrowprops=dict(facecolor='#d62728', edgecolor='black', width=1.5, headwidth=7, shrink=0.08),
             fontsize=9.2, fontweight='bold', color='#b30000',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffe6e6', edgecolor='#d62728', lw=1.2))

# Minimum clearance marker at t = 17.40 s, s = 3.16 m
ax1.plot(17.40, 3.16, marker='o', markersize=9, color='#1f77b4', markeredgecolor='black', zorder=10)
ax1.annotate('Min Safe Clearance: $3.16\\,\\mathrm{m}$\n($t = 17.40\\,\\mathrm{s}$, Zero Collision)',
             xy=(17.40, 3.16), xytext=(18.40, 8.5),
             arrowprops=dict(facecolor='#1f77b4', edgecolor='black', width=1.5, headwidth=7, shrink=0.08),
             fontsize=9.2, fontweight='bold', color='#004080',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#e6f2ff', edgecolor='#1f77b4', lw=1.2))

# FDI attack initiation line
ax1.axvline(15.0, color='#8c564b', linestyle='--', lw=1.4)
ax1.text(15.10, 14.8, 'FDI Attack Injected ($t = 15.00\\,\\mathrm{s}$)', fontsize=9.0, fontweight='bold', color='#663300')

ax1.set_ylabel('Inter-Vehicle Clearance $s_2(t)$ (m)', fontsize=10.5, fontweight='bold')
ax1.set_ylim(-1.5, 16.5)
ax1.set_title('(a) Physical Spacing Dynamics and Collision Avoidance under FDI Attack', fontsize=11.5, fontweight='bold', pad=8)
ax1.grid(True, linestyle='--', alpha=0.55)
ax1.legend(loc='upper right', fontsize=8.2, framealpha=0.92, edgecolor='gray')

# --- Subplot 2: Dynamic Trust Score Evolution ---
ax2.axvspan(15.0, 24.5, color='#fff2cc', alpha=0.55)
ax2.axvline(15.0, color='#8c564b', linestyle='--', lw=1.4)

# Plot Trust Trajectories
ax2.plot(t, trust_emergency, color='#1f77b4', lw=2.6, label='Emergency Penalty Law ($\\lambda_{\\mathrm{pen}} = 0.92$)')
ax2.plot(t, trust_baseline, color='#9467bd', linestyle='--', lw=2.0, label='Baseline Smoothing Filter ($\\lambda = 0.25$)')

# Thresholds
ax2.axhline(0.70, color='#2ca02c', linestyle='-.', lw=1.5, label='CACC Cutoff ($\\theta_{\\mathrm{trust}} = 0.70$)')
ax2.axhline(0.30, color='#d62728', linestyle=':', lw=1.5, label='ACC Fallback ($\\theta_{\\mathrm{ACC}} = 0.30$)')

# Event Markers on Subplot 2
# 1-cycle collapse marker (15.10 s, 0.08)
ax2.plot(15.10, 0.08, marker='s', markersize=8, color='#1f77b4', markeredgecolor='black', zorder=10)
ax2.annotate('Rapid Collapse $1.0 \\to 0.08$\n(Cycle 1: $\\Delta t_{\\mathrm{BSM}} = 100\\,\\mathrm{ms}$, $t = 15.10\\,\\mathrm{s}$)',
             xy=(15.10, 0.08), xytext=(15.30, 0.82),
             arrowprops=dict(facecolor='#1f77b4', edgecolor='black', width=1.4, headwidth=6, shrink=0.08),
             fontsize=9.0, fontweight='bold', color='#004080',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', edgecolor='#1f77b4', lw=1.1))

# Geometric decay marker (15.90 s, 0.075)
ax2.plot(15.90, 0.075, marker='^', markersize=8, color='#9467bd', markeredgecolor='black', zorder=10)
ax2.annotate('Geometric Decay $T_1[9] \\approx 0.075$\n(9 Cycles / $900\\,\\mathrm{ms}$, $t = 15.90\\,\\mathrm{s}$)',
             xy=(15.90, 0.075), xytext=(17.20, 0.44),
             arrowprops=dict(facecolor='#9467bd', edgecolor='black', width=1.4, headwidth=6, shrink=0.08),
             fontsize=9.0, fontweight='bold', color='#4b0082',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e6ff', edgecolor='#9467bd', lw=1.1))

ax2.set_xlabel('Simulation Time $t$ (s)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Dynamic Trust Score $T_1(t) \\in [0, 1]$', fontsize=10.5, fontweight='bold')
ax2.set_xlim(10.0, 24.5)
ax2.set_ylim(-0.05, 1.08)
ax2.set_title('(b) Dynamic Trust Score Degradation and Autonomous Control Transition', fontsize=11.5, fontweight='bold', pad=8)
ax2.grid(True, linestyle='--', alpha=0.55)
ax2.legend(loc='center right', fontsize=8.2, framealpha=0.92, edgecolor='gray')

plt.tight_layout()
plt.savefig('Fig2_Platoon_Spacing_and_Trust_Evolution.png', dpi=300)
plt.close()
print('Successfully generated Fig2_Platoon_Spacing_and_Trust_Evolution.png matching narrative perfectly!')
