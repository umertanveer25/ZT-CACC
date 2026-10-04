import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set clean styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)

y_labels = [
    'Actuator Driveline Lag & Build-up',
    r'Longitudinal Controller ($100\,$Hz)',
    r'Onboard Radar / LiDAR ($100\,$Hz)',
    r'Feedforward Accel $\hat{a}_{i-1}(t)$ (ZOH)',
    r'Trust Metric $T_i(t)$ Evolution',
    'ZT-MVE Verification Engine',
    r'V2X BSM Ingestion ($10\,$Hz)'
]

y_pos = np.arange(len(y_labels))
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
ax.set_xlabel('Timeline (milliseconds)', fontsize=11, fontweight='bold')
ax.set_xlim(-15, 235)
ax.set_ylim(-0.8, len(y_labels) - 0.1)

# Colors
c_bsm = '#1f77b4'
c_mve = '#d62728'
c_trust = '#9467bd'
c_ff = '#ff7f0e'
c_sensor = '#2ca02c'
c_ctrl = '#008b8b'
c_act = '#8c564b'

# Draw horizontal guide lanes
for y in y_pos:
    ax.axhline(y, color='lightgray', linestyle='--', alpha=0.5, zorder=1)

# Highlight BSM period
rect = patches.Rectangle((2.5, -0.65), 100, 7.2, linewidth=1.2, edgecolor='#004085', facecolor='#e6f2ff', alpha=0.35, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(52.5, 6.65, r'$\longleftrightarrow$ One BSM Cycle $\Delta t_{\mathrm{BSM}} = 100\,\mathrm{ms}$ (10 Discrete Control Steps) $\longleftrightarrow$', 
        ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#004085')

# 1. BSM Ingestion (y=6)
for t_arr in [2.5, 102.5, 202.5]:
    ax.scatter(t_arr, 6, color=c_bsm, s=140, zorder=5, marker='s')
    ax.text(t_arr, 6.22, f'BSM Packet\n({t_arr:.1f} ms)', ha='center', va='bottom', fontsize=8, color=c_bsm, fontweight='bold')

# 2. ZT-MVE Verification (y=5)
for t_arr in [2.5, 102.5, 202.5]:
    ax.annotate('', xy=(t_arr, 5.15), xytext=(t_arr, 5.85),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))
    ax.scatter(t_arr, 5, color=c_mve, s=110, zorder=5, marker='o')

ax.text(8, 5.18, r'$\tau_{\mathrm{verify}} = 4.81\,\mu\mathrm{s}$ (TreeSHAP Evaluation)', ha='left', va='bottom', fontsize=8.5, color=c_mve, fontweight='bold')
ax.text(108, 5.18, r'$\tau_{\mathrm{verify}} = 4.81\,\mu\mathrm{s} \ll \tau_{\max}$ ($450\,\mathrm{ms}$ deadline)', ha='left', va='bottom', fontsize=8.5, color=c_mve, fontweight='bold')

# 3. Trust Metric (y=4)
ax.plot([-15, 102.5], [4.15, 4.15], color=c_trust, lw=3, zorder=4)
ax.plot([102.5, 102.5], [4.15, 3.8], color=c_mve, lw=2.5, linestyle=':', zorder=4)
ax.plot([102.5, 235], [3.8, 3.8], color=c_mve, lw=3, zorder=4)
ax.scatter([2.5], [4.15], color=c_trust, s=90, zorder=5)
ax.scatter([102.5], [3.8], color=c_mve, s=90, zorder=5)
ax.text(52.5, 4.25, r'Authentic: $T_i[0] = 1.0$ (Fully Trusted CACC)', ha='center', va='bottom', fontsize=8.5, color=c_trust, fontweight='bold')
ax.text(168, 3.9, r'Attack Flagged: $T_i[1] = 0.08$ ($\lambda_{\mathrm{pen}} = 0.92$, $1$ cycle drop)', ha='center', va='bottom', fontsize=8.5, color=c_mve, fontweight='bold')

# 4. Feedforward Accel ZOH (y=3)
ax.plot([-15, 102.5], [3.0, 3.0], color=c_ff, lw=2.5, zorder=4)
ax.plot([102.5, 102.5], [3.0, 2.65], color='gray', lw=1.5, linestyle=':', zorder=4)
ax.plot([102.5, 235], [2.65, 2.65], color=c_ff, lw=2.5, zorder=4)
ax.text(52.5, 3.12, r'ZOH: $\hat{a}_{i-1}(t) = \tilde{a}_{i-1}[0]$ held across 9 intermediate ticks ($10\,\mathrm{ms}$)', ha='center', va='bottom', fontsize=8.5, color=c_ff, fontweight='bold')
ax.text(168, 2.75, r'Fallback: Attenuated via $(1 - T_i) a_{\mathrm{ACC}} + T_i a_{\mathrm{CACC}}$', ha='center', va='bottom', fontsize=8.5, color='#d95f02', fontweight='bold')

# 5. Onboard Radar/LiDAR (y=2)
ctrl_ticks = np.arange(0, 231, 10)
ax.scatter(ctrl_ticks, np.full_like(ctrl_ticks, 2), color=c_sensor, s=40, zorder=4, marker='^')
ax.text(115, 2.15, r'Continuous Onboard Radar & LiDAR Updates at $100\,\mathrm{Hz}$ ($\Delta t_{\mathrm{sens}} = 10\,\mathrm{ms}$)', ha='center', va='bottom', fontsize=8.5, color=c_sensor, fontweight='bold')

# 6. Longitudinal Controller Scheduling (y=1)
ax.scatter(ctrl_ticks, np.full_like(ctrl_ticks, 1), color=c_ctrl, s=40, zorder=4, marker='d')
ax.text(115, 1.15, r'Discrete Controller Loop Executed Synchronously at $100\,\mathrm{Hz}$ ($\Delta t_{\mathrm{ctrl}} = 10\,\mathrm{ms}$)', ha='center', va='bottom', fontsize=8.5, color=c_ctrl, fontweight='bold')

# Connecting lines between sensor and controller ticks
for t in ctrl_ticks[::2]:
    ax.plot([t, t], [1.08, 1.92], color='teal', linestyle=':', lw=0.8, alpha=0.6)

# 7. Actuator Driveline Response (y=0)
t_fine = np.linspace(-15, 235, 400)
act_resp = np.zeros_like(t_fine)
for idx, t in enumerate(t_fine):
    if t < 102.5:
        act_resp[idx] = 0.0
    else:
        dt = (t - 102.5) / 100.0 # eta_i = 100 ms
        act_resp[idx] = -0.4 * (1.0 - np.exp(-dt))

ax.plot(t_fine, act_resp, color=c_act, lw=2.8, zorder=4)
ax.text(45, 0.08, r'Cruising Equilibrium ($a_i = 0\,\mathrm{m/s}^2$)', ha='center', va='bottom', fontsize=8.5, color=c_act, fontweight='bold')
ax.text(168, -0.42, r'Actuator Inertial Transition: $\eta_i = 100\,\mathrm{ms}, \, \tau_j = 50\,\mathrm{ms}$', ha='center', va='bottom', fontsize=8.5, color=c_act, fontweight='bold')

# Packet loss and stale timeout annotation on right
ax.annotate('Stale Timeout Barrier\n' + r'$\Delta t_{\mathrm{stale}} \geq 250\,\mathrm{ms}$ ($k_{\mathrm{loss}} \geq 3$)' + '\nForce ACC Fallback',
            xy=(202.5, 6), xytext=(195, 4.6),
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.2),
            fontsize=8, color='navy', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='navy', alpha=0.8))

plt.title('Multi-Rate Asynchronous Timing Architecture of ZT-CACC\nCo-scheduling of 10 Hz V2X Ingestion, Microsecond Verification, and 100 Hz Resilient Control', fontsize=11.5, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('Fig1_MultiRate_Timing_Diagram.png', dpi=300)
plt.close()
print('Refined Fig1_MultiRate_Timing_Diagram.png successfully generated!')
