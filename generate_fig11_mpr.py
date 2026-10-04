import numpy as np
import matplotlib.pyplot as plt

# Styling parameters consistent with IEEE Transactions figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

# Exact data from Table IX in manuscript
mpr = np.array([0, 25, 50, 75, 100])
flow = np.array([1850, 2210, 2680, 3140, 3544])
speed = np.array([84.2, 91.5, 98.4, 103.8, 108.0])
min_ttc = np.array([1.42, 2.15, 2.84, 3.42, 4.18])

# 95% Confidence intervals over simulation seeds
flow_ci = np.array([32, 28, 25, 22, 18])
ttc_ci = np.array([0.08, 0.07, 0.06, 0.05, 0.04])
speed_ci = np.array([1.2, 1.0, 0.8, 0.7, 0.5])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

# -------------------------------------------------------------------------
# Subplot (a): Highway Lane Capacity vs. Market Penetration Rate (MPR)
# -------------------------------------------------------------------------
ax1.plot(mpr, flow, color='#1f77b4', marker='o', markersize=7, linewidth=2.2,
         label='Lane Capacity (veh/h/lane)')
ax1.fill_between(mpr, flow - flow_ci, flow + flow_ci, color='#1f77b4', alpha=0.18,
                 label='95% Confidence Interval')

# Data callouts
for x_val, y_val in zip(mpr, flow):
    offset_y = 65 if x_val != 100 else -95
    ax1.annotate(f'{y_val} veh/h', xy=(x_val, y_val), xytext=(x_val - 4, y_val + offset_y),
                 fontsize=8.5, fontweight='bold', color='#0d47a1')

ax1.set_xlabel('CAV Market Penetration Rate MPR (%)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Lane Throughput (veh/h/lane)', fontsize=10, fontweight='bold')
ax1.set_xlim(-4, 106)
ax1.set_ylim(1700, 3750)
ax1.set_xticks(mpr)
ax1.set_xticklabels([f'{x}%' for x in mpr], fontsize=9.5, fontweight='bold')
ax1.set_title('(a) Highway Lane Capacity vs. CAV Penetration', fontsize=10.5, fontweight='bold', pad=8)
ax1.grid(True, linestyle='--', alpha=0.55)
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.92)

# -------------------------------------------------------------------------
# Subplot (b): Safety Margin (Min TTC) and Mean Cruising Speed
# -------------------------------------------------------------------------
color_ttc = '#2ca02c'
color_speed = '#d62728'

# Primary axis: Minimum Time-to-Collision (TTC)
line1 = ax2.plot(mpr, min_ttc, color=color_ttc, marker='s', markersize=7, linewidth=2.2,
                 label='Min Time-to-Collision (s)')
ax2.fill_between(mpr, min_ttc - ttc_ci, min_ttc + ttc_ci, color=color_ttc, alpha=0.18)

# Safety threshold line (standard critical TTC threshold = 1.5 s)
ax2.axhline(1.50, color='#7f7f7f', linestyle=':', linewidth=1.4, label='Critical Safety Limit (1.5 s)')

for x_val, y_val in zip(mpr, min_ttc):
    ax2.annotate(f'{y_val:.2f} s', xy=(x_val, y_val), xytext=(x_val - 3, y_val + 0.14),
                 fontsize=8.5, fontweight='bold', color='#1b5e20')

ax2.set_xlabel('CAV Market Penetration Rate MPR (%)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Minimum Time-to-Collision TTC (s)', fontsize=10, fontweight='bold', color=color_ttc)
ax2.set_xlim(-4, 106)
ax2.set_ylim(1.0, 4.65)
ax2.set_xticks(mpr)
ax2.set_xticklabels([f'{x}%' for x in mpr], fontsize=9.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_ttc)
ax2.grid(True, linestyle='--', alpha=0.55)

# Secondary twin axis: Mean Corridor Speed (km/h)
ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(mpr, speed, color=color_speed, marker='^', markersize=7, linewidth=2.0,
                      linestyle='--', label='Mean Speed (km/h)')
ax2_twin.fill_between(mpr, speed - speed_ci, speed + speed_ci, color=color_speed, alpha=0.15)
ax2_twin.set_ylabel('Mean Cruising Speed (km/h)', fontsize=10, fontweight='bold', color=color_speed)
ax2_twin.set_ylim(80.0, 115.0)
ax2_twin.tick_params(axis='y', labelcolor=color_speed)

for x_val, y_val in zip(mpr, speed):
    ax2_twin.annotate(f'{y_val:.1f}', xy=(x_val, y_val), xytext=(x_val + 1.2, y_val - 2.2),
                      fontsize=8.0, fontweight='bold', color='#b71c1c')

# Combined legend for ax2
lines_all = line1 + line2 + [plt.Line2D([0], [0], color='#7f7f7f', linestyle=':', linewidth=1.4)]
labels_all = ['Min Time-to-Collision (s)', 'Mean Speed (km/h)', 'Critical Safety Limit (1.5 s)']
ax2.legend(lines_all, labels_all, loc='center left', fontsize=8.2, framealpha=0.92)
ax2.set_title('(b) Minimum TTC Safety Margin & Cruising Speed', fontsize=10.5, fontweight='bold', pad=8)

plt.tight_layout()
output_path = 'Fig11_Mixed_Traffic_MPR_Throughput_and_Safety.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Successfully saved figure to {output_path}')
