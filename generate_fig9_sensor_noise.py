import numpy as np
import matplotlib.pyplot as plt

# Styling parameters consistent with IEEE Transactions figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

# Data points from Table VI
sigmas = np.array([0.10, 0.50, 1.00, 2.00, 2.50])
acc = np.array([99.77, 99.12, 97.84, 94.65, 92.10])
f1 = np.array([99.77, 99.10, 97.80, 94.58, 92.04])
fpr = np.array([0.08, 0.34, 0.82, 2.15, 3.40])
fnr = np.array([0.38, 1.42, 3.50, 8.55, 12.40])
rmse = np.array([1.16, 1.24, 1.41, 1.72, 1.98])
roc_auc = np.array([0.9998, 0.9991, 0.9975, 0.9912, 0.9854])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

# -------------------------------------------------------------------------
# Subplot (a): Classification Metric Retention under Sensor Range Noise
# -------------------------------------------------------------------------
# Background shaded regimes for sensor noise levels
ax1.axvspan(0.05, 0.30, color='#2ca02c', alpha=0.10, label=r'Nominal Noise ($\sigma \leq 0.3\,\mathrm{m}$)')
ax1.axvspan(0.30, 1.50, color='#ff7f0e', alpha=0.10, label=r'Moderate Noise ($0.3 < \sigma \leq 1.5\,\mathrm{m}$)')
ax1.axvspan(1.50, 2.65, color='#d62728', alpha=0.10, label=r'Elevated Noise ($\sigma > 1.5\,\mathrm{m}$)')

# Curves
ax1.plot(sigmas, acc, color='#008040', marker='s', markersize=6, linewidth=2.0, label='Accuracy (%)')
ax1.plot(sigmas, f1, color='#1f4090', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=r'$F_1$-Score (%)')

# Value annotations for key points
ax1.annotate(f'{acc[0]:.2f}%', xy=(sigmas[0], acc[0]), xytext=(sigmas[0]+0.05, acc[0]-0.8),
             fontsize=8, fontweight='bold', color='#008040')
ax1.annotate(f'{acc[2]:.2f}%', xy=(sigmas[2], acc[2]), xytext=(sigmas[2]-0.15, acc[2]-1.6),
             fontsize=8, fontweight='bold', color='#008040')
ax1.annotate(f'{acc[-1]:.2f}%', xy=(sigmas[-1], acc[-1]), xytext=(sigmas[-1]-0.35, acc[-1]+1.0),
             fontsize=8, fontweight='bold', color='#008040')

ax1.set_xlim(0.05, 2.65)
ax1.set_ylim(88.0, 101.5)
ax1.set_xlabel(r'Sensor Range Noise Std. Dev. $\sigma$ (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Performance Score (%)', fontsize=10, fontweight='bold')
ax1.set_title('(a) Robustness Under Sensor-Noise Sweep', fontsize=11, fontweight='bold', pad=8)
ax1.grid(True, linestyle='--', alpha=0.55)
ax1.legend(loc='lower left', fontsize=8.2, framealpha=0.92)

# -------------------------------------------------------------------------
# Subplot (b): Area Under ROC Curve & Invariant Error Rates
# -------------------------------------------------------------------------
color_roc = '#6f2dbd'
ax2.plot(sigmas, roc_auc, color=color_roc, marker='^', markersize=7, linewidth=2.2, label='ROC-AUC Score')
ax2.set_xlabel(r'Sensor Range Noise Std. Dev. $\sigma$ (m)', fontsize=10, fontweight='bold')
ax2.set_ylabel('ROC-AUC Score', fontsize=10, fontweight='bold', color=color_roc)
ax2.set_ylim(0.965, 1.002)
ax2.set_xlim(0.05, 2.65)
ax2.tick_params(axis='y', labelcolor=color_roc)
ax2.grid(True, linestyle='--', alpha=0.55)

# Annotate ROC-AUC endpoints
ax2.annotate(f'{roc_auc[0]:.4f}', xy=(sigmas[0], roc_auc[0]), xytext=(sigmas[0]+0.05, roc_auc[0]-0.003),
             fontsize=8.5, fontweight='bold', color=color_roc)
ax2.annotate(f'{roc_auc[-1]:.4f}', xy=(sigmas[-1], roc_auc[-1]), xytext=(sigmas[-1]-0.35, roc_auc[-1]+0.003),
             fontsize=8.5, fontweight='bold', color=color_roc)

# Secondary twin axis for False Alarm (FPR) and Miss (FNR) percentages
ax2_twin = ax2.twinx()
ax2_twin.plot(sigmas, fpr, color='#d62728', marker='x', markersize=6, linewidth=1.8, linestyle=':', label='FPR (%)')
ax2_twin.plot(sigmas, fnr, color='#ff7f0e', marker='v', markersize=6, linewidth=1.8, linestyle='-.', label='FNR (%)')
ax2_twin.set_ylabel('Error Rates (%)', fontsize=10, fontweight='bold', color='#b22222')
ax2_twin.set_ylim(-0.5, 15.0)
ax2_twin.tick_params(axis='y', labelcolor='#b22222')

# Combine legends for ax2 and ax2_twin
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize=8.2, framealpha=0.92)
ax2.set_title('(b) Discriminatory Power & Error Probabilities', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout()
output_path = 'Fig9_Adverse_Weather_Noise_Stress_Test.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Successfully saved figure to {output_path}')
