import numpy as np
import matplotlib.pyplot as plt

# Styling parameters consistent with IEEE Transactions figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

# -------------------------------------------------------------------------
# Subplot (a): Defense Against Colluding Byzantine Attackers
# (Directly matches Table VII in manuscript)
# -------------------------------------------------------------------------
m_labels = ['M=1 Attacker', 'M=2 Colluders', 'M=3 Colluders']
x = np.arange(len(m_labels))
width = 0.32

# Baseline Vulnerable (Pairwise verification without physical invariant consensus)
# and Proposed ZT-CACC (with 95% CI over 10 folds)
f1_baseline = [99.70, 81.20, 62.80]
f1_baseline_ci = [0.12, 0.85, 1.40]

f1_zt = [99.71, 98.40, 96.75]
f1_zt_ci = [0.04, 0.12, 0.18]

bars1 = ax1.bar(x - width/2, f1_baseline, width, yerr=f1_baseline_ci, capsize=4,
                color='#e05a5a', edgecolor='black', linewidth=0.8,
                label='Pairwise Verification (Vulnerable Baseline)')
bars2 = ax1.bar(x + width/2, f1_zt, width, yerr=f1_zt_ci, capsize=4,
                color='#3a78d8', edgecolor='black', linewidth=0.8,
                label='Proposed ZT-CACC (Multi-Modal Consensus)')

# Annotate values
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#b22222')

for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a4ca8')

ax1.set_ylabel(r'$F_1$-Score (%)', fontsize=10, fontweight='bold')
ax1.set_ylim(50.0, 105.0)
ax1.set_xticks(x)
ax1.set_xticklabels(m_labels, fontsize=9.5, fontweight='bold')
ax1.set_title('(a) Defense Against Colluding Byzantine Attackers', fontsize=10.5, fontweight='bold', pad=8)
ax1.grid(axis='y', linestyle='--', alpha=0.55)
ax1.legend(loc='lower left', fontsize=8.2, framealpha=0.92)

# -------------------------------------------------------------------------
# Subplot (b): Component Ablation Study
# (Exact match with Table VIII: All 6 configurations with 95% CIs)
# -------------------------------------------------------------------------
ablation_configs = [
    'Raw BSMs Only (No Invariants)',
    r'w/o Radar Doppler Invariant ($r_v$)',
    'w/o Multi-RAT Adaptive Broker',
    r'w/o LiDAR Spatial Residual ($r_p$)',
    'w/o Trust Weighting (Binary)',
    'Full ZT-CACC Framework'
]

f1_ablation = [53.80, 91.72, 93.08, 94.15, 96.35, 99.77]
f1_ablation_ci = [0.68, 0.29, 0.26, 0.24, 0.19, 0.04]

y_pos = np.arange(len(ablation_configs))
colors_bar = ['#d9534f', '#f0ad4e', '#5bc0de', '#5bc0de', '#428bca', '#2e9b67']

bars_b = ax2.barh(y_pos, f1_ablation, xerr=f1_ablation_ci, capsize=4,
                  color=colors_bar, edgecolor='black', linewidth=0.8, height=0.62)

for bar, val in zip(bars_b, f1_ablation):
    w = bar.get_width()
    ax2.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
             va='center', ha='left', fontsize=8.5, fontweight='bold', color='#222222')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(ablation_configs, fontsize=8.8, fontweight='bold')
ax2.set_xlabel(r'$F_1$-Score (%) [Mean $\pm$ 95% CI]', fontsize=10, fontweight='bold')
ax2.set_xlim(45.0, 107.0)
ax2.set_title('(b) Component Ablation Contribution (Table VIII)', fontsize=10.5, fontweight='bold', pad=8)
ax2.grid(axis='x', linestyle='--', alpha=0.55)

plt.tight_layout()
output_path = 'Fig10_Colluding_Attacks_and_Ablation_Study.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Successfully saved figure to {output_path}')
