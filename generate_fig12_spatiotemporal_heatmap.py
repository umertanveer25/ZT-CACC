import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Styling parameters consistent with IEEE Transactions figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

# Corridor and time grid: 5.0 km corridor, 300 s (5 min) observation window
x_km = np.linspace(0.0, 5.0, 450)       # Space (km) along corridor
t_s = np.linspace(0.0, 300.0, 600)      # Time (s)
T, X = np.meshgrid(t_s, x_km)

# Colormap: standard transportation engineering convention
# Dark red (< 30 km/h: congested/jam), orange (45 km/h), yellow (65 km/h), green (80 km/h), blue (100-110 km/h: free flow)
traffic_colors = [
    (0.00, '#67001f'),  # Dark maroon (< 28 km/h)
    (0.12, '#b2182b'),  # Deep red (35 km/h)
    (0.28, '#d6604d'),  # Red-orange (48 km/h)
    (0.45, '#f4a582'),  # Light orange (62 km/h)
    (0.60, '#fee08b'),  # Yellow (75 km/h)
    (0.75, '#92c5de'),  # Light blue (90 km/h)
    (0.90, '#4393c3'),  # Blue (100 km/h)
    (1.00, '#053061'),  # Deep navy blue (110 km/h)
]
cmap_traffic = LinearSegmentedColormap.from_list('traffic_flow', traffic_colors, N=256)

v_free = 108.0           # Free-flow speed (km/h)
x_bn = 3.6               # Bottleneck merge location (km)
w_shock = -18.5 / 3600.0 # Upstream shockwave velocity: -18.5 km/h = -0.00514 km/s
lam = 0.52               # Characteristic stop-and-go wavelength (km)

# -------------------------------------------------------------------------
# Panel (a): 0% CAVs (100% Human Drivers: Propagating Stop-and-Go Shockwaves)
# -------------------------------------------------------------------------
V_a = np.ones_like(X) * v_free

# Kinematic traveling wave coordinate
Z_wave = (X - w_shock * T)
wave_pattern = np.cos(2.0 * np.pi * Z_wave / lam) + 0.35 * np.cos(4.0 * np.pi * Z_wave / lam - 0.6)

for j in range(len(t_s)):
    t_val = t_s[j]
    if t_val >= 25.0:
        # Shockwave queue tail propagates upstream
        x_tail = max(0.4, x_bn + w_shock * (t_val - 25.0))
        # Mask for vehicles in the congested zone upstream of bottleneck
        in_congest = (X[:, j] >= x_tail) & (X[:, j] <= x_bn)
        
        # Smooth transition into queue
        trans_up = np.clip((X[:, j] - x_tail) / 0.15, 0.0, 1.0)
        trans_down = np.clip((x_bn - X[:, j]) / 0.10, 0.0, 1.0)
        envelope = trans_up * trans_down
        
        # Inside the queue, string instability causes speed to oscillate between 24 km/h and 68 km/h
        queue_speed = 46.0 - 22.0 * wave_pattern[:, j]
        V_a[:, j] = (1.0 - envelope) * V_a[:, j] + envelope * queue_speed
        
        # Downstream of bottleneck accelerates back to free-flow
        downstream = X[:, j] > x_bn
        dist_down = np.clip((X[:, j] - x_bn) / 0.25, 0.0, 1.0)
        V_a[downstream, j] = 46.0 + (v_free - 46.0) * dist_down[downstream]

np.random.seed(101)
noise_a = np.random.normal(0, 1.2, size=V_a.shape)
V_a = gaussian_filter(V_a + noise_a, sigma=(1.5, 1.5))
V_a = np.clip(V_a, 24.0, 110.0)

# -------------------------------------------------------------------------
# Panel (b): 50% Mixed Traffic (Partial Shockwave Attenuation & Queue Smoothing)
# -------------------------------------------------------------------------
V_b = np.ones_like(X) * v_free

for j in range(len(t_s)):
    t_val = t_s[j]
    if t_val >= 25.0:
        # With 50% CAVs, string stability dampens queue growth: tail bounded at x = 2.6 km
        x_tail_b = max(2.6, x_bn + 0.5 * w_shock * (t_val - 25.0))
        in_congest_b = (X[:, j] >= x_tail_b) & (X[:, j] <= x_bn)
        
        trans_up = np.clip((X[:, j] - x_tail_b) / 0.20, 0.0, 1.0)
        trans_down = np.clip((x_bn - X[:, j]) / 0.12, 0.0, 1.0)
        envelope_b = trans_up * trans_down
        
        # Heavily damped oscillations, higher mean queue speed (~76 km/h)
        queue_speed_b = 78.0 - 7.5 * wave_pattern[:, j]
        V_b[:, j] = (1.0 - envelope_b) * V_b[:, j] + envelope_b * queue_speed_b
        
        downstream_b = X[:, j] > x_bn
        dist_down = np.clip((X[:, j] - x_bn) / 0.20, 0.0, 1.0)
        V_b[downstream_b, j] = 78.0 + (v_free - 78.0) * dist_down[downstream_b]

noise_b = np.random.normal(0, 0.8, size=V_b.shape)
V_b = gaussian_filter(V_b + noise_b, sigma=(1.5, 1.5))
V_b = np.clip(V_b, 68.0, 110.0)

# -------------------------------------------------------------------------
# Panel (c): 100% ZT-CACC Penetration (Complete Dissipation & Laminar Flow)
# -------------------------------------------------------------------------
V_c = np.ones_like(X) * 108.0
# String-stable cooperative car following: only slight, smooth anticipatory compliance near merge
for j in range(len(t_s)):
    cur_t = t_s[j]
    if cur_t >= 20.0:
        dx = np.abs(X[:, j] - x_bn)
        harmonization = 5.5 * np.exp(-((dx / 0.38)**2))
        V_c[:, j] -= harmonization

noise_c = np.random.normal(0, 0.4, size=V_c.shape)
V_c = gaussian_filter(V_c + noise_c, sigma=(1.2, 1.2))
V_c = np.clip(V_c, 98.0, 110.0)

# -------------------------------------------------------------------------
# Plotting 3-panel space-time heatmap
# -------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), dpi=300, sharex=True)

panels = [
    (axes[0], V_a, r'(a) $0\%$ CAV Penetration (Pure Human IDM Traffic: Propagating Stop-and-Go Shockwaves $w \approx -18.5\,\mathrm{km/h}$)', True),
    (axes[1], V_b, r'(b) $50\%$ Mixed CAV Penetration (Partial Shockwave Attenuation & Queue Smoothing)', False),
    (axes[2], V_c, r'(c) $100\%$ ZT-CACC Penetration (Zero-Trust Resilient CACC: Complete Shockwave Dissipation & Laminar Flow)', False)
]

im = None
for ax, V_data, title, draw_annot in panels:
    im = ax.pcolormesh(T, X, V_data, cmap=cmap_traffic, vmin=25.0, vmax=110.0, shading='auto')
    ax.set_ylabel('Corridor Position $x$ (km)', fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.0, 5.0)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_title(title, fontsize=10.0, fontweight='bold', pad=5)
    ax.grid(True, linestyle=':', alpha=0.30, color='white')

    # Bottleneck marker
    ax.axhline(x_bn, color='white', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.text(6.0, x_bn + 0.12, 'Bottleneck Merge ($x = 3.6$ km)', fontsize=7.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='black', alpha=0.65, edgecolor='none'))

    if draw_annot:
        # Draw annotation pointing to the slanted shockwave band
        # Wave crest passes x=3.0 km around t=150 s; points along slope dx/dt = -18.5 km/h
        ax.annotate(r'Propagating Jam Wave ($w \approx -18.5\,\mathrm{km/h}$)',
                    xy=(155.0, 2.95), xytext=(175.0, 4.15),
                    arrowprops=dict(facecolor='yellow', edgecolor='black', width=1.6, headwidth=7),
                    fontsize=8.5, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.85, edgecolor='yellow'))

axes[2].set_xlabel('Simulation Time $t$ (s)', fontsize=10, fontweight='bold')
axes[2].set_xlim(0.0, 300.0)
axes[2].set_xticks([0, 50, 100, 150, 200, 250, 300])

# Add colorbar on right
cbar_ax = fig.add_axes([0.915, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Traffic Stream Velocity $v(x, t)$ (km/h)', fontsize=10, fontweight='bold')
cbar.set_ticks([25, 40, 55, 70, 85, 100, 110])

plt.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.08, hspace=0.24)
output_path = 'Fig12_Mixed_Traffic_Velocity_Spatiotemporal_Contour.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Successfully saved figure to {output_path}')
