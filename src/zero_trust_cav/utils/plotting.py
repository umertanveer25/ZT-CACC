"""
Plotting and Publication Figure Styling Utilities.
Configures Matplotlib and Seaborn for IEEE Transactions publication standards.
"""

import matplotlib.pyplot as plt

def set_publication_style():
    """Applies IEEE Transactions typography and grid aesthetics."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('default')
        
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'grid.alpha': 0.6,
        'grid.linestyle': '--'
    })
