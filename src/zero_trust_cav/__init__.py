"""
Zero-Trust CAV Platooning Package
=================================
A modular, high-performance research framework for Zero-Trust Transaction Verification,
Heterogeneous Multi-RAT V2X Switching, and String-Stable CACC Platooning.
"""

__version__ = "1.0.0"
__author__ = "Umer Tanveer and Abdul Salam"
__license__ = "MIT"

from .config import (
    SimulationConfig,
    MultiRATConfig,
    CACCConfig,
    DetectionConfig,
    AttackConfig,
)

__all__ = [
    "SimulationConfig",
    "MultiRATConfig",
    "CACCConfig",
    "DetectionConfig",
    "AttackConfig",
    "__version__",
]
