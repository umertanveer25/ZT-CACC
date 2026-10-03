"""
Core modules for vehicle platoon dynamics, control laws, and physical layer broker.
"""

from .platoon import Vehicle, Platoon
from .controller import ResilientCACCController, TrustManager
from .multi_rat import MultiRATBroker, RATType
from .attacks import CyberAttackInjector, AttackType

__all__ = [
    "Vehicle",
    "Platoon",
    "ResilientCACCController",
    "TrustManager",
    "MultiRATBroker",
    "RATType",
    "CyberAttackInjector",
    "AttackType",
]
