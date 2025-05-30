"""
Strange attractor implementations
"""

from .base import Attractor, AttractorConfig
from .henon import HenonMap
from .clifford import CliffordMap
from .ikeda import IkedaMap

AVAILABLE_ATTRACTORS = {
    'henon': HenonMap,
    'clifford': CliffordMap,
    'ikeda': IkedaMap
}

__all__ = ['Attractor', 'AttractorConfig', 'HenonMap', 'CliffordMap', 'IkedaMap', 'AVAILABLE_ATTRACTORS'] 