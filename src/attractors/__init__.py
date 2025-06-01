"""
Attractor implementations and registry
"""

from .base import Attractor, AttractorConfig
from .henon import HenonMap
from .clifford import CliffordMap

# available attractors registry
AVAILABLE_ATTRACTORS = {
    'henon': HenonMap,
    'clifford': CliffordMap
}

__all__ = ['Attractor', 'AttractorConfig', 'HenonMap', 'CliffordMap', 'AVAILABLE_ATTRACTORS'] 
