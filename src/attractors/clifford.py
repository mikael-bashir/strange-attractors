"""
Clifford map implementation
"""

from .base import Attractor, AttractorConfig
from typing import Dict, Tuple
import math

class CliffordMap(Attractor):
    def __init__(self):
        config = AttractorConfig(
            name="Clifford",
            params={'a': -1.7, 'b': 1.8, 'c': -0.9, 'd': -0.4},
            bounds={'x': (-2.5, 2.5), 'y': (-2.5, 2.5)},
            init_bounds={'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}
        )
        super().__init__(config)
    
    def get_jacobian(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Return Jacobian matrix elements for Clifford map"""
        a, b, c, d = params['a'], params['b'], params['c'], params['d']
        j11 = -c * a * math.sin(a * x)
        j12 = a * math.cos(a * y)
        j21 = b * math.cos(b * x)
        j22 = -d * b * math.sin(b * y)
        return j11, j12, j21, j22
    
    def evolve_point(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float]:
        """Evolve a single point under Clifford map"""
        a, b, c, d = params['a'], params['b'], params['c'], params['d']
        x_new = math.sin(a * y) + c * math.cos(a * x)
        y_new = math.sin(b * x) + d * math.cos(b * y)
        return x_new, y_new 