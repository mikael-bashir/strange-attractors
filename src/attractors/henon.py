"""
Hénon map implementation
"""

from .base import Attractor, AttractorConfig
from typing import Dict, Tuple

class HenonMap(Attractor):
    def __init__(self):
        config = AttractorConfig(
            name="Hénon",
            params={'a': 1.4, 'b': 0.3},
            bounds={'x': (-2.0, 2.0), 'y': (-0.6, 0.6)},
            init_bounds={'x': (-1.0, 1.0), 'y': (-0.5, 0.5)}
        )
        super().__init__(config)
    
    def get_jacobian(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Return Jacobian matrix elements for Hénon map"""
        a = params['a']
        b = params['b']
        return -2.0 * a * x, 1.0, b, 0.0
    
    def evolve_point(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float]:
        """Evolve a single point under Hénon map"""
        a = params['a']
        b = params['b']
        x_new = 1.0 - a * x * x + y
        y_new = b * x
        return x_new, y_new 