"""
Ikeda map implementation
"""

from .base import Attractor, AttractorConfig
from typing import Dict, Tuple
import math

class IkedaMap(Attractor):
    def __init__(self):
        config = AttractorConfig(
            name="Ikeda",
            params={'u': 0.90},
            bounds={'x': (0.0, 2.0), 'y': (-2.0, 2.0)},
            init_bounds={'x': (0.1, 0.5), 'y': (-0.5, 0.5)}
        )
        super().__init__(config)
    
    def get_jacobian(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Return Jacobian matrix elements for Ikeda map"""
        u = params['u']
        t = 0.4 - 6.0 / (1.0 + x*x + y*y)
        dt_dx = 6.0 * 2.0 * x / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y))
        dt_dy = 6.0 * 2.0 * y / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y))
        
        j11 = u * (math.cos(t) - x * dt_dx * math.sin(t))
        j12 = u * (-x * dt_dy * math.sin(t))
        j21 = u * (math.sin(t) + x * dt_dx * math.cos(t))
        j22 = u * (1.0 + x * dt_dy * math.cos(t))
        
        return j11, j12, j21, j22
    
    def evolve_point(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float]:
        """Evolve a single point under Ikeda map"""
        u = params['u']
        t = 0.4 - 6.0 / (1.0 + x*x + y*y)
        x_new = 1.0 + u * (x * math.cos(t) - y * math.sin(t))
        y_new = u * (x * math.sin(t) + y * math.cos(t))
        return x_new, y_new 