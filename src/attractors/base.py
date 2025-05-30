"""
Base class for strange attractors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class AttractorConfig:
    name: str
    params: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    init_bounds: Dict[str, Tuple[float, float]]

class Attractor(ABC):
    def __init__(self, config: AttractorConfig):
        self.config = config
        self.param_names = list(config.params.keys())
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def params(self) -> Dict[str, float]:
        return self.config.params
    
    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self.config.bounds
    
    @property
    def init_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self.config.init_bounds
    
    @abstractmethod
    def get_jacobian(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Return the Jacobian matrix elements (j11, j12, j21, j22) at point (x,y)"""
        
    @abstractmethod
    def evolve_point(self, x: float, y: float, params: Dict[str, float]) -> Tuple[float, float]:
        """Evolve a single point (x,y) one step forward""" 