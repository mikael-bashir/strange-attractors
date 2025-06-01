"""
Noise type definitions and generation functions
"""

from enum import Enum
import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict, Union

class NoiseType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"

@dataclass
class NoiseConfig:
    type: NoiseType
    params: Dict[str, float]
    
    @classmethod
    def from_cli(cls) -> 'NoiseConfig':
        print("\nChoose noise type:")
        for i, noise_type in enumerate(NoiseType, 1):
            print(f"{i}. {noise_type.value}")
        
        while True:
            try:
                choice = int(input("\nEnter choice (1-3): "))
                noise_type = list(NoiseType)[choice - 1]
                break
            except (ValueError, IndexError):
                print("Invalid choice. Please enter 1, 2, or 3.")
        
        params = {}
        if noise_type == NoiseType.UNIFORM:
            while True:
                try:
                    low = float(input("Enter lower bound: "))
                    high = float(input("Enter upper bound: "))
                    if low < high:
                        params = {'low': low, 'high': high}
                        break
                    print("Upper bound must be greater than lower bound.")
                except ValueError:
                    print("Please enter valid numbers.")
                    
        elif noise_type == NoiseType.GAUSSIAN:
            while True:
                try:
                    mean = float(input("Enter mean: "))
                    std = float(input("Enter standard deviation: "))
                    if std > 0:
                        params = {'mean': mean, 'std': std}
                        break
                    print("Standard deviation must be positive.")
                except ValueError:
                    print("Please enter valid numbers.")
        
        return cls(noise_type, params)
    
    def get_range(self) -> float:
        """Get effective noise range based on noise type"""
        if self.type == NoiseType.GAUSSIAN:
            return self.params['std']
        elif self.type == NoiseType.UNIFORM:
            return (self.params['high'] - self.params['low']) / 2
        return 0.0

def generate_noise(size: int, config: NoiseConfig) -> cp.ndarray:
    """Generate noise on GPU using CuPy."""
    if config.type == NoiseType.NONE:
        return cp.zeros(size, dtype=cp.float32)
    
    elif config.type == NoiseType.UNIFORM:
        return cp.random.uniform(
            config.params['low'],
            config.params['high'],
            size=size,
            dtype=cp.float32
        )
    
    elif config.type == NoiseType.GAUSSIAN:
        return cp.random.normal(
            config.params['mean'],
            config.params['std'],
            size=size,
            dtype=cp.float32
        )
    
    raise ValueError(f"Unknown noise type: {config.type}")

# Default noise config for each attractor
ATTRACTOR_NOISE_CONFIGS = {
    'henon': NoiseConfig(
        type=NoiseType.UNIFORM,
        params={'low': -0.001, 'high': 0.001}
    ),
    'clifford': NoiseConfig(
        type=NoiseType.UNIFORM,
        params={'low': -0.15, 'high': 0.15}
    )
} 
