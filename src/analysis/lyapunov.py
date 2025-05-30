"""
Lyapunov exponent tracking and analysis
"""

import numpy as np
import cupy as cp
from typing import Dict, Tuple, List, Optional

# Configuration
LYAP_RENORM_INTERVAL = 10  # Renormalize every N steps
LYAP_SAVE_INTERVAL = 100   # Save values every N steps

class LyapunovTracker:
    """Track Lyapunov exponents during CUDA simulation"""
    
    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.history: List[Dict] = []
        self.total_time = 0
        
        # Initialize tangent space matrices (identity) on GPU
        self.tangent_xx = cp.ones(num_particles, dtype=cp.float32)
        self.tangent_xy = cp.zeros(num_particles, dtype=cp.float32)
        self.tangent_yx = cp.zeros(num_particles, dtype=cp.float32)
        self.tangent_yy = cp.ones(num_particles, dtype=cp.float32)
        
        # Running sums for both Lyapunov exponents
        self.lyap_sums1 = cp.zeros(num_particles, dtype=cp.float32)
        self.lyap_sums2 = cp.zeros(num_particles, dtype=cp.float32)
        self.step_count = cp.zeros(num_particles, dtype=cp.int32)
    
    def get_current_exponents(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get current Lyapunov spectrum (mean and max)"""
        sums1_cpu = cp.asnumpy(self.lyap_sums1)
        sums2_cpu = cp.asnumpy(self.lyap_sums2)
        counts_cpu = cp.asnumpy(self.step_count)
        
        valid_mask = counts_cpu > 0
        if np.any(valid_mask):
            # Individual particle Lyapunov spectrums
            lyap1 = sums1_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
            lyap2 = sums2_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
            
            # Mean and max of first exponent
            mean_lyap1 = np.mean(lyap1)
            max_lyap1 = np.max(lyap1)
            
            # Mean and max of second exponent (use min as typically negative)
            mean_lyap2 = np.mean(lyap2)
            max_lyap2 = np.min(lyap2)
            
            return (mean_lyap1, mean_lyap2), (max_lyap1, max_lyap2)
        
        return (0.0, 0.0), (0.0, 0.0)
    
    def save_snapshot(self, noise_range: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Save current Lyapunov spectrum to history"""
        mean_spectrum, max_spectrum = self.get_current_exponents()
        mean_lyap1, mean_lyap2 = mean_spectrum
        max_lyap1, max_lyap2 = max_spectrum
        
        # Compute Kaplan-Yorke dimensions
        mean_ky_dim = self._compute_kaplan_yorke_dim(mean_spectrum)
        max_ky_dim = self._compute_kaplan_yorke_dim(max_spectrum)
        
        self.history.append({
            'time': self.total_time,
            'mean_lyap1': mean_lyap1,
            'mean_lyap2': mean_lyap2,
            'max_lyap1': max_lyap1,
            'max_lyap2': max_lyap2,
            'mean_ky_dim': mean_ky_dim,
            'max_ky_dim': max_ky_dim,
            'noise_range': noise_range
        })
        return mean_spectrum, max_spectrum
    
    @staticmethod
    def _compute_kaplan_yorke_dim(lyap_spectrum: Tuple[float, float]) -> Optional[float]:
        """
        Compute Kaplan-Yorke dimension from Lyapunov spectrum.
        For 2D system: KY dim = 1 + λ1/|λ2| if λ1 + λ2 < 0 and λ1 > 0
        """
        lambda1, lambda2 = lyap_spectrum
        
        if lambda1 + lambda2 < 0 and lambda1 > 0:
            ky_dim = 1.0 + lambda1/abs(lambda2)
            return min(ky_dim, 2.0)  # cap at embedding dimension
        elif lambda1 <= 0:
            return 1.0  # periodic/stable
        else:
            return 2.0  # fully chaotic: both exponents positive → attractor fills phase space 