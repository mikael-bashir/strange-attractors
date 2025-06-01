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
    
    def get_detailed_exponents(self) -> Dict[str, float]:
        """Get detailed lyapunov statistics including percentiles"""
        sums1_cpu = cp.asnumpy(self.lyap_sums1)
        sums2_cpu = cp.asnumpy(self.lyap_sums2)
        counts_cpu = cp.asnumpy(self.step_count)
        
        valid_mask = counts_cpu > 0
        if not np.any(valid_mask):
            return {'num_valid': 0}
            
        # Individual particle Lyapunov spectrums
        lyap1 = sums1_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
        lyap2 = sums2_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
        
        # Remove any infinities or nans
        lyap1_clean = lyap1[np.isfinite(lyap1)]
        lyap2_clean = lyap2[np.isfinite(lyap2)]
        
        detailed_stats = {
            'num_valid': len(lyap1_clean),
            'num_particles': self.num_particles
        }
        
        if len(lyap1_clean) > 0:
            detailed_stats.update({
                # λ₁ statistics
                'lyap1_mean': np.mean(lyap1_clean),
                'lyap1_std': np.std(lyap1_clean),
                'lyap1_min': np.min(lyap1_clean),
                'lyap1_max': np.max(lyap1_clean),
                'lyap1_p10': np.percentile(lyap1_clean, 10),
                'lyap1_p25': np.percentile(lyap1_clean, 25),
                'lyap1_p50': np.percentile(lyap1_clean, 50),  # median
                'lyap1_p75': np.percentile(lyap1_clean, 75),
                'lyap1_p90': np.percentile(lyap1_clean, 90),
            })
            
        if len(lyap2_clean) > 0:
            detailed_stats.update({
                # λ₂ statistics  
                'lyap2_mean': np.mean(lyap2_clean),
                'lyap2_std': np.std(lyap2_clean),
                'lyap2_min': np.min(lyap2_clean),
                'lyap2_max': np.max(lyap2_clean),
                'lyap2_p10': np.percentile(lyap2_clean, 10),
                'lyap2_p25': np.percentile(lyap2_clean, 25),
                'lyap2_p50': np.percentile(lyap2_clean, 50),  # median
                'lyap2_p75': np.percentile(lyap2_clean, 75),
                'lyap2_p90': np.percentile(lyap2_clean, 90),
            })
            
        # Compute Kaplan-Yorke dimensions if both exponents available
        if len(lyap1_clean) > 0 and len(lyap2_clean) > 0:
            mean_ky = self._compute_kaplan_yorke_dim((detailed_stats['lyap1_mean'], detailed_stats['lyap2_mean']))
            median_ky = self._compute_kaplan_yorke_dim((detailed_stats['lyap1_p50'], detailed_stats['lyap2_p50']))
            
            detailed_stats.update({
                'ky_dim_mean': mean_ky,
                'ky_dim_median': median_ky
            })
            
        return detailed_stats

    def save_snapshot(self, noise_range: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Save current Lyapunov spectrum to history"""
        mean_spectrum, max_spectrum = self.get_current_exponents()
        detailed_stats = self.get_detailed_exponents()
        mean_lyap1, mean_lyap2 = mean_spectrum
        max_lyap1, max_lyap2 = max_spectrum
        
        # Compute Kaplan-Yorke dimensions
        mean_ky_dim = self._compute_kaplan_yorke_dim(mean_spectrum)
        max_ky_dim = self._compute_kaplan_yorke_dim(max_spectrum)
        
        # Create comprehensive snapshot
        snapshot = {
            'time': self.total_time,
            'mean_lyap1': mean_lyap1,
            'mean_lyap2': mean_lyap2,
            'max_lyap1': max_lyap1,
            'max_lyap2': max_lyap2,
            'mean_ky_dim': mean_ky_dim,
            'max_ky_dim': max_ky_dim,
            'noise_range': noise_range,
            **detailed_stats
        }
        
        self.history.append(snapshot)
        return mean_spectrum, max_spectrum
    
    def get_recent_trends(self, window_steps: int = 500) -> Dict[str, float]:
        """Compute recent trends in lyapunov exponents"""
        if len(self.history) < 3:
            return {'trend_lyap1': 0.0, 'trend_lyap2': 0.0, 'confidence': 0.0}
            
        # Filter recent history
        current_time = self.total_time
        recent_history = [h for h in self.history if h['time'] >= current_time - window_steps]
        
        if len(recent_history) < 3:
            recent_history = self.history[-3:]  # at least last 3 points
            
        times = np.array([h['time'] for h in recent_history])
        lyap1_values = np.array([h['mean_lyap1'] for h in recent_history])
        lyap2_values = np.array([h['mean_lyap2'] for h in recent_history])
        
        # linear regression slopes
        if len(times) >= 3:
            slope1, _ = np.polyfit(times, lyap1_values, 1)
            slope2, _ = np.polyfit(times, lyap2_values, 1)
            
            # r-squared for confidence
            lyap1_pred = np.polyval([slope1, _], times)
            ss_res1 = np.sum((lyap1_values - lyap1_pred) ** 2)
            ss_tot1 = np.sum((lyap1_values - np.mean(lyap1_values)) ** 2)
            r_squared1 = 1 - (ss_res1 / (ss_tot1 + 1e-10))
            
            return {
                'trend_lyap1': slope1,
                'trend_lyap2': slope2,
                'confidence': max(0.0, r_squared1),
                'window_size': len(recent_history)
            }
        
        return {'trend_lyap1': 0.0, 'trend_lyap2': 0.0, 'confidence': 0.0}
    
    def format_detailed_status(self) -> str:
        """Format detailed current status string"""
        if not self.history:
            return "No lyapunov data available"
            
        latest = self.history[-1]
        trends = self.get_recent_trends()
        
        status = (f"λ₁={latest['mean_lyap1']:.4f}±{latest.get('lyap1_std', 0):.3f}, "
                 f"λ₂={latest['mean_lyap2']:.4f}±{latest.get('lyap2_std', 0):.3f}")
        
        # Add trend info
        trend1 = trends['trend_lyap1']
        trend_symbol = "↗" if trend1 > 1e-5 else ("↘" if trend1 < -1e-5 else "→")
        status += f" {trend_symbol}"
        
        # Add regime classification
        if latest['mean_lyap1'] > 0 and latest['mean_lyap2'] < 0:
            regime = "chaotic"
        elif latest['mean_lyap1'] > 0 and latest['mean_lyap2'] > 0:
            regime = "hyperchaotic"
        elif latest['mean_lyap1'] < 0:
            regime = "synchronized"
        else:
            regime = "marginal"
            
        status += f" [{regime}]"
        
        # Add KY dimension if available
        # if 'ky_dim_mean' in latest:
        #     ky = latest['ky_dim_mean']
        #     if ky is not None:
        #         status += f" KY={ky:.3f}"
                
        return status
    
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