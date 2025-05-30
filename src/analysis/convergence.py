"""
Convergence detection for parameter sweeps
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ConvergenceConfig:
    """configuration for convergence detection"""
    min_steps: int = 1000  # minimum steps before checking convergence
    max_steps: int = 20000  # maximum steps to run
    lyap_tolerance: float = 1e-4  # convergence threshold for lyapunov exponents
    window_size: int = 500  # steps to average over for convergence check
    stability_checks: int = 3  # number of consecutive stable windows required
    divergence_threshold: float = 10.0  # threshold for detecting divergence
    min_history_length: int = 5  # minimum snapshots needed for convergence check


class ConvergenceTracker:
    """monitors simulation convergence based on lyapunov exponents"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """reset convergence state"""
        self.convergence_history: List[Dict[str, Any]] = []
        self.stability_count = 0
        self.converged = False
        self.diverged = False
        self.early_stop_reason: Optional[str] = None
        
    def check_convergence(self, lyap_history: List[Dict]) -> bool:
        """
        check if lyapunov exponents have converged
        
        args:
            lyap_history: list of lyapunov snapshots from LyapunovTracker
            
        returns:
            true if converged, false otherwise
        """
        if len(lyap_history) < self.config.min_history_length:
            return False
            
        # extract time series of lyapunov exponents
        times = np.array([h['time'] for h in lyap_history])
        mean_lyap1 = np.array([h['mean_lyap1'] for h in lyap_history])
        mean_lyap2 = np.array([h['mean_lyap2'] for h in lyap_history])
        ky_dims = np.array([h.get('mean_ky_dim', np.nan) for h in lyap_history])
        
        # check for early stopping conditions first
        if self._check_early_stop(times, mean_lyap1, mean_lyap2, ky_dims):
            return True
            
        # need minimum steps before checking convergence
        current_time = times[-1]
        if current_time < self.config.min_steps:
            return False
            
        # check if we've hit max steps
        if current_time >= self.config.max_steps:
            self.early_stop_reason = "max_steps_reached"
            return True
            
        # look at recent window and check for convergence
        window_indices = self._get_window_indices(times, self.config.window_size)
        if len(window_indices) < 2:
            return False
            
        # stability metrics
        window_lyap1 = mean_lyap1[window_indices]
        window_lyap2 = mean_lyap2[window_indices]
        window_ky = ky_dims[window_indices]
        
        stability_metrics = self._compute_stability_metrics(
            window_lyap1, window_lyap2, window_ky
        )

        self.convergence_history.append({
            'time': current_time,
            'window_size': len(window_indices),
            **stability_metrics
        })
        
        # is it stable?
        is_stable = self._is_window_stable(stability_metrics)
        
        if is_stable:
            self.stability_count += 1
        else:
            self.stability_count = 0
            
        # if enough stable windows, we've converged
        if self.stability_count >= self.config.stability_checks:
            self.converged = True
            return True
            
        return False
        
    def _check_early_stop(self, times: np.ndarray, lyap1: np.ndarray, 
                         lyap2: np.ndarray, ky_dims: np.ndarray) -> bool:
        """check for early stopping conditions (divergence, nan, etc)"""
        
        # nans/infinities
        if np.any(~np.isfinite(lyap1)) or np.any(~np.isfinite(lyap2)):
            self.early_stop_reason = "nan_or_inf_detected"
            self.diverged = True
            return True
            
        # extreme divergence
        recent_lyap1 = lyap1[-3:] if len(lyap1) >= 3 else lyap1
        if np.any(np.abs(recent_lyap1) > self.config.divergence_threshold):
            self.early_stop_reason = "lyapunov_divergence"
            self.diverged = True
            return True
            
        # check for dimension collapse (might indicate numerical issues)
        recent_ky = ky_dims[-3:] if len(ky_dims) >= 3 else ky_dims
        valid_ky = recent_ky[np.isfinite(recent_ky)]
        if len(valid_ky) > 0 and np.any(valid_ky < 0.1):
            self.early_stop_reason = "dimension_collapse"
            return True
            
        return False
        
    def _get_window_indices(self, times: np.ndarray, window_size: int) -> np.ndarray:
        """get indices for the most recent window of given size"""
        if len(times) == 0:
            return np.array([], dtype=int)
            
        current_time = times[-1]
        window_start = current_time - window_size
        return np.where(times >= window_start)[0]
        
    def _compute_stability_metrics(self, lyap1: np.ndarray, lyap2: np.ndarray, 
                                  ky_dims: np.ndarray) -> Dict[str, float]:
        """compute stability metrics for a window of lyapunov data"""
        
        # remove nans for computation
        valid_lyap1 = lyap1[np.isfinite(lyap1)]
        valid_lyap2 = lyap2[np.isfinite(lyap2)]
        valid_ky = ky_dims[np.isfinite(ky_dims)]
        
        metrics = {}
        
        # coefficient of variation (std/mean) for each exponent
        if len(valid_lyap1) > 1:
            cv1 = np.std(valid_lyap1) / (np.abs(np.mean(valid_lyap1)) + 1e-10)
            metrics['lyap1_cv'] = cv1
            metrics['lyap1_mean'] = np.mean(valid_lyap1)
            metrics['lyap1_std'] = np.std(valid_lyap1)
        else:
            metrics.update({'lyap1_cv': np.inf, 'lyap1_mean': np.nan, 'lyap1_std': np.nan})
            
        if len(valid_lyap2) > 1:
            cv2 = np.std(valid_lyap2) / (np.abs(np.mean(valid_lyap2)) + 1e-10)
            metrics['lyap2_cv'] = cv2
            metrics['lyap2_mean'] = np.mean(valid_lyap2)
            metrics['lyap2_std'] = np.std(valid_lyap2)
        else:
            metrics.update({'lyap2_cv': np.inf, 'lyap2_mean': np.nan, 'lyap2_std': np.nan})
            
        # kaplan-yorke dimension stability
        if len(valid_ky) > 1:
            ky_cv = np.std(valid_ky) / (np.mean(valid_ky) + 1e-10)
            metrics['ky_cv'] = ky_cv
            metrics['ky_mean'] = np.mean(valid_ky)
            metrics['ky_std'] = np.std(valid_ky)
        else:
            metrics.update({'ky_cv': np.inf, 'ky_mean': np.nan, 'ky_std': np.nan})
            
        # trend analysis (linear fit slope)
        if len(valid_lyap1) > 2:
            x = np.arange(len(valid_lyap1))
            slope1, _ = np.polyfit(x, valid_lyap1, 1)
            metrics['lyap1_trend'] = slope1
        else:
            metrics['lyap1_trend'] = 0.0
            
        return metrics
        
    def _is_window_stable(self, metrics: Dict[str, float]) -> bool:
        """determine if a window represents stable convergence"""
        
        # check coefficient of variation for both exponents
        lyap1_stable = metrics.get('lyap1_cv', np.inf) < self.config.lyap_tolerance
        lyap2_stable = metrics.get('lyap2_cv', np.inf) < self.config.lyap_tolerance
        
        # check that trend is small (not rapidly changing)
        trend_stable = abs(metrics.get('lyap1_trend', 0.0)) < self.config.lyap_tolerance
        
        # kaplan-yorke dimension should also be stable
        ky_stable = metrics.get('ky_cv', np.inf) < self.config.lyap_tolerance * 2  # bit more lenient
        
        return lyap1_stable and lyap2_stable and trend_stable and ky_stable
        
    def should_stop_early(self) -> bool:
        """check if simulation should stop early due to divergence/issues"""
        return self.diverged or self.early_stop_reason is not None
        
    def get_convergence_info(self) -> Dict[str, Any]:
        """get detailed convergence information"""
        return {
            'converged': self.converged,
            'diverged': self.diverged,
            'early_stop_reason': self.early_stop_reason,
            'stability_count': self.stability_count,
            'required_stability': self.config.stability_checks,
            'convergence_history': self.convergence_history[-10:],  # last 10 checks
            'final_metrics': self.convergence_history[-1] if self.convergence_history else None
        }
        
    def estimate_remaining_steps(self, current_time: int) -> Optional[int]:
        """estimate how many more steps needed for convergence"""
        if self.converged or self.should_stop_early():
            return 0
            
        if current_time >= self.config.max_steps:
            return 0
            
        # if we have recent stability, estimate based on that
        if self.stability_count > 0:
            # need (stability_checks - current_count) more stable windows
            remaining_windows = self.config.stability_checks - self.stability_count
            return remaining_windows * self.config.window_size
            
        # otherwise use conservative estimate
        return min(self.config.max_steps - current_time, self.config.window_size * 2)


def test_convergence_tracker():
    """test convergence tracker with synthetic data"""
    
    config = ConvergenceConfig(
        min_steps=100,
        max_steps=2000,
        lyap_tolerance=0.05,
        window_size=50,
        stability_checks=2
    )
    
    tracker = ConvergenceTracker(config)
    
    # generate synthetic converging lyapunov data
    times = np.arange(0, 1000, 50)
    np.random.seed(42)
    
    # converging to lyap1=0.4, lyap2=-0.8
    target_lyap1, target_lyap2 = 0.4, -0.8
    
    lyap_history = []
    for i, t in enumerate(times):
        # add decreasing noise to simulate convergence
        noise_scale = 0.1 * np.exp(-i * 0.1)
        lyap1 = target_lyap1 + np.random.normal(0, noise_scale)
        lyap2 = target_lyap2 + np.random.normal(0, noise_scale)
        
        # compute ky dimension
        if lyap1 + lyap2 < 0 and lyap1 > 0:
            ky_dim = 1.0 + lyap1/abs(lyap2)
        else:
            ky_dim = 1.0
            
        lyap_history.append({
            'time': t,
            'mean_lyap1': lyap1,
            'mean_lyap2': lyap2,
            'mean_ky_dim': ky_dim
        })
        
        # check convergence
        converged = tracker.check_convergence(lyap_history)
        print(f"time {t}: lyap=({lyap1:.4f}, {lyap2:.4f}), "
              f"stable_count={tracker.stability_count}, converged={converged}")
        
        if converged:
            break
            
    info = tracker.get_convergence_info()
    print(f"\nfinal convergence info: {info}")
    
    assert tracker.converged or tracker.should_stop_early()
    print("convergence tracker tests passed")


if __name__ == "__main__":
    test_convergence_tracker() 