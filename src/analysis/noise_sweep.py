"""
- looks in a noise range
- track regime transitions using confidence interval + deterministic intervals
- find α* where chaos↔sync boundaries occur
- plot results
"""

import os
import sys
import time
import json
import numpy as np
import cupy as cp
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

from analysis.regime_analyzer import RegimeAnalyzer, RegimeConfig, RegimeClassification
from attractors.base import Attractor
from attractors import AVAILABLE_ATTRACTORS
from compute.cuda_backend import CUDABackend
from analysis.lyapunov import LyapunovTracker
from noise.types import NoiseConfig, NoiseType, generate_noise


@dataclass
class ConvergenceConfig:
    """convergence detection for individual noise sweeps"""
    max_steps: int = 10000
    patience: int = 1000
    slope_threshold: float = 1e-5  # trend detection
    cv_threshold: float = 0.001  # coefficient of variation
    window_fractions: List[float] = None  # analysis windows
    
    def __post_init__(self):
        if self.window_fractions is None:
            self.window_fractions = [0.05, 0.20, 0.50]


@dataclass 
class NoiseSweepConfig:
    """configuration for noise sweep analysis"""
    # attractor settings (fixed across sweep)
    attractor_name: str
    base_params: Dict[str, float]
    
    # noise sweep settings
    noise_amplitudes: List[float]
    noise_type: str = "uniform"
    
    # simulation settings per noise point
    particles: int = 10000
    lyap_particles: int = 1000
    num_blocks: int = 256
    threads_per_block: int = 256
    
    # convergence per point
    convergence: ConvergenceConfig = None
    
    # regime analysis settings
    regime_config: RegimeConfig = None
    num_trajectories: int = 10  # CHANGED: 50 → 10
    use_deterministic_heuristic: bool = True  # NEW: enable smart filtering
    
    # output settings
    output_dir: str = "noise_sweep_results"
    save_detailed_data: bool = True
    
    def __post_init__(self):
        if self.convergence is None:
            self.convergence = ConvergenceConfig()
        if self.regime_config is None:
            self.regime_config = RegimeConfig()


@dataclass
class NoisePointResult:
    """comprehensive result for single noise amplitude"""
    noise_amplitude: float
    regime_classification: RegimeClassification
    converged: bool
    steps_taken: int
    compute_time: float
    kaplan_yorke_dim: Optional[float]
    confidence_analysis_quality: float  # overall r² from scaling fits
    deterministic_bounds_width: float
    confidence_interval_width: float
    interval_scaling_ratio: float  # how much tighter confidence vs deterministic
    trajectory_data: Optional[List[List[Dict]]] = None  # if save_detailed_data
    
    # visualization data
    snapshot_images: Optional[List[np.ndarray]] = None  # 2D histograms for animation
    pullback_steps: Optional[List[int]] = None  # steps where pullback snapshots were taken
    intermediate_interval: Optional[int] = None  # steps between intermediate snapshots
    bounds: Optional[Dict[str, Tuple[float, float]]] = None  # attractor bounds for plotting
    
    @property
    def is_chaotic(self) -> bool:
        return self.regime_classification.is_chaotic
    
    @property
    def is_synchronized(self) -> bool:
        return self.regime_classification.is_synchronized
    
    @property
    def is_hyperchaotic(self) -> bool:
        return self.regime_classification.is_hyperchaotic
    
    @property
    def has_strange_attractor(self) -> bool:
        return self.regime_classification.has_strange_attractor
    
    @property
    def regime_type(self) -> str:
        return self.regime_classification.regime_type


class ConvergenceTracker:
    """detect convergence for individual simulation runs"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.history: List[Dict] = []
        self.converged = False
        self.convergence_step = None
        self.patience_counter = 0
        
    def update(self, lyap_data: Dict) -> bool:
        """update with new lyapunov data, return True if converged"""
        self.history.append(lyap_data)
        
        if len(self.history) < 100:  # need minimum data
            return False
        
        # check multiple convergence criteria
        slope_converged = self._check_slope_convergence()
        cv_converged = self._check_cv_convergence()
        
        if slope_converged and cv_converged:
            if not self.converged:
                self.converged = True
                self.convergence_step = lyap_data['time']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # require patience period after initial convergence
            return self.patience_counter >= self.config.patience
        
        else:
            # reset if convergence lost
            if self.converged:
                self.converged = False
                self.patience_counter = 0
            return False
    
    def _check_slope_convergence(self) -> bool:
        """check if recent trend is flat enough"""
        if len(self.history) < 50:
            return False
        
        recent_data = self.history[-50:]
        times = np.array([h['time'] for h in recent_data])
        lyap1_values = np.array([h['mean_lyap1'] for h in recent_data])
        
        if len(times) < 3:
            return False
        
        slope, _ = np.polyfit(times, lyap1_values, 1)
        return abs(slope) < self.config.slope_threshold
    
    def _check_cv_convergence(self) -> bool:
        """check coefficient of variation in recent window"""
        if len(self.history) < 30:
            return False
        
        recent_data = self.history[-30:]
        lyap1_values = np.array([h['mean_lyap1'] for h in recent_data])
        
        mean_val = np.mean(lyap1_values)
        if abs(mean_val) < 1e-6:  # avoid division by zero
            return np.std(lyap1_values) < 1e-6
        
        cv = np.std(lyap1_values) / abs(mean_val)
        return cv < self.config.cv_threshold
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """summarize convergence analysis"""
        return {
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'total_steps': len(self.history),
            'final_slope': self._get_recent_slope() if len(self.history) > 10 else 0,
            'final_cv': self._get_recent_cv() if len(self.history) > 10 else 1
        }
    
    def _get_recent_slope(self) -> float:
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        times = np.array([h['time'] for h in recent])
        lyap1 = np.array([h['mean_lyap1'] for h in recent])
        if len(times) < 2:
            return 0
        slope, _ = np.polyfit(times, lyap1, 1)
        return slope
    
    def _get_recent_cv(self) -> float:
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        lyap1 = np.array([h['mean_lyap1'] for h in recent])
        mean_val = np.mean(lyap1)
        if abs(mean_val) < 1e-6:
            return 0 if np.std(lyap1) < 1e-6 else 1
        return np.std(lyap1) / abs(mean_val)


class NoiseSweepResults:
    """container for noise sweep results"""
    
    def __init__(self, config: NoiseSweepConfig):
        self.config = config
        self.results: List[NoisePointResult] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
    def add_result(self, result: NoisePointResult):
        """add result for noise amplitude"""
        self.results.append(result)
        
    def finalize(self):
        """finalize sweep"""
        self.end_time = time.time()
        
    def find_critical_transitions(self) -> Dict[str, Optional[float]]:
        """find all critical transition points"""
        if not self.results:
            return {}
        
        sorted_results = sorted(self.results, key=lambda r: r.noise_amplitude)
        
        transitions = {
            'chaos_to_sync': self._find_transition(
                lambda r: r.is_chaotic, lambda r: r.is_synchronized, sorted_results
            ),
            'sync_to_chaos': self._find_transition(
                lambda r: r.is_synchronized, lambda r: r.is_chaotic, sorted_results
            ),
            'strange_to_periodic': self._find_transition(
                lambda r: r.has_strange_attractor, lambda r: not r.has_strange_attractor, sorted_results
            ),
            'hyperchaos_threshold': self._find_transition(
                lambda r: r.is_chaotic and not r.is_hyperchaotic, lambda r: r.is_hyperchaotic, sorted_results
            )
        }
        
        return transitions
    
    def _find_transition(self, before_condition: Callable, after_condition: Callable, 
                        sorted_results: List[NoisePointResult]) -> Optional[float]:
        """find transition point using interpolation"""
        for i in range(len(sorted_results) - 1):
            curr = sorted_results[i]
            next_res = sorted_results[i + 1]
            
            if before_condition(curr) and after_condition(next_res):
                # interpolate using λ₁ bounds
                eps1, bounds1 = curr.noise_amplitude, curr.regime_classification.lambda1_bounds
                eps2, bounds2 = next_res.noise_amplitude, next_res.regime_classification.lambda1_bounds
                
                # use center of bounds for interpolation
                lam1 = (bounds1[0] + bounds1[1]) / 2
                lam2 = (bounds2[0] + bounds2[1]) / 2
                
                if lam2 != lam1:
                    t = -lam1 / (lam2 - lam1)
                    critical_eps = eps1 + t * (eps2 - eps1)
                    return max(eps1, min(eps2, critical_eps))  # clamp to interval
        
        return None
    
    def get_regime_phase_diagram(self) -> Dict[str, List[float]]:
        """categorize noise amplitudes by regime"""
        phases = {
            'synchronized': [],
            'chaotic': [],
            'hyperchaotic': [],
            'strange_attractor': [],
            'uncertain': [],
            'undefined': []
        }
        
        for result in self.results:
            regime = result.regime_type
            noise_amp = result.noise_amplitude
            
            if result.is_synchronized:
                phases['synchronized'].append(noise_amp)
            elif result.is_hyperchaotic:
                phases['hyperchaotic'].append(noise_amp)
            elif result.is_chaotic:
                phases['chaotic'].append(noise_amp)
            
            if result.has_strange_attractor:
                phases['strange_attractor'].append(noise_amp)
            
            if 'uncertain' in regime:
                phases['uncertain'].append(noise_amp)
            elif 'undefined' in regime:
                phases['undefined'].append(noise_amp)
        
        return phases
    
    def get_bounds_scaling_analysis(self) -> Dict[str, Any]:
        """analyze how bounds scale with noise"""
        if not self.results:
            return {}
        
        noise_amps = [r.noise_amplitude for r in self.results]
        det_widths = [r.deterministic_bounds_width for r in self.results]
        conf_widths = [r.confidence_interval_width for r in self.results]
        ratios = [r.interval_scaling_ratio for r in self.results]
        
        return {
            'noise_amplitudes': noise_amps,
            'deterministic_widths': det_widths,
            'confidence_widths': conf_widths,
            'scaling_ratios': ratios,
            'avg_scaling_advantage': np.mean(ratios),
            'scaling_advantage_range': (min(ratios), max(ratios))
        }
    
    def save(self, filepath: str):
        """save comprehensive results"""
        results_data = {
            'config': asdict(self.config),
            'results': [asdict(r) for r in self.results],
            'critical_transitions': self.find_critical_transitions(),
            'phase_diagram': self.get_regime_phase_diagram(),
            'bounds_scaling': self.get_bounds_scaling_analysis(),
            'runtime': self.end_time - self.start_time if self.end_time else None,
            'timestamp': self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    @classmethod 
    def load(cls, filepath: str) -> 'NoiseSweepResults':
        """load results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # reconstruct objects
        config = NoiseSweepConfig(**data['config'])
        results_obj = cls(config)
        
        for result_data in data['results']:
            # reconstruct RegimeClassification
            regime_data = result_data['regime_classification']
            regime_classification = RegimeClassification(**regime_data)
            
            # reconstruct NoisePointResult
            result_data['regime_classification'] = regime_classification
            result = NoisePointResult(**result_data)
            results_obj.add_result(result)
        
        results_obj.start_time = data['timestamp']
        results_obj.end_time = data.get('runtime', 0) + results_obj.start_time
        
        return results_obj


class NoiseSweepRunner:
    """noise sweep with regime analysis and critical transition detection"""
    
    def __init__(self, config: NoiseSweepConfig):
        self.config = config
        self.attractor = AVAILABLE_ATTRACTORS[config.attractor_name]()
        self.backend = CUDABackend()
        
    def run_single_amplitude(self, noise_amplitude: float) -> NoisePointResult:
        """run analysis for single noise amplitude"""
        print(f"  analyzing α = {noise_amplitude:.6f}...")
        start_time = time.time()
        
        # setup noise configuration
        print(f"    setting up noise config (type: {self.config.noise_type})")
        noise_config = NoiseConfig(
            NoiseType.UNIFORM if self.config.noise_type == "uniform" else NoiseType.GAUSSIAN,
            {'low': -noise_amplitude, 'high': noise_amplitude} if self.config.noise_type == "uniform"
            else {'mean': 0, 'std': noise_amplitude}
        )
        
        # run multiple independent trajectories for regime analysis
        print(f"    running {self.config.num_trajectories} independent trajectories...")
        regime_analyzer = RegimeAnalyzer(self.config.regime_config)
        trajectory_data = []
        viz_data = None
        
        for traj_idx in range(self.config.num_trajectories):
            print(f"      trajectory {traj_idx+1}/{self.config.num_trajectories}...", end=" ")
            # run single trajectory
            lyap_history, traj_viz_data = self._run_single_trajectory(noise_config, traj_idx)
            regime_analyzer.add_trajectory(lyap_history)
            print(f"✓ ({len(lyap_history)} points)")
            
            if self.config.save_detailed_data:
                trajectory_data.append(lyap_history)
            
            if traj_viz_data is not None:
                viz_data = traj_viz_data
        
        # comprehensive regime analysis
        print(f"    performing regime analysis...")
        regime_analysis = regime_analyzer.analyze_regime(
            self.config.base_params, noise_amplitude, 
            use_deterministic_heuristic=self.config.use_deterministic_heuristic
        )
        
        # extract key metrics
        classification = regime_analysis['regime_classification']
        ky_analysis = regime_analysis['kaplan_yorke_analysis']
        scaling_comparison = regime_analysis['scaling_comparison']
        
        compute_time = time.time() - start_time
        print(f"    completed in {compute_time:.1f}s")
        
        return NoisePointResult(
            noise_amplitude=noise_amplitude,
            regime_classification=classification,
            converged=True,  # assume convergence from multi-trajectory approach
            steps_taken=self.config.convergence.max_steps,  # approximation
            compute_time=compute_time,
            kaplan_yorke_dim=ky_analysis.get('ky_dimension'),
            confidence_analysis_quality=regime_analysis['confidence_analysis'].get('scaling_diagnostics', {}).get('overall_quality', 0),
            deterministic_bounds_width=scaling_comparison.get('deterministic_width', np.inf),
            confidence_interval_width=scaling_comparison.get('confidence_interval_width', np.inf),
            interval_scaling_ratio=scaling_comparison.get('confidence_tighter_by_factor', 1),
            trajectory_data=trajectory_data if self.config.save_detailed_data else None,
            snapshot_images=viz_data['snapshot_images'] if viz_data else None,
            pullback_steps=viz_data['pullback_steps'] if viz_data else None,
            intermediate_interval=viz_data['intermediate_interval'] if viz_data else None,
            bounds=viz_data['bounds'] if viz_data else None
        )
    
    def _run_single_trajectory(self, noise_config: NoiseConfig, traj_idx: int) -> Tuple[List[Dict], Optional[Dict]]:
        """run single trajectory with convergence detection and visualization"""
        
        # initialize particles
        print(f"init...", end="")
        np.random.seed(42 + traj_idx)  # seeding for reproducibility
        cp.random.seed(42 + traj_idx)
        particles_x = cp.random.uniform(
            self.attractor.init_bounds['x'][0],
            self.attractor.init_bounds['x'][1],
            self.config.particles,
            dtype=cp.float32
        )
        particles_y = cp.random.uniform(
            self.attractor.init_bounds['y'][0],
            self.attractor.init_bounds['y'][1],
            self.config.particles,
            dtype=cp.float32
        )
        
        # lyapunov tracking
        print(f"lyap...", end="")
        lyap_tracker = LyapunovTracker(
            num_particles=min(self.config.lyap_particles, self.config.particles)
        )
        
        # convergence tracking
        convergence_tracker = ConvergenceTracker(self.config.convergence)
        
        # visualization setup
        viz_data = None
        if traj_idx == 0:  # only capture snapshots for first trajectory
            viz_data = {
                'snapshot_images': [],
                'pullback_steps': [0],  # start with initial state
                'intermediate_interval': 200,  # steps between intermediate snapshots
                'bounds': {
                    'x': self.attractor.bounds['x'],
                    'y': self.attractor.bounds['y']
                }
            }
            
            # capture initial state
            x_cpu = cp.asnumpy(particles_x)
            y_cpu = cp.asnumpy(particles_y)
            hist, _, _ = np.histogram2d(
                x_cpu, y_cpu,
                bins=(600, 600),  # high resolution for animation
                range=[
                    self.attractor.bounds['x'],
                    self.attractor.bounds['y']
                ]
            )
            viz_data['snapshot_images'].append(hist.T)  # transpose for correct orientation
        
        # main evolution loop
        print(f"evolve...", end="")
        step = 0
        lyap_save_interval = 100
        
        while step < self.config.convergence.max_steps:
            # progress indicator every 1000 steps
            if step % 1000 == 0 and step > 0:
                print(f"{step//1000}k.", end="")
            
            # generate noise for this chunk (100 steps for massive speedup while enabling lyapunov renormalization)
            chunk_size = 100
            remaining_steps = self.config.convergence.max_steps - step
            actual_chunk = min(chunk_size, remaining_steps)
            
            # generate 1D noise arrays for all attractor parameters
            noise_params = {}
            for param_name in self.config.base_params.keys():
                noise_params[param_name] = generate_noise(actual_chunk, noise_config)
            
            # evolve system
            self.backend.evolve_ensemble(
                self.attractor, particles_x, particles_y, 
                noise_params, actual_chunk, lyap_tracker
            )
            
            # update lyapunov tracking
            step += actual_chunk
            lyap_tracker.total_time = step
            
            # periodic analysis and visualization
            if step % lyap_save_interval == 0 and step > 0:
                mean_spectrum, max_spectrum = lyap_tracker.save_snapshot(
                    noise_range=max(abs(noise_config.params.get('low', 0)), 
                                  abs(noise_config.params.get('high', 0)),
                                  noise_config.params.get('std', 0))
                )
                
                # check convergence
                if lyap_tracker.history:
                    if convergence_tracker.update(lyap_tracker.history[-1]):
                        print(f"converged@{step}.", end="")
                        break
                
                # capture snapshot for visualization
                if viz_data is not None and step % 200 == 0:  # adjust frequency as needed
                    x_cpu = cp.asnumpy(particles_x)
                    y_cpu = cp.asnumpy(particles_y)
                    hist, _, _ = np.histogram2d(
                        x_cpu, y_cpu,
                        bins=(600, 600),
                        range=[
                            self.attractor.bounds['x'],
                            self.attractor.bounds['y']
                        ]
                    )
                    viz_data['snapshot_images'].append(hist.T)
                    viz_data['pullback_steps'].append(step)
        
        return lyap_tracker.history, viz_data
    
    def run(self) -> NoiseSweepResults:
        """run full noise sweep with critical transition detection"""
        print(f"noise sweep: {len(self.config.noise_amplitudes)} noise amplitudes")
        print(f"attractor: {self.config.attractor_name} with params {self.config.base_params}")
        print(f"trajectories per amplitude: {self.config.num_trajectories}")
        print()
        
        results = NoiseSweepResults(self.config)
        
        for i, noise_amp in enumerate(self.config.noise_amplitudes):
            print(f"[{i+1}/{len(self.config.noise_amplitudes)}] ", end="")
            
            try:
                result = self.run_single_amplitude(noise_amp)
                results.add_result(result)
                
                # progress update
                regime = result.regime_classification.regime_type
                confidence = result.regime_classification.confidence_level * 100
                print(f"    → {regime} ({confidence:.0f}% confidence)")
                
                if result.kaplan_yorke_dim is not None:
                    print(f"    → D_KY = {result.kaplan_yorke_dim:.3f}")
                
            except Exception as e:
                print(f"    → ERROR: {e}")
        
        results.finalize()
        
        # analyze critical transitions
        transitions = results.find_critical_transitions()
        print(f"\ncritical transitions found:")
        for transition_type, critical_noise in transitions.items():
            if critical_noise is not None:
                print(f"  {transition_type}: α* ≈ {critical_noise:.6f}")
            else:
                print(f"  {transition_type}: not detected")
        
        return results


# convenience functions
def quick_noise_sweep(attractor_name: str,
                     attractor_params: Dict[str, float], 
                     noise_range: Tuple[float, float],
                     num_points: int = 20,
                     num_trajectories: int = 30) -> NoiseSweepResults:
    """quick noise sweep for critical transition detection"""
    
    noise_amplitudes = np.linspace(noise_range[0], noise_range[1], num_points)
    
    config = NoiseSweepConfig(
        attractor_name=attractor_name,
        base_params=attractor_params,
        noise_amplitudes=noise_amplitudes,
        num_trajectories=num_trajectories,
        particles=5000,  # reduced for speed
        lyap_particles=500
    )
    
    runner = NoiseSweepRunner(config)
    return runner.run()


def find_chaos_sync_boundary(attractor_name: str,
                            attractor_params: Dict[str, float],
                            noise_range: Tuple[float, float],
                            tolerance: float = 1e-4,
                            max_iterations: int = 10) -> Optional[float]:
    """binary search for precise chaos↔sync boundary"""
    
    low, high = noise_range
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2
        
        # test regime at midpoint
        result = quick_noise_sweep(attractor_name, attractor_params, (mid, mid), 1, 20)
        
        if not result.results:
            break
            
        regime = result.results[0].regime_classification.regime_type
        
        if "chaos" in regime:
            low = mid
        elif "sync" in regime:
            high = mid
        else:
            # uncertain - need more data or different approach
            break
            
        if abs(high - low) < tolerance:
            return (low + high) / 2
    
    return (low + high) / 2 if abs(high - low) < tolerance * 10 else None 
