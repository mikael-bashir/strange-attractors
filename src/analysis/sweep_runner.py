"""
Parameter sweep runner - orchestrates the execution of parameter sweeps
"""

import os
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import cupy as cp

from .param_space import ParamSpace
from .convergence import ConvergenceTracker, ConvergenceConfig
from ..attractors.base import Attractor
from ..attractors import AVAILABLE_ATTRACTORS
from ..compute.cuda_backend import CUDABackend
from ..analysis.lyapunov import LyapunovTracker
from ..noise.types import NoiseConfig, NoiseType, generate_noise


@dataclass
class SweepConfig:
    attractor_name: str
    particles: int = 10000
    lyap_particles: int = 1000
    num_blocks: int = 256
    threads_per_block: int = 256
    convergence: ConvergenceConfig = None
    noise_config: Dict[str, Any] = None
    output_dir: str = "sweep_results"
    save_trajectories: bool = False
    save_lyapunov_history: bool = True
    checkpoint_freq: int = 10
    
    def __post_init__(self):
        if self.convergence is None:
            self.convergence = ConvergenceConfig()
        if self.noise_config is None:
            self.noise_config = {'type': 'none'}


@dataclass 
class ParamPointResult:
    """result for a single parameter point"""
    params: Dict[str, float]
    converged: bool
    steps_taken: int
    compute_time: float
    final_lyapunov: Tuple[float, float]
    kaplan_yorke_dim: Optional[float]
    early_stop_reason: Optional[str]
    lyapunov_history: Optional[List[Dict]] = None


class SweepResults:
    """container for sweep results and analysis"""
    
    def __init__(self, config: SweepConfig, param_space: ParamSpace):
        self.config = config
        self.param_space = param_space
        self.results: List[ParamPointResult] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
    def add_result(self, result: ParamPointResult):
        """add result for a parameter point"""
        self.results.append(result)
        
    def finalize(self):
        """finalize sweep results"""
        self.end_time = time.time()
        
    def get_summary(self) -> Dict[str, Any]:
        """get summary statistics"""
        if not self.results:
            return {}
            
        total_time = (self.end_time or time.time()) - self.start_time
        converged_count = sum(1 for r in self.results if r.converged)
        
        lyap1_values = [r.final_lyapunov[0] for r in self.results if r.converged]
        lyap2_values = [r.final_lyapunov[1] for r in self.results if r.converged]
        ky_values = [r.kaplan_yorke_dim for r in self.results 
                    if r.converged and r.kaplan_yorke_dim is not None]
        
        summary = {
            'total_param_points': len(self.results),
            'converged_points': converged_count,
            'convergence_rate': converged_count / len(self.results),
            'total_compute_time': total_time,
            'avg_time_per_point': total_time / len(self.results),
            'param_space_size': self.param_space.size(),
            'completion_rate': len(self.results) / self.param_space.size()
        }
        
        if lyap1_values:
            summary.update({
                'lyap1_stats': {
                    'mean': np.mean(lyap1_values),
                    'std': np.std(lyap1_values), 
                    'min': np.min(lyap1_values),
                    'max': np.max(lyap1_values)
                },
                'lyap2_stats': {
                    'mean': np.mean(lyap2_values),
                    'std': np.std(lyap2_values),
                    'min': np.min(lyap2_values),
                    'max': np.max(lyap2_values)
                }
            })
            
        if ky_values:
            summary['ky_dim_stats'] = {
                'mean': np.mean(ky_values),
                'std': np.std(ky_values),
                'min': np.min(ky_values),
                'max': np.max(ky_values)
            }
            
        return summary
        
    def save(self, filepath: str):
        """save results to file"""
        data = {
            'config': asdict(self.config),
            'param_space_config': self.param_space.config,
            'results': [asdict(r) for r in self.results],
            'summary': self.get_summary(),
            'metadata': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'param_space_size': self.param_space.size()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    @classmethod
    def load(cls, filepath: str) -> 'SweepResults':
        """load results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = SweepConfig(**data['config'])
        param_space = ParamSpace(data['param_space_config'])
        
        results = cls(config, param_space)
        results.start_time = data['metadata']['start_time']
        results.end_time = data['metadata']['end_time']
        
        for r_data in data['results']:
            result = ParamPointResult(**r_data)
            results.add_result(result)
            
        return results


class SweepRunner:
    """orchestrates parameter sweep execution"""
    
    def __init__(self, config: SweepConfig):
        self.config = config
        self.backend = CUDABackend(config.num_blocks, config.threads_per_block)
        self.attractor = AVAILABLE_ATTRACTORS[config.attractor_name]()
        os.makedirs(config.output_dir, exist_ok=True)
        
        # prepare noise config
        if isinstance(config.noise_config, dict):
            noise_type = NoiseType(config.noise_config.get('type', 'none'))
            noise_params = config.noise_config.get('params', {})
            self.noise_config = NoiseConfig(noise_type, noise_params)
        elif isinstance(config.noise_config, NoiseConfig):
            self.noise_config = config.noise_config
        else:
            # default to no noise
            self.noise_config = NoiseConfig(NoiseType.NONE, {})
            
    def run_single_point(self, params: Dict[str, float]) -> ParamPointResult:
        """run simulation for a single parameter point"""
        start_time = time.time()
        
        # update attractor parameters
        original_params = self.attractor.config.params.copy()
        self.attractor.config.params.update(params)
        
        # initialize particles
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
        
        lyap_tracker = None
        if (self.config.lyap_particles > 0 and 
            self.attractor.name in ["Hénon", "Clifford"]):
            lyap_tracker = LyapunovTracker(
                num_particles=min(self.config.lyap_particles, self.config.particles)
            )

        convergence_tracker = ConvergenceTracker(self.config.convergence)
        total_steps = 0
        converged = False
        while not converged and total_steps < self.config.convergence.max_steps:
            remaining_steps = self.config.convergence.max_steps - total_steps
            # batch size is at least 1/4 of window size, or remaining steps, or 500
            batch_size = min(self.config.convergence.window_size // 4, remaining_steps, 500)
            
            if batch_size <= 0:
                break
                
            # generate noise and run the batch
            noise_params = {}
            for param_name in self.attractor.param_names:
                noise_params[param_name] = generate_noise(batch_size, self.noise_config)
            self.backend.evolve_ensemble(
                self.attractor, particles_x, particles_y, 
                noise_params, batch_size, lyap_tracker
            )
            
            total_steps += batch_size
            if lyap_tracker is not None:
                lyap_tracker.total_time = total_steps
                mean_spectrum, max_spectrum = lyap_tracker.save_snapshot(
                    self.noise_config.get_range() if hasattr(self.noise_config, 'get_range') else 0.0
                )
                
                # has it converged?
                converged = convergence_tracker.check_convergence(lyap_tracker.history)
                
                if converged or convergence_tracker.should_stop_early():
                    break
                    
        compute_time = time.time() - start_time
        
        if lyap_tracker is not None and lyap_tracker.history:
            final_lyap = (lyap_tracker.history[-1]['mean_lyap1'], 
                         lyap_tracker.history[-1]['mean_lyap2'])
            ky_dim = lyap_tracker.history[-1].get('mean_ky_dim')
        else:
            final_lyap = (0.0, 0.0)
            ky_dim = None
            
        # restore original attractor params
        self.attractor.config.params = original_params
        
        return ParamPointResult(
            params=params.copy(),
            converged=convergence_tracker.converged,
            steps_taken=total_steps,
            compute_time=compute_time,
            final_lyapunov=final_lyap,
            kaplan_yorke_dim=ky_dim,
            early_stop_reason=convergence_tracker.early_stop_reason,
            lyapunov_history=lyap_tracker.history if self.config.save_lyapunov_history else None
        )
        
    def run(self, param_space: ParamSpace, resume_from: Optional[str] = None) -> SweepResults:
        """run complete parameter sweep"""
        
        # initialize or resume results
        if resume_from and os.path.exists(resume_from):
            print(f"resuming sweep from {resume_from}")
            results = SweepResults.load(resume_from)
            completed_params = {str(r.params) for r in results.results}
        else:
            print(f"starting new sweep with {param_space.size():,} parameter points")
            results = SweepResults(self.config, param_space)
            completed_params = set()
            
        # save sweep configuration
        config_file = os.path.join(self.config.output_dir, "sweep_config.json")
        with open(config_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'param_space': param_space.config
            }, f, indent=2, default=str)
            
        print(f"sweep config saved to {config_file}")
        
        # main sweep loop
        checkpoint_counter = 0
        for i, params in enumerate(param_space.iter_params()):
            
            # skip if already completed (for resume)
            if str(params) in completed_params:
                continue
                
            print(f"running param point {len(results.results)+1}/{param_space.size()}: {params}")
            
            try:
                result = self.run_single_point(params)
                results.add_result(result)
                
                # log progress
                conv_status = "converged" if result.converged else f"stopped ({result.early_stop_reason})"
                print(f"  → {conv_status} in {result.steps_taken} steps ({result.compute_time:.2f}s)")
                print(f"  → lyapunov: ({result.final_lyapunov[0]:.4f}, {result.final_lyapunov[1]:.4f})")
                
                # checkpoint periodically
                checkpoint_counter += 1
                if checkpoint_counter >= self.config.checkpoint_freq:
                    checkpoint_file = os.path.join(self.config.output_dir, "checkpoint.json")
                    results.save(checkpoint_file)
                    print(f"  → checkpoint saved")
                    checkpoint_counter = 0
                    
            except Exception as e:
                print(f"  → error: {e}")
                # create error result
                error_result = ParamPointResult(
                    params=params.copy(),
                    converged=False,
                    steps_taken=0,
                    compute_time=0.0,
                    final_lyapunov=(np.nan, np.nan),
                    kaplan_yorke_dim=None,
                    early_stop_reason=f"error: {str(e)}"
                )
                results.add_result(error_result)
                
        # finalize and save results
        results.finalize()
        final_file = os.path.join(self.config.output_dir, "final_results.json")
        results.save(final_file)
        
        # print summary
        summary = results.get_summary()
        print(f"\nsweep completed!")
        print(f"total points: {summary['total_param_points']}")
        print(f"converged: {summary['converged_points']} ({summary['convergence_rate']:.2%})")
        print(f"total time: {summary['total_compute_time']:.2f}s")
        print(f"avg time per point: {summary['avg_time_per_point']:.2f}s")
        print(f"results saved to {final_file}")
        
        return results


def test_sweep_runner():
    """test sweep runner with small parameter space"""
    
    # simple test config
    config = SweepConfig(
        attractor_name="henon",
        particles=1000,
        lyap_particles=100,
        convergence=ConvergenceConfig(
            min_steps=100,
            max_steps=1000,
            lyap_tolerance=0.1,
            window_size=50,
            stability_checks=2
        ),
        output_dir="test_sweep"
    )
    
    # small param space for testing
    param_space_config = {
        'parameters': {
            'a': {
                'type': 'linspace',
                'start': 1.35,
                'stop': 1.45,
                'num': 3
            },
            'b': {
                'type': 'fixed',
                'value': 0.3
            }
        }
    }
    
    param_space = ParamSpace(param_space_config)
    
    print(f"testing sweep runner with {param_space.size()} param points")
    
    runner = SweepRunner(config)
    results = runner.run(param_space)
    
    print(f"test completed with {len(results.results)} results")
    return results


if __name__ == "__main__":
    test_sweep_runner() 