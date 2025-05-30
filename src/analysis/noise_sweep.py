"""
Noise amplitude sweep - find critical noise thresholds for fiber collapse
"""

import os
import sys
import time
import json
import numpy as np
import cupy as cp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# handle imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    # try relative imports first (when run as module)
    from .convergence import ConvergenceTracker, ConvergenceConfig
    from ..attractors.base import Attractor
    from ..attractors import AVAILABLE_ATTRACTORS
    from ..compute.cuda_backend import CUDABackend
    from ..analysis.lyapunov import LyapunovTracker
    from ..noise.types import NoiseConfig, NoiseType, generate_noise
except ImportError:
    # fallback to absolute imports (when run directly)
    from src.analysis.convergence import ConvergenceTracker, ConvergenceConfig
    from src.attractors.base import Attractor
    from src.attractors import AVAILABLE_ATTRACTORS
    from src.compute.cuda_backend import CUDABackend
    from src.analysis.lyapunov import LyapunovTracker
    from src.noise.types import NoiseConfig, NoiseType, generate_noise


@dataclass
class NoiseSweepConfig:
    """configuration for noise amplitude sweep"""
    # attractor settings (FIXED)
    attractor_name: str
    base_params: Dict[str, float]  # fixed base parameters
    
    # noise amplitude sweep settings
    noise_amplitudes: List[float]  # amplitudes to test
    noise_type: str = "uniform"  # uniform or gaussian
    
    # compute settings
    particles: int = 10000
    lyap_particles: int = 1000
    num_blocks: int = 256
    threads_per_block: int = 256
    
    # convergence settings
    convergence: ConvergenceConfig = None
    
    # output settings
    output_dir: str = "noise_sweep_results"
    save_lyapunov_history: bool = True
    
    # noise sequence settings
    sequence_length: int = 10000  # length of base noise sequence
    random_seed: int = 42  # for reproducible sequences
    
    def __post_init__(self):
        if self.convergence is None:
            self.convergence = ConvergenceConfig()


@dataclass
class NoisePointResult:
    """result for a single noise amplitude"""
    noise_amplitude: float
    converged: bool
    steps_taken: int
    compute_time: float
    final_lyapunov: Tuple[float, float]
    kaplan_yorke_dim: Optional[float]
    early_stop_reason: Optional[str]
    lyapunov_history: Optional[List[Dict]] = None
    
    @property
    def is_chaotic(self) -> bool:
        """check if system is chaotic (lambda1 > 0)"""
        return self.final_lyapunov[0] > 0
        
    @property
    def is_synchronized(self) -> bool:
        """check if system is synchronized (lambda1 < 0)"""
        return self.final_lyapunov[0] < 0
    
    @property
    def is_hyperchaotic(self) -> bool:
        """check if system is hyperchaotic (both exponents positive)"""
        return self.final_lyapunov[0] > 0 and self.final_lyapunov[1] > 0
    
    @property
    def is_strange(self) -> bool:
        """check if attractor is strange (fractional KY dimension)"""
        if self.kaplan_yorke_dim is None:
            return False
        return not np.isclose(self.kaplan_yorke_dim, round(self.kaplan_yorke_dim), atol=0.01)
    
    @property
    def attractor_type(self) -> str:
        """classify attractor type based on lyapunov spectrum"""
        l1, l2 = self.final_lyapunov
        
        if np.isnan(l1) or np.isnan(l2):
            return "undefined"
        elif l1 > 0 and l2 > 0:
            return "hyperchaotic"
        elif l1 > 0 and l2 < 0:
            return "chaotic"
        elif l1 < 0 and l2 < 0:
            return "synchronized"
        elif l1 == 0:
            return "marginal"
        else:
            return "unknown"


class NoiseSweepResults:
    """container for noise sweep results"""
    
    def __init__(self, config: NoiseSweepConfig):
        self.config = config
        self.results: List[NoisePointResult] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.base_noise_sequence: Optional[np.ndarray] = None
        
    def add_result(self, result: NoisePointResult):
        """add result for a noise amplitude"""
        self.results.append(result)
        
    def finalize(self):
        """finalize sweep results"""
        self.end_time = time.time()
        
    def find_critical_noise(self) -> Optional[float]:
        """estimate critical noise threshold where lambda1 crosses zero (chaos→sync)"""
        return self._find_transition(lambda r: r.is_chaotic, lambda r: r.is_synchronized)
    
    def find_hyperchaos_threshold(self) -> Optional[float]:
        """find threshold where system becomes hyperchaotic (both λ > 0)"""
        return self._find_transition(lambda r: r.is_chaotic and not r.is_hyperchaotic, 
                                   lambda r: r.is_hyperchaotic)
    
    def find_strangeness_threshold(self) -> Optional[float]:
        """find threshold where attractor loses strangeness (fractional→integer KY dim)"""
        return self._find_transition(lambda r: r.is_strange, lambda r: not r.is_strange)
    
    def _find_transition(self, before_condition, after_condition) -> Optional[float]:
        """generic transition finder using linear interpolation"""
        if not self.results:
            return None
            
        sorted_results = sorted(self.results, key=lambda r: r.noise_amplitude)
        
        for i in range(len(sorted_results) - 1):
            curr = sorted_results[i]
            next_res = sorted_results[i + 1]
            
            if before_condition(curr) and after_condition(next_res):
                # use λ₁ for interpolation (most transitions involve it)
                eps1, lam1 = curr.noise_amplitude, curr.final_lyapunov[0]
                eps2, lam2 = next_res.noise_amplitude, next_res.final_lyapunov[0]
                
                if lam2 != lam1:
                    t = -lam1 / (lam2 - lam1)
                    critical_eps = eps1 + t * (eps2 - eps1)
                    return critical_eps
                    
        return None
    
    def get_phase_diagram(self) -> Dict[str, List[float]]:
        """categorize noise points by attractor type"""
        phases = {
            'synchronized': [],
            'chaotic': [],
            'hyperchaotic': [],
            'marginal': [],
            'undefined': []
        }
        
        for result in self.results:
            if result.converged:
                phases[result.attractor_type].append(result.noise_amplitude)
                
        return phases
        
    def get_summary(self) -> Dict[str, Any]:
        """get comprehensive summary with all transitions"""
        if not self.results:
            return {}
            
        total_time = (self.end_time or time.time()) - self.start_time
        converged_count = sum(1 for r in self.results if r.converged)
        
        # basic statistics
        converged_results = [r for r in self.results if r.converged]
        noise_amps = [r.noise_amplitude for r in converged_results]
        lyap1_values = [r.final_lyapunov[0] for r in converged_results]
        lyap2_values = [r.final_lyapunov[1] for r in converged_results]
        ky_values = [r.kaplan_yorke_dim for r in converged_results if r.kaplan_yorke_dim is not None]
        
        # phase transitions
        critical_transitions = {
            'chaos_to_sync': self.find_critical_noise(),
            'chaos_to_hyperchaos': self.find_hyperchaos_threshold(),
            'strange_to_regular': self.find_strangeness_threshold()
        }
        
        # phase diagram
        phase_diagram = self.get_phase_diagram()
        
        summary = {
            'total_noise_points': len(self.results),
            'converged_points': converged_count,
            'convergence_rate': converged_count / len(self.results),
            'total_compute_time': total_time,
            'avg_time_per_point': total_time / len(self.results),
            'noise_range': [min(noise_amps), max(noise_amps)] if noise_amps else None,
            'critical_transitions': critical_transitions,
            'phase_diagram': phase_diagram
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
            summary.update({
                'kaplan_yorke_stats': {
                    'mean': np.mean(ky_values),
                    'std': np.std(ky_values),
                    'min': np.min(ky_values),
                    'max': np.max(ky_values),
                    'strange_fraction': sum(1 for r in converged_results if r.is_strange) / len(converged_results)
                }
            })
            
        return summary
        
    def save(self, filepath: str):
        """save results to file"""
        data = {
            'config': asdict(self.config),
            'results': [asdict(r) for r in self.results],
            'summary': self.get_summary(),
            'metadata': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'base_noise_sequence_shape': self.base_noise_sequence.shape if self.base_noise_sequence is not None else None
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    @classmethod
    def load(cls, filepath: str) -> 'NoiseSweepResults':
        """load results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        config = NoiseSweepConfig(**data['config'])
        results = cls(config)
        results.start_time = data['metadata']['start_time']
        results.end_time = data['metadata']['end_time']
        
        for r_data in data['results']:
            result = NoisePointResult(**r_data)
            results.add_result(result)
            
        return results


class NoiseSweepRunner:
    """orchestrates noise amplitude sweep execution"""
    
    def __init__(self, config: NoiseSweepConfig):
        self.config = config
        self.backend = CUDABackend(config.num_blocks, config.threads_per_block)
        self.attractor = AVAILABLE_ATTRACTORS[config.attractor_name]()
        
        # set fixed base parameters
        self.attractor.config.params.update(config.base_params)
        
        # setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # generate fixed base noise sequence
        np.random.seed(config.random_seed)
        if config.noise_type == "uniform":
            self.base_noise_sequence = np.random.uniform(-1.0, 1.0, config.sequence_length)
        elif config.noise_type == "gaussian":
            self.base_noise_sequence = np.random.normal(0.0, 1.0, config.sequence_length)
        else:
            raise ValueError(f"unsupported noise type: {config.noise_type}")
            
        print(f"generated base noise sequence with {len(self.base_noise_sequence)} samples")
        print(f"attractor: {self.attractor.name} with params {self.attractor.params}")
        
    def run_single_amplitude(self, noise_amplitude: float) -> NoisePointResult:
        """run simulation for a single noise amplitude"""
        start_time = time.time()
        
        print(f"  testing noise amplitude: {noise_amplitude}")
        
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
        
        # setup lyapunov tracker
        lyap_tracker = None
        if (self.config.lyap_particles > 0 and 
            self.attractor.name in ["Hénon", "Clifford"]):
            lyap_tracker = LyapunovTracker(
                num_particles=min(self.config.lyap_particles, self.config.particles)
            )
            
        # setup convergence tracker
        convergence_tracker = ConvergenceTracker(self.config.convergence)
        
        # adaptive simulation loop
        total_steps = 0
        converged = False
        noise_index = 0
        
        while not converged and total_steps < self.config.convergence.max_steps:
            # determine batch size for this iteration
            remaining_steps = self.config.convergence.max_steps - total_steps
            batch_size = min(self.config.convergence.window_size // 4, remaining_steps, 500)
            
            if batch_size <= 0:
                break
                
            # get scaled noise for this batch
            noise_end = min(noise_index + batch_size, len(self.base_noise_sequence))
            actual_batch_size = noise_end - noise_index
            
            if actual_batch_size <= 0:
                print(f"    ran out of noise sequence at step {total_steps}")
                break
                
            # scale base noise by amplitude
            scaled_noise = noise_amplitude * self.base_noise_sequence[noise_index:noise_end]
            
            # create noise params for GPU
            noise_params = {}
            for param_name in self.attractor.param_names:
                noise_params[param_name] = cp.array(scaled_noise, dtype=cp.float32)
                
            # run simulation batch
            self.backend.evolve_ensemble(
                self.attractor, particles_x, particles_y, 
                noise_params, actual_batch_size, lyap_tracker
            )
            
            total_steps += actual_batch_size
            noise_index += actual_batch_size
            
            # update lyapunov tracker
            if lyap_tracker is not None:
                lyap_tracker.total_time = total_steps
                mean_spectrum, max_spectrum = lyap_tracker.save_snapshot(noise_amplitude)
                
                # check convergence
                converged = convergence_tracker.check_convergence(lyap_tracker.history)
                
                if converged or convergence_tracker.should_stop_early():
                    break
                    
        # compute final results
        compute_time = time.time() - start_time
        
        if lyap_tracker is not None and lyap_tracker.history:
            final_lyap = (lyap_tracker.history[-1]['mean_lyap1'], 
                         lyap_tracker.history[-1]['mean_lyap2'])
            ky_dim = lyap_tracker.history[-1].get('mean_ky_dim')
        else:
            final_lyap = (0.0, 0.0)
            ky_dim = None
            
        return NoisePointResult(
            noise_amplitude=noise_amplitude,
            converged=convergence_tracker.converged,
            steps_taken=total_steps,
            compute_time=compute_time,
            final_lyapunov=final_lyap,
            kaplan_yorke_dim=ky_dim,
            early_stop_reason=convergence_tracker.early_stop_reason,
            lyapunov_history=lyap_tracker.history if self.config.save_lyapunov_history else None
        )
        
    def run(self) -> NoiseSweepResults:
        """run complete noise amplitude sweep"""
        
        print(f"starting noise amplitude sweep with {len(self.config.noise_amplitudes)} points")
        results = NoiseSweepResults(self.config)
        results.base_noise_sequence = self.base_noise_sequence
        
        # save sweep configuration
        config_file = os.path.join(self.config.output_dir, "noise_sweep_config.json")
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
            
        print(f"sweep config saved to {config_file}")
        
        # main sweep loop
        for i, noise_amp in enumerate(self.config.noise_amplitudes):
            print(f"running noise point {i+1}/{len(self.config.noise_amplitudes)}: ε={noise_amp}")
            
            try:
                result = self.run_single_amplitude(noise_amp)
                results.add_result(result)
                
                # enhanced progress logging
                conv_status = "converged" if result.converged else f"stopped ({result.early_stop_reason})"
                print(f"  → {conv_status} in {result.steps_taken} steps ({result.compute_time:.2f}s)")
                print(f"  → lyapunov: ({result.final_lyapunov[0]:.4f}, {result.final_lyapunov[1]:.4f}) [{result.attractor_type.upper()}]")
                
                if result.kaplan_yorke_dim is not None:
                    strange_status = "strange" if result.is_strange else "regular"
                    print(f"  → kaplan-yorke: {result.kaplan_yorke_dim:.3f} [{strange_status}]")
                
            except Exception as e:
                print(f"  → error: {e}")
                # create error result
                error_result = NoisePointResult(
                    noise_amplitude=noise_amp,
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
        final_file = os.path.join(self.config.output_dir, "noise_sweep_results.json")
        results.save(final_file)
        
        # enhanced summary with all transitions
        summary = results.get_summary()
        print(f"\nnoise amplitude sweep completed!")
        print(f"total points: {summary['total_noise_points']}")
        print(f"converged: {summary['converged_points']} ({summary['convergence_rate']:.2%})")
        print(f"total time: {summary['total_compute_time']:.2f}s")
        print(f"avg time per point: {summary['avg_time_per_point']:.2f}s")
        
        # critical transitions
        transitions = summary['critical_transitions']
        print(f"\n=== phase transitions ===")
        
        if transitions['chaos_to_sync']:
            print(f"🎯 chaos→sync: εc ≈ {transitions['chaos_to_sync']:.4f}")
        if transitions['chaos_to_hyperchaos']:
            print(f"🌪️  chaos→hyperchaos: εh ≈ {transitions['chaos_to_hyperchaos']:.4f}")
        if transitions['strange_to_regular']:
            print(f"🔸 strange→regular: εs ≈ {transitions['strange_to_regular']:.4f}")
            
        if not any(transitions.values()):
            noise_range = summary['noise_range']
            if noise_range:
                print(f"⚠️  no transitions found in range [{noise_range[0]:.3f}, {noise_range[1]:.3f}]")
            else:
                print(f"⚠️  no valid results to analyze")
                
        # phase diagram
        phases = summary['phase_diagram']
        print(f"\n=== phase diagram ===")
        for phase, noise_list in phases.items():
            if noise_list:
                print(f"{phase}: {len(noise_list)} points (ε ∈ [{min(noise_list):.3f}, {max(noise_list):.3f}])")
                
        if 'kaplan_yorke_stats' in summary:
            ky_stats = summary['kaplan_yorke_stats']
            print(f"\nstrangeness: {ky_stats['strange_fraction']:.1%} of attractors have fractional KY dimension")
            print(f"ky range: [{ky_stats['min']:.3f}, {ky_stats['max']:.3f}] (mean: {ky_stats['mean']:.3f})")
        
        print(f"\nresults saved to {final_file}")
        
        return results


def interactive_cli():
    """interactive cli for noise amplitude sweeps"""
    
    print("=== noise amplitude sweep tool ===")
    print("find critical noise thresholds for fiber collapse\n")
    
    # select attractor
    available = list(AVAILABLE_ATTRACTORS.keys())
    print(f"available attractors: {', '.join(available)}")
    
    while True:
        attractor = input("select attractor [henon]: ").strip().lower()
        if not attractor:
            attractor = "henon"
        if attractor in available:
            break
        print(f"invalid choice. options: {', '.join(available)}")
    
    # get base parameters
    print(f"\nbase parameters for {attractor}:")
    base_params = {}
    
    if attractor == "henon":
        a = input("a parameter [1.4]: ").strip()
        b = input("b parameter [0.3]: ").strip()
        base_params['a'] = float(a) if a else 1.4
        base_params['b'] = float(b) if b else 0.3
        
    elif attractor == "clifford":
        a = input("a parameter [-1.4]: ").strip()
        b = input("b parameter [1.6]: ").strip()
        c = input("c parameter [1.0]: ").strip()
        d = input("d parameter [0.7]: ").strip()
        base_params['a'] = float(a) if a else -1.4
        base_params['b'] = float(b) if b else 1.6
        base_params['c'] = float(c) if c else 1.0
        base_params['d'] = float(d) if d else 0.7
        
    elif attractor == "ikeda":
        u = input("u parameter [0.918]: ").strip()
        base_params['u'] = float(u) if u else 0.918
    
    # noise amplitude range
    print(f"\nnoise amplitude sweep:")
    min_eps = input("minimum noise amplitude [0.01]: ").strip()
    max_eps = input("maximum noise amplitude [2.0]: ").strip()
    num_points = input("number of points [10]: ").strip()
    
    min_eps = float(min_eps) if min_eps else 0.01
    max_eps = float(max_eps) if max_eps else 2.0
    num_points = int(num_points) if num_points else 10
    
    noise_amplitudes = np.linspace(min_eps, max_eps, num_points).tolist()
    
    # output directory
    output_dir = input("output directory [noise_sweep_results]: ").strip()
    if not output_dir:
        output_dir = "noise_sweep_results"
    
    # optional advanced settings
    print(f"\nadvanced settings (press enter for defaults):")
    particles = input("total particles [10000]: ").strip()
    lyap_particles = input("lyapunov particles [1000]: ").strip()
    max_steps = input("max steps [2000]: ").strip()
    
    particles = int(particles) if particles else 10000
    lyap_particles = int(lyap_particles) if lyap_particles else 1000
    max_steps = int(max_steps) if max_steps else 2000
    
    # create config
    config = NoiseSweepConfig(
        attractor_name=attractor,
        base_params=base_params,
        noise_amplitudes=noise_amplitudes,
        noise_type="uniform",
        particles=particles,
        lyap_particles=lyap_particles,
        convergence=ConvergenceConfig(
            min_steps=100,
            max_steps=max_steps,
            lyap_tolerance=0.1,
            window_size=50,
            stability_checks=2
        ),
        output_dir=output_dir,
        sequence_length=10000,
        random_seed=42
    )
    
    # confirm settings
    print(f"\n=== sweep configuration ===")
    print(f"attractor: {attractor}")
    print(f"base params: {base_params}")
    print(f"noise range: ε ∈ [{min_eps:.3f}, {max_eps:.3f}] ({num_points} points)")
    print(f"particles: {particles} total, {lyap_particles} for lyapunov")
    print(f"max steps: {max_steps}")
    print(f"output: {output_dir}/")
    
    confirm = input("\nproceed with sweep? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("cancelled.")
        return
    
    # run sweep
    print(f"\nstarting noise amplitude sweep...")
    try:
        runner = NoiseSweepRunner(config)
        results = runner.run()
        
        print(f"\n=== detailed results ===")
        for result in results.results:
            status_icon = {"chaotic": "🌀", "hyperchaotic": "🌪️", "synchronized": "🔒", "marginal": "⚖️", "undefined": "❓"}.get(result.attractor_type, "❓")
            conv = "✓" if result.converged else "✗"
            
            output = f"  ε={result.noise_amplitude:.3f}: λ₁={result.final_lyapunov[0]:.4f}, λ₂={result.final_lyapunov[1]:.4f} {status_icon} {conv}"
            if result.kaplan_yorke_dim is not None:
                strange = "★" if result.is_strange else "○"
                output += f" KY={result.kaplan_yorke_dim:.3f}{strange}"
            print(output)
            
        # transition summary
        summary = results.get_summary()
        transitions = summary['critical_transitions']
        
        print(f"\n=== critical thresholds ===")
        if transitions['chaos_to_sync']:
            print(f"🎯 fiber collapse: εc ≈ {transitions['chaos_to_sync']:.4f}")
        if transitions['chaos_to_hyperchaos']:
            print(f"🌪️  hyperchaos onset: εh ≈ {transitions['chaos_to_hyperchaos']:.4f}")
        if transitions['strange_to_regular']:
            print(f"🔸 strangeness loss: εs ≈ {transitions['strange_to_regular']:.4f}")
            
        # recommendations
        print(f"\n=== recommendations ===")
        if not any(transitions.values()):
            print("• no transitions found - try wider noise range or different attractor")
        else:
            if transitions['chaos_to_sync']:
                εc = transitions['chaos_to_sync']
                print(f"• explore on-off intermittency near εc = {εc:.4f}")
                print(f"• study scaling laws with noise range [{εc*0.8:.4f}, {εc*1.2:.4f}]")
            if transitions['chaos_to_hyperchaos']:
                εh = transitions['chaos_to_hyperchaos']
                print(f"• investigate explosion dynamics near εh = {εh:.4f}")
            if transitions['strange_to_regular']:
                εs = transitions['strange_to_regular']
                print(f"• analyze fractal dimension collapse near εs = {εs:.4f}")
                
        print(f"\nconvergence: {summary['converged_points']}/{summary['total_noise_points']} ({summary['convergence_rate']:.1%})")
        print(f"total time: {summary['total_compute_time']:.1f}s")
        
    except Exception as e:
        print(f"\nerror during sweep: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    try:
        interactive_cli()
    except KeyboardInterrupt:
        print(f"\n\ncancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nunexpected error: {e}")
        sys.exit(1) 