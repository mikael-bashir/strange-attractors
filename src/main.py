#!/usr/bin/env python3
"""
comprehensive regime analysis CLI

implements the complete fused analysis pipeline from theory.md:
- confidence interval analysis + deterministic intervals + regime classification
- kaplan-yorke dimension tracking  
- critical noise detection and regime boundary mapping
- single command interface for all regime analysis needs
- visualization of attractor evolution and analysis results

usage:
  python main.py regime -a henon -n 0.001  # single point analysis
  python main.py sweep -a henon -r 0.0001 0.01  # critical transitions
  python main.py find-critical -a henon -r 0.001 0.1  # precise boundary
  python main.py visualize -a henon -n 0.001  # create visualizations
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

from analysis.regime_analyzer import RegimeAnalyzer, RegimeConfig
from analysis.noise_sweep import (
    NoiseSweepRunner, NoiseSweepConfig, quick_noise_sweep, 
    find_chaos_sync_boundary
)
from attractors import AVAILABLE_ATTRACTORS
from noise.types import ATTRACTOR_NOISE_CONFIGS
from viz import create_morphing_animation, plot_lyapunov_analysis, save_lyapunov_data


def create_output_dir(base_name: str) -> str:
    """create timestamped output directory"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_attractor_params(attractor_name: str, custom_params: Optional[Dict] = None) -> Dict[str, float]:
    """get default or custom attractor parameters"""
    defaults = {
        'henon': {'a': 1.4, 'b': 0.3},
        'clifford': {'a': -1.4, 'b': 1.6, 'c': 1.0, 'd': 0.7}
    }
    
    base_params = defaults.get(attractor_name, {})
    if custom_params:
        base_params.update(custom_params)
    
    return base_params


def cmd_regime_analysis(args) -> None:
    """single-point comprehensive regime analysis"""
    print("═" * 60)
    print("COMPREHENSIVE REGIME ANALYSIS")
    print("═" * 60)
    
    # setup
    attractor_params = get_attractor_params(args.attractor, args.params)
    noise_amplitude = args.noise
    
    print(f"attractor: {args.attractor}")
    print(f"parameters: {attractor_params}")
    print(f"noise amplitude: {noise_amplitude}")
    print(f"trajectories: {args.num_trajectories}")
    print()
    
    # configuration
    print("setting up analysis configuration...")
    regime_config = RegimeConfig(
        min_trajectories=args.num_trajectories,
        confidence_level=args.confidence_level,
        enable_ky_tracking=not args.no_kaplan_yorke,
        attractor_bound=args.attractor_bound
    )
    
    # noise sweep with single point (reuse infrastructure)
    sweep_config = NoiseSweepConfig(
        attractor_name=args.attractor,
        base_params=attractor_params,
        noise_amplitudes=[noise_amplitude],
        num_trajectories=args.num_trajectories,
        particles=args.particles,
        lyap_particles=args.lyap_particles,
        regime_config=regime_config,
        save_detailed_data=args.save_data,
        use_deterministic_heuristic=not args.no_deterministic_heuristic
    )
    
    print("initializing CUDA backend and attractor...")
    runner = NoiseSweepRunner(sweep_config)
    
    print("running multi-trajectory simulation...")
    print("(this may take a few minutes depending on settings)")
    results = runner.run()
    
    if not results.results:
        print("❌ analysis failed - no results")
        return
    
    result = results.results[0]
    classification = result.regime_classification
    
    # comprehensive output
    print("\n" + "═" * 60)
    print("REGIME CLASSIFICATION")
    print("═" * 60)
    
    print(f"regime type: {classification.regime_type}")
    print(f"confidence: {classification.confidence_level*100:.1f}% via {classification.confidence_source}")
    
    bounds = classification.lambda1_bounds
    print(f"λ₁ bounds: [{bounds[0]:.6f}, {bounds[1]:.6f}]")
    print(f"interval width: {bounds[1] - bounds[0]:.6f}")
    
    if result.kaplan_yorke_dim is not None:
        print(f"kaplan-yorke dimension: {result.kaplan_yorke_dim:.4f}")
        is_strange = result.has_strange_attractor
        print(f"strange attractor: {'yes' if is_strange else 'no'}")
    
    print(f"\ninterval scaling comparison:")
    print(f"deterministic interval width: {result.deterministic_bounds_width:.6f}")
    print(f"confidence interval width: {result.confidence_interval_width:.6f}")
    print(f"confidence intervals {result.interval_scaling_ratio:.0f}x tighter")
    
    print(f"\nanalysis quality:")
    print(f"confidence analysis r²: {result.confidence_analysis_quality:.3f}")
    print(f"computation time: {result.compute_time:.1f}s")
    
    # save results
    if args.output_dir:
        print(f"\nsaving results...")
        output_dir = create_output_dir(args.output_dir)
        results_file = os.path.join(output_dir, "regime_analysis.json")
        
        analysis_data = {
            'attractor': args.attractor,
            'parameters': attractor_params,
            'noise_amplitude': noise_amplitude,
            'result': {
                'regime_type': classification.regime_type,
                'confidence_level': classification.confidence_level,
                'confidence_source': classification.confidence_source,
                'lambda1_bounds': bounds,
                'lambda2_bounds': classification.lambda2_bounds,
                'ky_dimension': classification.ky_dimension,
                'kaplan_yorke_dim': result.kaplan_yorke_dim,
                'interval_scaling_ratio': result.interval_scaling_ratio,
                'analysis_quality': result.confidence_analysis_quality,
                'has_strange_attractor': result.has_strange_attractor
            },
            'config': {
                'num_trajectories': args.num_trajectories,
                'particles': args.particles,
                'confidence_level': args.confidence_level
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"results saved to: {results_file}")


def cmd_noise_sweep(args) -> None:
    """comprehensive noise sweep for critical transition detection"""
    print("═" * 60)
    print("NOISE SWEEP ANALYSIS")
    print("═" * 60)
    
    # setup
    attractor_params = get_attractor_params(args.attractor, args.params)
    
    # noise amplitude range
    if args.noise_range and len(args.noise_range) == 2:
        noise_min, noise_max = args.noise_range
        noise_amplitudes = list(np.linspace(noise_min, noise_max, args.num_points))
    else:
        print("❌ must specify --noise-range with two values")
        return
    
    print(f"attractor: {args.attractor}")
    print(f"parameters: {attractor_params}")
    print(f"noise range: [{noise_min:.6f}, {noise_max:.6f}]")
    print(f"sweep points: {args.num_points}")
    print(f"trajectories per point: {args.num_trajectories}")
    print()
    
    # configuration
    regime_config = RegimeConfig(
        min_trajectories=args.num_trajectories,
        confidence_level=args.confidence_level,
        enable_ky_tracking=not args.no_kaplan_yorke,
        attractor_bound=args.attractor_bound
    )
    
    sweep_config = NoiseSweepConfig(
        attractor_name=args.attractor,
        base_params=attractor_params,
        noise_amplitudes=noise_amplitudes,
        num_trajectories=args.num_trajectories,
        particles=args.particles,
        lyap_particles=args.lyap_particles,
        regime_config=regime_config,
        save_detailed_data=args.save_data,
        use_deterministic_heuristic=not args.no_deterministic_heuristic
    )
    
    runner = NoiseSweepRunner(sweep_config)
    results = runner.run()
    
    # analyze results
    transitions = results.find_critical_transitions()
    phase_diagram = results.get_regime_phase_diagram()
    
    print("\n" + "═" * 60)
    print("CRITICAL TRANSITIONS")
    print("═" * 60)
    
    for transition_type, critical_noise in transitions.items():
        if critical_noise is not None:
            print(f"{transition_type}: α* ≈ {critical_noise:.6f}")
        else:
            print(f"{transition_type}: not detected")
    
    print("\n" + "═" * 60)
    print("REGIME PHASE DIAGRAM")
    print("═" * 60)
    
    for regime, noise_points in phase_diagram.items():
        if noise_points:
            print(f"{regime}: {len(noise_points)} points")
            if len(noise_points) <= 5:
                points_str = ", ".join(f"{p:.4f}" for p in sorted(noise_points))
                print(f"  α = [{points_str}]")
            else:
                points_sorted = sorted(noise_points)
                print(f"  α ∈ [{points_sorted[0]:.4f}, {points_sorted[-1]:.4f}]")
    
    # bounds scaling analysis
    bounds_analysis = results.get_bounds_scaling_analysis()
    if bounds_analysis:
        print(f"\naverage scaling advantage: {bounds_analysis['avg_scaling_advantage']:.0f}x")
        ratio_range = bounds_analysis['scaling_advantage_range']
        print(f"scaling ratio range: [{ratio_range[0]:.0f}x, {ratio_range[1]:.0f}x]")
    
    # save results
    if args.output_dir:
        output_dir = create_output_dir(args.output_dir)
        results.save(os.path.join(output_dir, "noise_sweep_results.json"))
        print(f"\nfull results saved to: {output_dir}")


def cmd_find_critical(args) -> None:
    """precise critical noise boundary detection using binary search"""
    print("═" * 60)
    print("CRITICAL NOISE BOUNDARY DETECTION")
    print("═" * 60)
    
    attractor_params = get_attractor_params(args.attractor, args.params)
    
    if not args.range or len(args.range) != 2:
        print("❌ must specify --range with two values")
        return
    
    noise_range = tuple(args.range)
    
    print(f"attractor: {args.attractor}")
    print(f"parameters: {attractor_params}")
    print(f"search range: [{noise_range[0]:.6f}, {noise_range[1]:.6f}]")
    print(f"tolerance: {args.tolerance:.2e}")
    print(f"max iterations: {args.max_iterations}")
    print()
    
    print("binary search for chaos↔sync boundary...")
    
    critical_noise = find_chaos_sync_boundary(
        args.attractor, 
        attractor_params,
        noise_range,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations
    )
    
    print("\n" + "═" * 60)
    print("CRITICAL BOUNDARY")
    print("═" * 60)
    
    if critical_noise is not None:
        print(f"critical noise amplitude: α* ≈ {critical_noise:.8f}")
        print(f"chaos↔sync transition at α* ± {args.tolerance:.2e}")
        
        # verify with single analysis at critical point
        if args.verify:
            print(f"\nverification analysis at α* = {critical_noise:.8f}...")
            
            # quick verification
            result = quick_noise_sweep(
                args.attractor, attractor_params, 
                (critical_noise, critical_noise), 1, 30
            )
            
            if result.results:
                regime = result.results[0].regime_classification
                print(f"verification: {regime.regime_type} ({regime.confidence_level*100:.0f}% confidence)")
                
                bounds = regime.lambda1_bounds
                print(f"λ₁ ∈ [{bounds[0]:.6f}, {bounds[1]:.6f}]")
                
                # how close to λ₁ = 0?
                center = (bounds[0] + bounds[1]) / 2
                print(f"λ₁ center: {center:.6f} (distance from 0: {abs(center):.6f})")
    else:
        print("❌ critical boundary not found in specified range")
        print("try expanding the search range or adjusting tolerance")
    
    # save result
    if args.output_dir and critical_noise is not None:
        output_dir = create_output_dir(args.output_dir)
        
        critical_data = {
            'attractor': args.attractor,
            'parameters': attractor_params,
            'search_range': noise_range,
            'critical_noise': critical_noise,
            'tolerance': args.tolerance,
            'max_iterations': args.max_iterations
        }
        
        results_file = os.path.join(output_dir, "critical_noise.json")
        with open(results_file, 'w') as f:
            json.dump(critical_data, f, indent=2)
        
        print(f"\nresults saved to: {results_file}")


def cmd_visualize(args) -> None:
    """create visualizations of attractor evolution and analysis"""
    print("═" * 60)
    print("ATTRACTOR VISUALIZATION")
    print("═" * 60)
    
    # setup
    attractor_params = get_attractor_params(args.attractor, args.params)
    noise_amplitude = args.noise
    
    print(f"attractor: {args.attractor}")
    print(f"parameters: {attractor_params}")
    print(f"noise amplitude: {noise_amplitude}")
    print(f"particles: {args.particles}")
    print()
    
    # configuration
    regime_config = RegimeConfig(
        min_trajectories=1,  # just need one for visualization
        confidence_level=0.95,
        enable_ky_tracking=not args.no_kaplan_yorke,
        attractor_bound=args.attractor_bound
    )
    
    sweep_config = NoiseSweepConfig(
        attractor_name=args.attractor,
        base_params=attractor_params,
        noise_amplitudes=[noise_amplitude],
        num_trajectories=1,
        particles=args.particles,
        lyap_particles=args.lyap_particles,
        regime_config=regime_config,
        save_detailed_data=True,  # need trajectory data for visualization
        use_deterministic_heuristic=not args.no_deterministic_heuristic
    )
    
    print("initializing CUDA backend and attractor...")
    runner = NoiseSweepRunner(sweep_config)
    
    print("running simulation for visualization...")
    results = runner.run()
    
    if not results.results:
        print("❌ visualization failed - no results")
        return
    
    result = results.results[0]
    
    # create output directory
    if args.output_dir:
        output_dir = create_output_dir(args.output_dir)
    else:
        output_dir = create_output_dir("visualization_results")
    
    # generate visualizations based on flags
    if not args.analysis_only:
        print("\ncreating morphing attractor animation...")
        gif_file = create_morphing_animation(
            result.snapshot_images,
            args.attractor,
            result.bounds,
            output_dir,
            fps=args.fps,
            pullback_steps=result.pullback_steps,
            intermediate_interval=result.intermediate_interval
        )
        print(f"saved animation to: {gif_file}")
    
    if not args.animation_only and result.trajectory_data and result.trajectory_data[0]:
        print("\ncreating lyapunov analysis plots...")
        plot_file = plot_lyapunov_analysis(
            result.trajectory_data[0],  # use first trajectory's data
            args.attractor,
            {'type': 'uniform', 'params': {'low': -noise_amplitude, 'high': noise_amplitude}},
            output_dir
        )
        print(f"saved analysis plots to: {plot_file}")
        
        if args.save_data:
            data_file = save_lyapunov_data(
                result.trajectory_data[0],  # use first trajectory's data
                args.attractor,
                {'type': 'uniform', 'params': {'low': -noise_amplitude, 'high': noise_amplitude}},
                output_dir
            )
            print(f"saved analysis data to: {data_file}")


def main():
    parser = argparse.ArgumentParser(description='regime analysis CLI')
    subparsers = parser.add_subparsers(dest='command', help='analysis commands')
    
    # shared arguments
    def add_common_args(parser):
        parser.add_argument('-a', '--attractor', type=str, default='henon',
                           choices=list(AVAILABLE_ATTRACTORS.keys()),
                           help='attractor type')
        parser.add_argument('-p', '--params', type=json.loads, default=None,
                           help='custom attractor parameters as JSON')
        parser.add_argument('-t', '--num-trajectories', type=int, default=10,
                           help='number of independent trajectories')
        parser.add_argument('-P', '--particles', type=int, default=10000,
                           help='particles per trajectory')
        parser.add_argument('-L', '--lyap-particles', type=int, default=1000,
                           help='particles for lyapunov tracking')
        parser.add_argument('-c', '--confidence-level', type=float, default=0.95,
                           help='confidence level for statistical bounds')
        parser.add_argument('-b', '--attractor-bound', type=float, default=2.5,
                           help='bound |x|,|y| ≤ R for deterministic analysis')
        parser.add_argument('--no-kaplan-yorke', action='store_true',
                           help='disable kaplan-yorke dimension tracking')
        parser.add_argument('--no-deterministic-heuristic', action='store_true',
                           help='disable deterministic heuristic optimization')
        parser.add_argument('-o', '--output-dir', type=str, default='regime_analysis_results',
                           help='output directory base name')
        parser.add_argument('-s', '--save-data', action='store_true',
                           help='save detailed trajectory data')
    
    # regime command - single point analysis
    regime_parser = subparsers.add_parser('regime', help='single-point regime analysis')
    add_common_args(regime_parser)
    regime_parser.add_argument('-n', '--noise', type=float, required=True,
                              help='noise amplitude for analysis')
    
    # sweep command - noise amplitude sweep  
    sweep_parser = subparsers.add_parser('sweep', help='noise sweep analysis')
    add_common_args(sweep_parser)
    sweep_parser.add_argument('-r', '--noise-range', type=float, nargs=2, required=True,
                             help='noise amplitude range (min max)')
    sweep_parser.add_argument('-N', '--num-points', type=int, default=20,
                             help='number of sweep points')
    
    # find-critical command - precise boundary detection
    critical_parser = subparsers.add_parser('find-critical', help='find critical noise boundary')
    add_common_args(critical_parser)
    critical_parser.add_argument('-r', '--range', type=float, nargs=2, required=True,
                                help='search range for critical noise (min max)')
    critical_parser.add_argument('-e', '--tolerance', type=float, default=1e-5,
                                help='convergence tolerance')
    critical_parser.add_argument('-i', '--max-iterations', type=int, default=15,
                                help='maximum binary search iterations')
    critical_parser.add_argument('-v', '--verify', action='store_true',
                                help='verify critical point with single analysis')
    
    # visualize command - create animations and plots
    viz_parser = subparsers.add_parser('visualize', help='create attractor visualizations')
    add_common_args(viz_parser)
    viz_parser.add_argument('-n', '--noise', type=float, required=True,
                           help='noise amplitude for visualization')
    viz_parser.add_argument('-f', '--fps', type=int, default=15,
                           help='frames per second for animation')
    viz_parser.add_argument('--animation-only', action='store_true',
                           help='only generate morphing animation')
    viz_parser.add_argument('--analysis-only', action='store_true',
                           help='only generate analysis plots')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # import numpy here to avoid startup overhead
    import numpy as np
    globals()['np'] = np
    
    # dispatch commands
    if args.command == 'regime':
        cmd_regime_analysis(args)
    elif args.command == 'sweep':
        cmd_noise_sweep(args)
    elif args.command == 'find-critical':
        cmd_find_critical(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    else:
        print(f"❌ unknown command: {args.command}")


if __name__ == '__main__':
    main() 
