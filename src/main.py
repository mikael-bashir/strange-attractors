"""
CUDA-accelerated strange attractor simulation with animation and analysis capabilities.

This script can run two types of simulations:
1. Simple simulation: evolve particles for N steps, optionally save Lyapunov analysis
2. Animation simulation: three-phase evolution to create morphing attractor GIFs

The animation simulation uses a sophisticated three-phase approach:
- Pullback phase: settle particles onto the random attractor (non-uniform sampling)  
- Intermediate phase: transition period with uniform snapshots
- Main phase: high-frequency snapshots for smooth animation
"""

import os
import argparse
import time
import numpy as np
import cupy as cp

from .noise.types import NoiseConfig, NoiseType, generate_noise, ATTRACTOR_NOISE_CONFIGS
from .attractors import AVAILABLE_ATTRACTORS
from .compute.cuda_backend import CUDABackend
from .analysis.lyapunov import LyapunovTracker
from .viz.animation import create_morphing_animation
from .viz.plots import plot_lyapunov_analysis, save_lyapunov_data

# Animation configuration - three-phase simulation for smooth morphing
# Phase 1: Pullback to settle on random attractor (non-uniform sampling)
PULLBACK_STEPS = 4000  # total steps to reach steady state on random attractor
PULLBACK_SNAPSHOT_STEPS = [1, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2800, 3600, 4000]  # when to capture frames

# Phase 2: Intermediate transition (uniform sampling)  
INTERMEDIATE_STEPS = 1000  # smooth transition between pullback and main phases
INTERMEDIATE_SNAPSHOTS = 5  # number of frames during transition

# Phase 3: Main animation (high-frequency sampling)
STEPS_PER_SNAPSHOT = 30  # time steps between each frame (controls animation speed)
NUM_SNAPSHOTS = 120  # total frames for main animation sequence
TOTAL_STEPS = PULLBACK_STEPS + INTERMEDIATE_STEPS + (NUM_SNAPSHOTS * STEPS_PER_SNAPSHOT)

# Lyapunov exponent computation settings
LYAP_COMPUTATION = True  # enable lyapunov tracking (only for supported attractors)
LYAP_RENORM_INTERVAL = 10  # renormalize tangent vectors every N steps  
LYAP_TRACK_PARTICLES = 100  # subset of particles used for lyapunov computation
LYAP_SAVE_INTERVAL = 100  # save lyapunov values every N steps

# Output visualization settings
WIDTH = 600  # histogram resolution (affects GIF quality)
HEIGHT = 600
FPS = 15  # for output GIF

def main():
    parser = argparse.ArgumentParser(description='CUDA-accelerated strange attractor simulation')
    
    # Attractor selection
    parser.add_argument('--attractor', type=str, default='henon',
                       choices=list(AVAILABLE_ATTRACTORS.keys()),
                       help='Type of attractor to simulate')
    
    # Simulation parameters
    parser.add_argument('--num-particles', type=int, default=0,
                       help='Number of particles to simulate (default = num_blocks * threads_per_block)')
    parser.add_argument('--num-steps', type=int, default=TOTAL_STEPS,
                       help='Number of evolution steps')
    parser.add_argument('--lyap-particles', type=int, default=LYAP_TRACK_PARTICLES,
                       help='Number of particles for Lyapunov tracking')
    
    # CUDA configuration
    parser.add_argument('--num-blocks', type=int, default=2048,
                       help='Number of CUDA blocks')
    parser.add_argument('--threads-per-block', type=int, default=256,
                       help='Threads per CUDA block')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base directory for output files (results will be saved in a timestamped subfolder)')
    parser.add_argument('--save-animation', action='store_true',
                       help='Save animation of evolution')
    parser.add_argument('--save-lyapunov', action='store_true',
                       help='Save Lyapunov analysis')
    
    # Noise configuration
    parser.add_argument('--custom-noise', action='store_true',
                       help='Use custom noise configuration instead of default')
    
    args = parser.parse_args()
    
    # determine particle count if not specified
    if args.num_particles <= 0:
        args.num_particles = args.num_blocks * args.threads_per_block
    
    # setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{timestamp}_{args.attractor}_attractor")
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir
    
    # Get attractor and backend
    attractor = AVAILABLE_ATTRACTORS[args.attractor]()
    backend = CUDABackend(args.num_blocks, args.threads_per_block)
    
    # Get noise configuration
    if args.custom_noise:
        noise_config = NoiseConfig.from_cli()
    else:
        noise_config = ATTRACTOR_NOISE_CONFIGS[args.attractor]
        print(f"\nUsing default noise configuration for {args.attractor}:")
        if noise_config.type == NoiseType.UNIFORM:
            print(f"Uniform noise: [{noise_config.params['low']}, {noise_config.params['high']}]")
        elif noise_config.type == NoiseType.GAUSSIAN:
            print(f"Gaussian noise: μ={noise_config.params['mean']}, σ={noise_config.params['std']}")
        else:
            print("No noise")
    
    # Initialize particle ensemble on GPU
    particles_x = cp.random.uniform(
        attractor.init_bounds['x'][0],
        attractor.init_bounds['x'][1],
        args.num_particles,
        dtype=cp.float32
    )
    particles_y = cp.random.uniform(
        attractor.init_bounds['y'][0],
        attractor.init_bounds['y'][1],
        args.num_particles,
        dtype=cp.float32
    )
    
    # Initialize Lyapunov tracking if requested
    lyap_tracker = None
    if args.lyap_particles > 0 and attractor.name in ["Hénon", "Clifford"]:
        lyap_tracker = LyapunovTracker(
            num_particles=min(args.lyap_particles, args.num_particles)
        )
        print(f"Lyapunov tracking enabled for {lyap_tracker.num_particles} particles")
    elif args.lyap_particles > 0:
        print(f"Warning: Lyapunov tracking only supported for Hénon and Clifford attractors")
    
    # Generate noise sequences on GPU (adjust length for animation)
    if args.save_animation:
        # animation phases based on original script
        PULLBACK_SNAPSHOT_STEPS = [1, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2800, 3600, 4000]
        PULLBACK_STEPS = 4000
        INTERMEDIATE_STEPS = 1000
        INTERMEDIATE_SNAPSHOTS = 5
        STEPS_PER_SNAPSHOT = 30
        NUM_SNAPSHOTS = 120
        total_steps = PULLBACK_STEPS + INTERMEDIATE_STEPS + STEPS_PER_SNAPSHOT * NUM_SNAPSHOTS
    else:
        total_steps = args.num_steps
    noise_params = {}
    for param_name in attractor.param_names:
        noise_params[param_name] = generate_noise(total_steps, noise_config)
    
    # Run simulation (single-run or segmented for animation)
    print(f"\nSimulating {attractor.name} attractor with {args.num_particles:,} particles...")
    print(f"Noise type: {noise_config.type.value}")
    start_time = time.time()
    
    if not args.save_animation:
        print(f"Using CUDA backend for {args.num_steps} steps")
        backend.evolve_ensemble(attractor, particles_x, particles_y, noise_params, args.num_steps, lyap_tracker)
        simulation_time = time.time() - start_time
        print(f"Simulation completed in {simulation_time:.2f} seconds")
        print(f"Performance: {args.num_particles * args.num_steps / simulation_time / 1e6:.2f} million iterations/sec")
    else:
        print(f"Running segmented evolution for animation ({TOTAL_STEPS} steps)")
        snapshot_images = []
        prev = 0
        
        # --- Pullback Phase with Non-uniform Snapshots ---
        print("Performing pullback to settle on random attractor...")
        for target in PULLBACK_SNAPSHOT_STEPS:
            seg = target - prev
            seg_noise = {p: noise_params[p][prev:target] for p in noise_params}
            backend.evolve_ensemble(attractor, particles_x, particles_y, seg_noise, seg, lyap_tracker)
            prev = target
            
            # Capture snapshot
            xs = cp.asnumpy(particles_x)
            ys = cp.asnumpy(particles_y)
            hist, _, _ = np.histogram2d(xs, ys, bins=(WIDTH, HEIGHT),
                                      range=[list(attractor.bounds['x']), 
                                            list(attractor.bounds['y'])])
            snapshot_images.append(hist.T)
            
            # Log progress
            if lyap_tracker is not None and target % LYAP_SAVE_INTERVAL == 0:
                current_lyap = lyap_tracker.save_snapshot(noise_config.get_range())
                print(f"Pullback snapshot {len(snapshot_images)}/{len(PULLBACK_SNAPSHOT_STEPS)} at step {target} - Lyapunov exp: {current_lyap}")
            else:
                print(f"Pullback snapshot {len(snapshot_images)}/{len(PULLBACK_SNAPSHOT_STEPS)} at step {target}")
        
        print("Pullback complete.")
        
        # --- Intermediate Phase ---
        print("Intermediate evolution phase...")
        interval = INTERMEDIATE_STEPS // INTERMEDIATE_SNAPSHOTS
        
        for idx in range(INTERMEDIATE_SNAPSHOTS):
            seg_noise = {p: noise_params[p][prev:prev+interval] for p in noise_params}
            backend.evolve_ensemble(attractor, particles_x, particles_y, seg_noise, interval, lyap_tracker)
            prev += interval
            
            # Capture snapshot
            xs = cp.asnumpy(particles_x)
            ys = cp.asnumpy(particles_y)
            hist, _, _ = np.histogram2d(xs, ys, bins=(WIDTH, HEIGHT),
                                      range=[list(attractor.bounds['x']), 
                                            list(attractor.bounds['y'])])
            snapshot_images.append(hist.T)
            
            # Log progress
            if lyap_tracker is not None:
                current_lyap = lyap_tracker.save_snapshot(noise_config.get_range())
                print(f"Intermediate snapshot {idx+1}/{INTERMEDIATE_SNAPSHOTS} - Lyapunov exp: {current_lyap}")
            else:
                print(f"Intermediate snapshot {idx+1}/{INTERMEDIATE_SNAPSHOTS}")
        
        print("Intermediate phase complete.")
        
        # --- Main Snapshot Generation Phase ---
        for snap_idx in range(NUM_SNAPSHOTS):
            print(f"Generating main snapshot {snap_idx+1}/{NUM_SNAPSHOTS}")
            
            seg_noise = {p: noise_params[p][prev:prev+STEPS_PER_SNAPSHOT] for p in noise_params}
            backend.evolve_ensemble(attractor, particles_x, particles_y, seg_noise, STEPS_PER_SNAPSHOT, lyap_tracker)
            prev += STEPS_PER_SNAPSHOT
            
            # Capture snapshot
            xs = cp.asnumpy(particles_x)
            ys = cp.asnumpy(particles_y)
            hist, _, _ = np.histogram2d(xs, ys, bins=(WIDTH, HEIGHT),
                                      range=[list(attractor.bounds['x']), 
                                            list(attractor.bounds['y'])])
            snapshot_images.append(hist.T)
            
            # Log progress and save lyapunov data periodically
            if lyap_tracker is not None and snap_idx % (LYAP_SAVE_INTERVAL // STEPS_PER_SNAPSHOT) == 0:
                current_lyap = lyap_tracker.save_snapshot(noise_config.get_range())
                print(f"Main snapshot {snap_idx+1}/{NUM_SNAPSHOTS} - Lyapunov exp: {current_lyap}")
        
        print(f"Collected {len(snapshot_images)} frames for animation")
        simulation_time = time.time() - start_time
        print(f"Animation simulation completed in {simulation_time:.2f} seconds")
        
        # Create animation
        filename = create_morphing_animation(
            snapshot_images=snapshot_images,
            attractor_name=attractor.name,
            bounds=attractor.bounds,
            output_dir=args.output_dir,
            fps=FPS,
            pullback_steps=PULLBACK_SNAPSHOT_STEPS,
            intermediate_interval=interval
        )
        print(f"Animation saved to: {filename}")
    
    # --- Save Lyapunov Analysis ---
    if args.save_lyapunov and lyap_tracker is not None:
        print("\nSaving Lyapunov analysis...")
        
        # Get final lyapunov spectrum
        mean_spectrum, max_spectrum = lyap_tracker.get_current_exponents()
        print(f"Mean Lyapunov spectrum: λ₁={mean_spectrum[0]:.4f}, λ₂={mean_spectrum[1]:.4f}")
        print(f"Max Lyapunov spectrum:  λ₁={max_spectrum[0]:.4f}, λ₂={max_spectrum[1]:.4f}")
        
        # Plot analysis
        plot_file = plot_lyapunov_analysis(
            history=lyap_tracker.history,
            attractor_name=attractor.name,
            noise_config=noise_config.__dict__,
            output_dir=args.output_dir
        )
        print(f"Lyapunov analysis plot saved to: {plot_file}")
        
        # Save data
        data_file = save_lyapunov_data(
            history=lyap_tracker.history,
            attractor_name=attractor.name,
            noise_config=noise_config.__dict__,
            output_dir=args.output_dir
        )
        print(f"Lyapunov analysis data saved to: {data_file}")
    
    # --- Display Final Statistics ---
    particles_x_cpu = cp.asnumpy(particles_x)
    particles_y_cpu = cp.asnumpy(particles_y)
    
    # Filter out any infinite/nan particles
    valid_mask = np.isfinite(particles_x_cpu) & np.isfinite(particles_y_cpu)
    valid_particles = np.sum(valid_mask)
    
    print(f"\nFinal Statistics:")
    print(f"Valid particles: {valid_particles:,} / {args.num_particles:,} ({100*valid_particles/args.num_particles:.1f}%)")
    
    if valid_particles > 0:
        x_range = np.ptp(particles_x_cpu[valid_mask])
        y_range = np.ptp(particles_y_cpu[valid_mask])
        print(f"X range: [{np.min(particles_x_cpu[valid_mask]):.3f}, {np.max(particles_x_cpu[valid_mask]):.3f}] (span: {x_range:.3f})")
        print(f"Y range: [{np.min(particles_y_cpu[valid_mask]):.3f}, {np.max(particles_y_cpu[valid_mask]):.3f}] (span: {y_range:.3f})")

if __name__ == '__main__':
    main() 