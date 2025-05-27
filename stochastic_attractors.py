# 1. Generate long global_noise_a, global_noise_b
# 2. Initialize particles_x_gpu, particles_y_gpu (ensemble, spread out)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import time
import math
import platform

# Set seed for reproducibility
np.random.seed(42)

# Platform detection and imports
IS_WINDOWS = platform.system() == 'Windows'
IS_MAC = platform.system() == 'Darwin'

if IS_WINDOWS:
    try:
        import cupy as cp
        from numba import cuda
        print("Windows detected - using CUDA backend")
        
        # Check for CUDA GPU
        if not cuda.is_available():
            print("ERROR: No CUDA GPU detected on Windows!")
            print("This script requires an NVIDIA GPU with CUDA support.")
            exit(1)
        
        print(f"CUDA GPU detected: {cuda.get_current_device().name.decode()}")
        BACKEND = 'cuda'
        
    except ImportError as e:
        print(f"ERROR: Failed to import CUDA libraries: {e}")
        print("Please install: pip install cupy numba")
        exit(1)

elif IS_MAC:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap
        print("Mac detected - using JAX Metal backend")
        
        # Check for Metal support
        if not any('metal' in str(device).lower() for device in jax.devices()):
            print("WARNING: Metal backend not detected, falling back to CPU")
            print("For best performance, install: pip install jax[metal]")
        else:
            print(f"Metal backend detected with {len(jax.devices())} device(s)")
        
        BACKEND = 'jax'
        
    except ImportError as e:
        print(f"ERROR: Failed to import JAX: {e}")
        print("Please install: pip install jax[metal]")
        exit(1)

else:
    print(f"ERROR: Unsupported platform: {platform.system()}")
    print("This script only supports Windows (CUDA) and macOS (JAX Metal)")
    exit(1)

# --- Attractor Configurations ---
ATTRACTORS = {
    1: {
        'name': 'Hénon',
        'params': {'a': 1.4, 'b': 0.3},
        'noise_range': 0.005,
        'bounds': {'x': (-2.0, 2.0), 'y': (-0.6, 0.6)},
        'init_bounds': {'x': (-1.0, 1.0), 'y': (-0.5, 0.5)}
    },
    2: {
        'name': 'Clifford',
        'params': {'a': -1.7, 'b': 1.8, 'c': -0.9, 'd': -0.4},
        'noise_range': 0.15,
        'bounds': {'x': (-2.5, 2.5), 'y': (-2.5, 2.5)},
        'init_bounds': {'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}
    },
    3: {
        'name': 'Ikeda',
        'params': {'u': 0.90},   # More stable base u
        'noise_range': 0.01,    # Smaller noise range initially
        'bounds': {'x': (0.0, 2.0), 'y': (-2.0, 2.0)}, # Start with a smaller view window
        'init_bounds': {'x': (0.1, 0.5), 'y': (-0.5, 0.5)}
    }
}

def choose_attractor():
    print("\nChoose an attractor:")
    for key, config in ATTRACTORS.items():
        print(f"{key}. {config['name']} Attractor")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if choice in ATTRACTORS:
                return choice
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")

# Get user choice
attractor_choice = choose_attractor()
config = ATTRACTORS[attractor_choice]

print(f"\nSelected: {config['name']} Attractor")

# --- Parameters ---
ATTRACTOR_NAME = config['name']
BASE_PARAMS = config['params']
NOISE_RANGE = config['noise_range']
X_MIN, X_MAX = config['bounds']['x']
Y_MIN, Y_MAX = config['bounds']['y']
INIT_X_MIN, INIT_X_MAX = config['init_bounds']['x']
INIT_Y_MIN, INIT_Y_MAX = config['init_bounds']['y']

# Simulation parameters
WIDTH = 600
HEIGHT = 600  # Square for better viewing
NUM_THREADS_PER_BLOCK = 256
NUM_BLOCKS = 2048  # 524k particles total

# Pullback and snapshot parameters
PULLBACK_STEPS = 4000
STEPS_PER_SNAPSHOT = 30
NUM_SNAPSHOTS = 120
FPS = 15

# Generate noise for all parameters
param_names = list(BASE_PARAMS.keys())
TOTAL_NOISE_LENGTH = PULLBACK_STEPS + (NUM_SNAPSHOTS * STEPS_PER_SNAPSHOT) + 1000

print(f"Generating {TOTAL_NOISE_LENGTH} noise values for {len(param_names)} parameters...")
noise_dict = {}
for param in param_names:
    noise_dict[param] = np.random.uniform(-NOISE_RANGE, NOISE_RANGE, 
                                         size=TOTAL_NOISE_LENGTH).astype(np.float32)

# Output
RESULTS_DIR = "results_morphing_attractors"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
GIF_FILENAME = os.path.join(RESULTS_DIR, f"morphing_{ATTRACTOR_NAME.lower()}_{TIMESTAMP}.gif")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Platform-specific kernels ---
if BACKEND == 'cuda':
    # CUDA kernels for Windows
    @cuda.jit
    def evolve_henon_kernel(particles_x, particles_y, noise_a, noise_b, 
                           num_steps, base_a, base_b):
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return

        x, y = particles_x[thread_id], particles_y[thread_id]
        
        for step in range(num_steps):
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            x_new = 1.0 - a * x * x + y
            y_new = b * x
            x, y = x_new, y_new

        particles_x[thread_id] = x
        particles_y[thread_id] = y

    @cuda.jit  
    def evolve_clifford_kernel(particles_x, particles_y, noise_a, noise_b, noise_c, noise_d,
                              num_steps, base_a, base_b, base_c, base_d):
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return

        x, y = particles_x[thread_id], particles_y[thread_id]
        
        for step in range(num_steps):
            a = base_a + noise_a[step]
            b = base_b + noise_b[step] 
            c = base_c + noise_c[step]
            d = base_d + noise_d[step]
            
            x_new = math.sin(a * y) + c * math.cos(a * x)
            y_new = math.sin(b * x) + d * math.cos(b * y)
            x, y = x_new, y_new

        particles_x[thread_id] = x
        particles_y[thread_id] = y

    @cuda.jit
    def evolve_ikeda_kernel(particles_x, particles_y, noise_u, 
                           num_steps, base_u):
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return

        x, y = particles_x[thread_id], particles_y[thread_id]
        
        for step in range(num_steps):
            u = base_u + noise_u[step]
            t = 0.4 - 6.0 / (1.0 + x*x + y*y)
            
            x_new = 1.0 + u * (x * math.cos(t) - y * math.sin(t))
            y_new = u * (x * math.sin(t) + y * math.cos(t))
            x, y = x_new, y_new

        particles_x[thread_id] = x
        particles_y[thread_id] = y

elif BACKEND == 'jax':
    # JAX functions for Mac
    @jit
    def evolve_henon_steps(state, noise_params, base_a, base_b):
        x, y = state
        
        def step_fn(carry, noise_step):
            x, y = carry
            a = base_a + noise_step[0]
            b = base_b + noise_step[1]
            x_new = 1.0 - a * x * x + y
            y_new = b * x
            return jnp.array([x_new, y_new]), None
        
        final_state, _ = jax.lax.scan(step_fn, jnp.array([x, y]), noise_params)
        return final_state

    @jit  
    def evolve_clifford_steps(state, noise_params, base_a, base_b, base_c, base_d):
        x, y = state
        
        def step_fn(carry, noise_step):
            x, y = carry
            a = base_a + noise_step[0]
            b = base_b + noise_step[1]
            c = base_c + noise_step[2]
            d = base_d + noise_step[3]
            
            x_new = jnp.sin(a * y) + c * jnp.cos(a * x)
            y_new = jnp.sin(b * x) + d * jnp.cos(b * y)
            return jnp.array([x_new, y_new]), None
        
        final_state, _ = jax.lax.scan(step_fn, jnp.array([x, y]), noise_params)
        return final_state

    @jit
    def evolve_ikeda_steps(state, noise_params, base_u):
        x, y = state
        
        def step_fn(carry, noise_step):
            x, y = carry
            u = base_u + noise_step[0]
            t = 0.4 - 6.0 / (1.0 + x*x + y*y)
            
            x_new = 1.0 + u * (x * jnp.cos(t) - y * jnp.sin(t))
            y_new = u * (x * jnp.sin(t) + y * jnp.cos(t))
            return jnp.array([x_new, y_new]), None
        
        final_state, _ = jax.lax.scan(step_fn, jnp.array([x, y]), noise_params)
        return final_state

    # Vectorized versions for all particles
    evolve_henon_all = vmap(evolve_henon_steps, in_axes=(0, None, None, None))
    evolve_clifford_all = vmap(evolve_clifford_steps, in_axes=(0, None, None, None, None, None))
    evolve_ikeda_all = vmap(evolve_ikeda_steps, in_axes=(0, None, None))

def evolve_particles(particles_x_gpu, particles_y_gpu, noise_segments, num_steps):
    """Dispatch to appropriate kernel based on attractor choice and backend"""
    if BACKEND == 'cuda':
        # CUDA backend (Windows)
        if attractor_choice == 1:  # Henon
            evolve_henon_kernel[NUM_BLOCKS, NUM_THREADS_PER_BLOCK](
                particles_x_gpu, particles_y_gpu, 
                noise_segments['a'], noise_segments['b'],
                num_steps, BASE_PARAMS['a'], BASE_PARAMS['b']
            )
        elif attractor_choice == 2:  # Clifford
            evolve_clifford_kernel[NUM_BLOCKS, NUM_THREADS_PER_BLOCK](
                particles_x_gpu, particles_y_gpu,
                noise_segments['a'], noise_segments['b'], 
                noise_segments['c'], noise_segments['d'],
                num_steps, BASE_PARAMS['a'], BASE_PARAMS['b'], 
                BASE_PARAMS['c'], BASE_PARAMS['d']
            )
        elif attractor_choice == 3:  # Ikeda
            evolve_ikeda_kernel[NUM_BLOCKS, NUM_THREADS_PER_BLOCK](
                particles_x_gpu, particles_y_gpu, noise_segments['u'],
                num_steps, BASE_PARAMS['u']
            )
    
    elif BACKEND == 'jax':
        # JAX backend (Mac)
        # Prepare noise parameters for JAX (steps x params)
        if attractor_choice == 1:  # Henon
            noise_params = jnp.stack([noise_segments['a'], noise_segments['b']], axis=1)
            new_states = evolve_henon_all(
                jnp.stack([particles_x_gpu, particles_y_gpu], axis=1),
                noise_params, BASE_PARAMS['a'], BASE_PARAMS['b']
            )
        elif attractor_choice == 2:  # Clifford
            noise_params = jnp.stack([
                noise_segments['a'], noise_segments['b'], 
                noise_segments['c'], noise_segments['d']
            ], axis=1)
            new_states = evolve_clifford_all(
                jnp.stack([particles_x_gpu, particles_y_gpu], axis=1),
                noise_params, BASE_PARAMS['a'], BASE_PARAMS['b'],
                BASE_PARAMS['c'], BASE_PARAMS['d']
            )
        elif attractor_choice == 3:  # Ikeda
            noise_params = noise_segments['u'][:, None]  # Add dimension for consistency
            new_states = evolve_ikeda_all(
                jnp.stack([particles_x_gpu, particles_y_gpu], axis=1),
                noise_params, BASE_PARAMS['u']
            )
        
        # Update particle positions
        particles_x_gpu[:] = new_states[:, 0]
        particles_y_gpu[:] = new_states[:, 1]

# --- Main Script ---
if __name__ == "__main__":
    total_particles = NUM_BLOCKS * NUM_THREADS_PER_BLOCK
    
    # Initialize particle ensemble spread across phase space
    print(f"Initializing {total_particles} particles...")
    init_x = np.random.uniform(INIT_X_MIN, INIT_X_MAX, size=total_particles).astype(np.float32)
    init_y = np.random.uniform(INIT_Y_MIN, INIT_Y_MAX, size=total_particles).astype(np.float32)
    
    if BACKEND == 'cuda':
        particles_x_gpu = cp.asarray(init_x)
        particles_y_gpu = cp.asarray(init_y)
    else:  # jax
        particles_x_gpu = jnp.array(init_x)
        particles_y_gpu = jnp.array(init_y)
    
    current_noise_idx = 0
    
    # --- Pullback Phase ---
    print("Performing pullback to settle on random attractor...")
    pullback_batch_size = 1000  # Process in batches to avoid kernel timeout
    
    for i in range(0, PULLBACK_STEPS, pullback_batch_size):
        batch_size = min(pullback_batch_size, PULLBACK_STEPS - i)
        
        if BACKEND == 'cuda':
            noise_segment_dict = {param: cp.asarray(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        else:  # jax
            noise_segment_dict = {param: jnp.array(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        
        evolve_particles(particles_x_gpu, particles_y_gpu, noise_segment_dict, batch_size)
        current_noise_idx += batch_size
        
        if i % 1000 == 0:
            print(f"Pullback progress: {i}/{PULLBACK_STEPS}")
    
    if BACKEND == 'cuda':
        cp.cuda.stream.get_current_stream().synchronize()
    print("Pullback complete.")
    
    # Debug: Check particle positions after pullback
    if BACKEND == 'cuda':
        debug_x = cp.asnumpy(particles_x_gpu)
        debug_y = cp.asnumpy(particles_y_gpu)
    else:  # jax
        debug_x = np.array(particles_x_gpu)
        debug_y = np.array(particles_y_gpu)
    
    print(f"After pullback - X range: [{debug_x.min():.3f}, {debug_x.max():.3f}]")
    print(f"After pullback - Y range: [{debug_y.min():.3f}, {debug_y.max():.3f}]")
    print(f"Particles in bounds: {np.sum((debug_x >= X_MIN) & (debug_x <= X_MAX) & (debug_y >= Y_MIN) & (debug_y <= Y_MAX))}/{len(debug_x)}")
    
    # --- Snapshot Generation Phase ---
    snapshot_images = []
    
    for snap_idx in range(NUM_SNAPSHOTS):
        print(f"Generating snapshot {snap_idx + 1}/{NUM_SNAPSHOTS}")
        
        # Get noise segment for this snapshot
        if BACKEND == 'cuda':
            noise_segment_dict = {param: cp.asarray(noise_dict[param][current_noise_idx : current_noise_idx + STEPS_PER_SNAPSHOT]) for param in param_names}
        else:  # jax
            noise_segment_dict = {param: jnp.array(noise_dict[param][current_noise_idx : current_noise_idx + STEPS_PER_SNAPSHOT]) for param in param_names}
        
        if any(len(noise_segment) < STEPS_PER_SNAPSHOT for noise_segment in noise_segment_dict.values()):
            print("Ran out of noise sequence!")
            break
        
        # Evolve particles for this snapshot
        evolve_particles(particles_x_gpu, particles_y_gpu, noise_segment_dict, STEPS_PER_SNAPSHOT)
        if BACKEND == 'cuda':
            cp.cuda.stream.get_current_stream().synchronize()
        current_noise_idx += STEPS_PER_SNAPSHOT
        
        # Get particle positions back to CPU for histogram
        if BACKEND == 'cuda':
            current_particles_x_cpu = cp.asnumpy(particles_x_gpu)
            current_particles_y_cpu = cp.asnumpy(particles_y_gpu)
        else:  # jax
            current_particles_x_cpu = np.array(particles_x_gpu)
            current_particles_y_cpu = np.array(particles_y_gpu)
        
        # Create 2D histogram for this snapshot
        hist_this_snapshot, _, _ = np.histogram2d(
            current_particles_x_cpu, current_particles_y_cpu, 
            bins=(WIDTH, HEIGHT), 
            range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]]
        )
        # Transpose because histogram2d(x,y) has x on first dim but imshow expects x on second dim
        hist_this_snapshot = hist_this_snapshot.T 
        
        snapshot_images.append(hist_this_snapshot)
    
    # --- Create morphing attractor animation ---
    print("Creating morphing attractor animation...")
    
    # Setup single plot for animation
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plt.style.use('dark_background')
    
    # Create initial plot
    first_img = (snapshot_images[0] > 0).astype(float)
    im = ax.imshow(1 - first_img, cmap='gray', vmin=0, vmax=1, 
                   origin='lower', interpolation='nearest',
                   extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'Morphing Stochastic {ATTRACTOR_NAME} Attractor', fontsize=16, color='white')
    
    # Add time indicator
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, color='white', verticalalignment='top')
    
    def animate(frame):
        # Update image data
        img_data = (snapshot_images[frame] > 0).astype(float)
        im.set_array(1 - img_data)
        
        # Update time indicator
        time_text.set_text(f'Time step: {frame * STEPS_PER_SNAPSHOT}')
        
        return [im, time_text]
    
    print(f"Creating animation with {len(snapshot_images)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(snapshot_images), 
                        interval=1000/FPS, blit=True, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=FPS)
    anim.save(GIF_FILENAME, writer=writer)
    print(f"Saved morphing {ATTRACTOR_NAME} attractor animation to {GIF_FILENAME}")
    
    plt.show()