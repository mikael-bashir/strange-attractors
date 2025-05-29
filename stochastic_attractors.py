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

# --- Lyapunov Exponent Configuration ---
LYAP_COMPUTATION = True  # Enable/disable lyapunov tracking
LYAP_RENORM_INTERVAL = 10  # Renormalize every N steps to prevent overflow
LYAP_TRACK_PARTICLES = 100  # Number of particles to track for lyap computation
LYAP_SAVE_INTERVAL = 100   # Save lyap values every N steps

# Noise distribution types and generators
class NoiseType:
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"

def generate_noise(size, noise_config):
    """
    Generate noise based on configuration.
    noise_config = {
        'type': NoiseType.UNIFORM | NoiseType.GAUSSIAN | NoiseType.NONE,
        'params': {
            # for uniform:
            'low': float,  # lower bound
            'high': float, # upper bound
            # for gaussian:
            'mean': float,
            'std': float,
        }
    }
    """
    if noise_config['type'] == NoiseType.NONE:
        return np.zeros(size, dtype=np.float32)
    elif noise_config['type'] == NoiseType.UNIFORM:
        return np.random.uniform(
            noise_config['params']['low'],
            noise_config['params']['high'],
            size=size
        ).astype(np.float32)
    elif noise_config['type'] == NoiseType.GAUSSIAN:
        return np.random.normal(
            noise_config['params']['mean'],
            noise_config['params']['std'],
            size=size
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise type: {noise_config['type']}")

# --- Attractor Configurations ---
ATTRACTORS = {
    1: {
        'name': 'Hénon',
        'params': {'a': 1.4, 'b': 0.3},
        # 'noise': {
        #     'type': NoiseType.GAUSSIAN,
        #     'params': {
        #         'mean': 0.5,
        #         'std': 0.005
        #     }
        # },

        'noise': {
            'type': NoiseType.UNIFORM,
            'params': {
                'low': -0.001,
                'high': 0.001
            }
        },
        'bounds': {'x': (-2.0, 2.0), 'y': (-0.6, 0.6)},
        'init_bounds': {'x': (-1.0, 1.0), 'y': (-0.5, 0.5)}
    },
    2: {
        'name': 'Clifford',
        'params': {'a': -1.7, 'b': 1.8, 'c': -0.9, 'd': -0.4},
        'noise': {
            'type': NoiseType.UNIFORM,
            'params': {
                'low': -0.15,
                'high': 0.15
            }
        },
        'bounds': {'x': (-2.5, 2.5), 'y': (-2.5, 2.5)},
        'init_bounds': {'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}
    },
    3: {
        'name': 'Ikeda',
        'params': {'u': 0.90},
        'noise': {
            'type': NoiseType.GAUSSIAN,
            'params': {
                'mean': 0.0,
                'std': 0.01
            }
        },
        'bounds': {'x': (0.0, 2.0), 'y': (-2.0, 2.0)},
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

# Compute effective noise range based on noise type
if config['noise']['type'] == NoiseType.GAUSSIAN:
    NOISE_RANGE = config['noise']['params']['std']
elif config['noise']['type'] == NoiseType.UNIFORM:
    # For uniform distribution, range is (high - low)/2
    NOISE_RANGE = (config['noise']['params']['high'] - config['noise']['params']['low']) / 2
else:
    NOISE_RANGE = 0.0

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
# Non-uniform pullback snapshot intervals to capture early dynamics
PULLBACK_SNAPSHOT_STEPS = [1, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2800, 3600, 4000]
INTERMEDIATE_STEPS = 1000  # Steps between pullback and main phase
INTERMEDIATE_SNAPSHOTS = 5  # Snapshots during intermediate phase
STEPS_PER_SNAPSHOT = 30
NUM_SNAPSHOTS = 120  # Main phase snapshots
FPS = 15

# Generate noise for all parameters
param_names = list(BASE_PARAMS.keys())
TOTAL_NOISE_LENGTH = PULLBACK_STEPS + (NUM_SNAPSHOTS * STEPS_PER_SNAPSHOT) + 1000

print(f"Generating {TOTAL_NOISE_LENGTH} noise values for {len(param_names)} parameters...")
noise_dict = {}
for param in param_names:
    noise_dict[param] = generate_noise(TOTAL_NOISE_LENGTH, config['noise'])

# Output
RESULTS_BASE_DIR = "results"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join(RESULTS_BASE_DIR, f"{TIMESTAMP}_{config['name'].lower()}_attractor")

if not os.path.exists(RESULTS_BASE_DIR):
    os.makedirs(RESULTS_BASE_DIR)
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

GIF_FILENAME = os.path.join(RUN_DIR, f"morphing_{ATTRACTOR_NAME.lower()}.gif")

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
    def evolve_henon_with_lyap_kernel(particles_x, particles_y, 
                                     tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                                     noise_a, noise_b, num_steps, base_a, base_b,
                                     lyap_sums1, lyap_sums2, step_count):
        """Evolve henon with lyapunov exponent computation"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return

        x, y = particles_x[thread_id], particles_y[thread_id]
        
        # Initialize tangent vectors as identity matrix
        txx, txy = tangent_xx[thread_id], tangent_xy[thread_id] 
        tyx, tyy = tangent_yx[thread_id], tangent_yy[thread_id]
        
        local_lyap_sum1 = 0.0  # for first exponent
        local_lyap_sum2 = 0.0  # for second exponent
        local_step_count = 0
        
        for step in range(num_steps):
            # Check for divergence
            if not (math.isfinite(x) and math.isfinite(y)):
                x = 0.0  # Reset to origin
                y = 0.0
                txx, txy = 1.0, 0.0  # Reset tangent space
                tyx, tyy = 0.0, 1.0
                continue
            
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            
            # Evolve main trajectory
            x_new = 1.0 - a * x * x + y
            y_new = b * x
            
            # Check for overflow in evolution
            if not (math.isfinite(x_new) and math.isfinite(y_new)):
                x = 0.0
                y = 0.0
                txx, txy = 1.0, 0.0
                tyx, tyy = 0.0, 1.0
                continue
            
            # Compute jacobian at current point
            # J = [[-2*a*x, 1], [b, 0]]
            j11, j12 = -2.0 * a * x, 1.0
            j21, j22 = b, 0.0
            
            # Evolve tangent vectors: T_new = J * T_old
            txx_new = j11 * txx + j12 * tyx
            txy_new = j11 * txy + j12 * tyy
            tyx_new = j21 * txx + j22 * tyx  
            tyy_new = j21 * txy + j22 * tyy
            
            # Check for overflow in tangent space - explicit checks instead of generator
            if not math.isfinite(txx_new) or not math.isfinite(txy_new) or not math.isfinite(tyx_new) or not math.isfinite(tyy_new):
                txx, txy = 1.0, 0.0
                tyx, tyy = 0.0, 1.0
                x, y = x_new, y_new  # Keep orbit evolution
                continue
            
            # Gram-Schmidt orthonormalization every LYAP_RENORM_INTERVAL steps
            if (step + 1) % LYAP_RENORM_INTERVAL == 0:
                # First vector norm (largest exponent)
                norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                if norm1 > 1e-12:  # Avoid division by zero
                    txx_new /= norm1
                    tyx_new /= norm1
                    if math.isfinite(math.log(norm1)):
                        local_lyap_sum1 += math.log(norm1)
                else:
                    txx_new, tyx_new = 1.0, 0.0  # Reset if too small
                
                # Second vector: subtract projection, then normalize
                dot_product = txy_new * txx_new + tyy_new * tyx_new
                txy_new -= dot_product * txx_new
                tyy_new -= dot_product * tyx_new
                
                norm2 = math.sqrt(txy_new * txy_new + tyy_new * tyy_new)
                if norm2 > 1e-12:  # Avoid division by zero
                    txy_new /= norm2
                    tyy_new /= norm2
                    if math.isfinite(math.log(norm2)):
                        local_lyap_sum2 += math.log(norm2)
                else:
                    txy_new, tyy_new = 0.0, 1.0  # Reset if too small
                
                local_step_count += 1
            
            # Update state
            x, y = x_new, y_new
            txx, txy = txx_new, txy_new
            tyx, tyy = tyx_new, tyy_new

        # Store results
        particles_x[thread_id] = x
        particles_y[thread_id] = y
        tangent_xx[thread_id] = txx
        tangent_xy[thread_id] = txy
        tangent_yx[thread_id] = tyx
        tangent_yy[thread_id] = tyy
        lyap_sums1[thread_id] += local_lyap_sum1
        lyap_sums2[thread_id] += local_lyap_sum2
        step_count[thread_id] += local_step_count

    @cuda.jit
    def evolve_clifford_with_lyap_kernel(particles_x, particles_y, 
                                        tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                                        noise_a, noise_b, noise_c, noise_d, 
                                        num_steps, base_a, base_b, base_c, base_d,
                                        lyap_sums1, lyap_sums2, step_count):
        """Evolve clifford with lyapunov exponent computation"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return

        x, y = particles_x[thread_id], particles_y[thread_id]
        
        # Initialize tangent vectors as identity matrix
        txx, txy = tangent_xx[thread_id], tangent_xy[thread_id] 
        tyx, tyy = tangent_yx[thread_id], tangent_yy[thread_id]
        
        local_lyap_sum1 = 0.0  # for first exponent
        local_lyap_sum2 = 0.0  # for second exponent
        local_step_count = 0
        
        for step in range(num_steps):
            # Check for divergence
            if not (math.isfinite(x) and math.isfinite(y)):
                x = 0.0  # Reset to origin
                y = 0.0
                txx, txy = 1.0, 0.0  # Reset tangent space
                tyx, tyy = 0.0, 1.0
                continue
            
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            c = base_c + noise_c[step]
            d = base_d + noise_d[step]
            
            # Evolve main trajectory
            x_new = math.sin(a * y) + c * math.cos(a * x)
            y_new = math.sin(b * x) + d * math.cos(b * y)
            
            # Check for overflow in evolution
            if not (math.isfinite(x_new) and math.isfinite(y_new)):
                x = 0.0
                y = 0.0
                txx, txy = 1.0, 0.0
                tyx, tyy = 0.0, 1.0
                continue
            
            # Compute jacobian at current point
            # J = [[-c*a*sin(a*x), a*cos(a*y)], [b*cos(b*x), -d*b*sin(b*y)]]
            j11 = -c * a * math.sin(a * x)
            j12 = a * math.cos(a * y)
            j21 = b * math.cos(b * x)
            j22 = -d * b * math.sin(b * y)
            
            # Evolve tangent vectors: T_new = J * T_old
            txx_new = j11 * txx + j12 * tyx
            txy_new = j11 * txy + j12 * tyy
            tyx_new = j21 * txx + j22 * tyx  
            tyy_new = j21 * txy + j22 * tyy
            
            # Check for overflow in tangent space - explicit checks
            if not math.isfinite(txx_new) or not math.isfinite(txy_new) or not math.isfinite(tyx_new) or not math.isfinite(tyy_new):
                txx, txy = 1.0, 0.0
                tyx, tyy = 0.0, 1.0
                x, y = x_new, y_new  # Keep orbit evolution
                continue
            
            # Gram-Schmidt orthonormalization every LYAP_RENORM_INTERVAL steps
            if (step + 1) % LYAP_RENORM_INTERVAL == 0:
                # First vector norm (largest exponent)
                norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                if norm1 > 1e-12:  # Avoid division by zero
                    txx_new /= norm1
                    tyx_new /= norm1
                    if math.isfinite(math.log(norm1)):
                        local_lyap_sum1 += math.log(norm1)
                else:
                    txx_new, tyx_new = 1.0, 0.0  # Reset if too small
                
                # Second vector: subtract projection, then normalize
                dot_product = txy_new * txx_new + tyy_new * tyx_new
                txy_new -= dot_product * txx_new
                tyy_new -= dot_product * tyx_new
                
                norm2 = math.sqrt(txy_new * txy_new + tyy_new * tyy_new)
                if norm2 > 1e-12:  # Avoid division by zero
                    txy_new /= norm2
                    tyy_new /= norm2
                    if math.isfinite(math.log(norm2)):
                        local_lyap_sum2 += math.log(norm2)
                else:
                    txy_new, tyy_new = 0.0, 1.0  # Reset if too small
                
                local_step_count += 1
            
            # Update state
            x, y = x_new, y_new
            txx, txy = txx_new, txy_new
            tyx, tyy = tyx_new, tyy_new

        # Store results
        particles_x[thread_id] = x
        particles_y[thread_id] = y
        tangent_xx[thread_id] = txx
        tangent_xy[thread_id] = txy
        tangent_yx[thread_id] = tyx
        tangent_yy[thread_id] = tyy
        lyap_sums1[thread_id] += local_lyap_sum1
        lyap_sums2[thread_id] += local_lyap_sum2
        step_count[thread_id] += local_step_count

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

class LyapunovTracker:
    """Track lyapunov exponents during simulation"""
    
    def __init__(self, num_particles, backend='cuda'):
        self.backend = backend
        self.num_particles = num_particles
        self.history = []
        self.total_time = 0
        
        if backend == 'cuda':
            # Initialize tangent space matrices (identity)
            self.tangent_xx = cp.ones(num_particles, dtype=cp.float32)
            self.tangent_xy = cp.zeros(num_particles, dtype=cp.float32)
            self.tangent_yx = cp.zeros(num_particles, dtype=cp.float32)
            self.tangent_yy = cp.ones(num_particles, dtype=cp.float32)
            
            # Running sums for both lyapunov exponents
            self.lyap_sums1 = cp.zeros(num_particles, dtype=cp.float32)  # first exponent
            self.lyap_sums2 = cp.zeros(num_particles, dtype=cp.float32)  # second exponent
            self.step_count = cp.zeros(num_particles, dtype=cp.int32)
    
    def get_current_exponents(self):
        """Get current lyapunov spectrum"""
        if self.backend == 'cuda':
            sums1_cpu = cp.asnumpy(self.lyap_sums1)
            sums2_cpu = cp.asnumpy(self.lyap_sums2)
            counts_cpu = cp.asnumpy(self.step_count)
            
            valid_mask = counts_cpu > 0
            if np.any(valid_mask):
                # Individual particle lyapunov spectrums
                lyap1 = sums1_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
                lyap2 = sums2_cpu[valid_mask] / counts_cpu[valid_mask] / LYAP_RENORM_INTERVAL
                
                # Mean and max of first exponent
                mean_lyap1 = np.mean(lyap1)
                max_lyap1 = np.max(lyap1)
                
                # Mean and max of second exponent
                mean_lyap2 = np.mean(lyap2)
                max_lyap2 = np.min(lyap2)  # use min as this is typically negative
                
                return (mean_lyap1, mean_lyap2), (max_lyap1, max_lyap2)
        return (0.0, 0.0), (0.0, 0.0)
    
    def save_snapshot(self):
        """Save current lyapunov spectrum to history"""
        mean_spectrum, max_spectrum = self.get_current_exponents()
        mean_lyap1, mean_lyap2 = mean_spectrum
        max_lyap1, max_lyap2 = max_spectrum
        
        # Compute Kaplan-Yorke dimensions
        mean_ky_dim = compute_kaplan_yorke_dim(mean_spectrum)
        max_ky_dim = compute_kaplan_yorke_dim(max_spectrum)
        
        self.history.append({
            'time': self.total_time,
            'mean_lyap1': mean_lyap1,
            'mean_lyap2': mean_lyap2,
            'max_lyap1': max_lyap1,
            'max_lyap2': max_lyap2,
            'mean_ky_dim': mean_ky_dim,
            'max_ky_dim': max_ky_dim,
            'noise_range': NOISE_RANGE
        })
        return mean_spectrum, max_spectrum

    def evolve_with_lyap(self, particles_x_gpu, particles_y_gpu, noise_segments, num_steps):
        """Evolve particles while computing lyapunov exponents"""
        if self.backend == 'cuda':
            if attractor_choice == 1:  # Henon
                evolve_henon_with_lyap_kernel[NUM_BLOCKS, NUM_THREADS_PER_BLOCK](
                    particles_x_gpu, particles_y_gpu,
                    self.tangent_xx, self.tangent_xy, self.tangent_yx, self.tangent_yy,
                    noise_segments['a'], noise_segments['b'],
                    num_steps, BASE_PARAMS['a'], BASE_PARAMS['b'],
                    self.lyap_sums1, self.lyap_sums2, self.step_count
                )
            elif attractor_choice == 2:  # Clifford
                evolve_clifford_with_lyap_kernel[NUM_BLOCKS, NUM_THREADS_PER_BLOCK](
                    particles_x_gpu, particles_y_gpu,
                    self.tangent_xx, self.tangent_xy, self.tangent_yx, self.tangent_yy,
                    noise_segments['a'], noise_segments['b'], 
                    noise_segments['c'], noise_segments['d'],
                    num_steps, BASE_PARAMS['a'], BASE_PARAMS['b'],
                    BASE_PARAMS['c'], BASE_PARAMS['d'],
                    self.lyap_sums1, self.lyap_sums2, self.step_count
                )
            
            cp.cuda.stream.get_current_stream().synchronize()
            self.total_time += num_steps

def evolve_particles_with_lyap_option(particles_x_gpu, particles_y_gpu, noise_segments, num_steps, lyap_tracker=None):
    """Enhanced evolve function that optionally computes lyapunov exponents"""
    if lyap_tracker is not None and LYAP_COMPUTATION and attractor_choice in [1, 2]:  # Henon and Clifford
        lyap_tracker.evolve_with_lyap(particles_x_gpu, particles_y_gpu, noise_segments, num_steps)
    else:
        evolve_particles(particles_x_gpu, particles_y_gpu, noise_segments, num_steps)

def compute_kaplan_yorke_dim(lyap_spectrum):
    """
    Compute the Kaplan-Yorke (Lyapunov) dimension from a lyapunov spectrum.
    For a 2D system, KY dim = 1 + λ1/|λ2| if λ1 + λ2 < 0 and λ1 > 0
    where λ1 is the largest lyapunov exponent.
    """
    if len(lyap_spectrum) != 2:
        return None
    
    lambda1, lambda2 = lyap_spectrum
    
    # Check conditions for KY dimension formula
    if lambda1 + lambda2 < 0 and lambda1 > 0:
        ky_dim = 1.0 + lambda1/abs(lambda2)
        return min(ky_dim, 2.0)  # cap at embedding dimension
    elif lambda1 <= 0:
        return 1.0  # periodic/stable behavior
    else:
        return 2.0  # fully chaotic

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
    
    # Initialize lyapunov tracker if enabled
    lyap_tracker = None
    if LYAP_COMPUTATION and attractor_choice in [1, 2]:  # Henon and Clifford
        print(f"Initializing lyapunov tracker for {LYAP_TRACK_PARTICLES} particles...")
        lyap_tracker = LyapunovTracker(LYAP_TRACK_PARTICLES, BACKEND)
    
    current_noise_idx = 0
    
    # --- Pullback Phase with Non-uniform Snapshots ---
    print("Performing pullback to settle on random attractor...")
    pullback_batch_size = 1000  # Process in batches to avoid kernel timeout
    snapshot_images = []
    next_snapshot_idx = 0
    current_step = 0
    
    # First snapshot at step 0 (initial condition)
    if PULLBACK_SNAPSHOT_STEPS[0] == 0:
        if BACKEND == 'cuda':
            snap_x = cp.asnumpy(particles_x_gpu)
            snap_y = cp.asnumpy(particles_y_gpu)
        else:  # jax
            snap_x = np.array(particles_x_gpu)
            snap_y = np.array(particles_y_gpu)
        
        hist_snap, _, _ = np.histogram2d(snap_x, snap_y, bins=(WIDTH, HEIGHT), 
                                       range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]])
        snapshot_images.append(hist_snap.T)
        print(f"Pullback snapshot 1/{len(PULLBACK_SNAPSHOT_STEPS)} at step 0 (initial condition)")
        next_snapshot_idx = 1
    
    for i in range(0, PULLBACK_STEPS, pullback_batch_size):
        batch_size = min(pullback_batch_size, PULLBACK_STEPS - i)
        
        if BACKEND == 'cuda':
            noise_segment_dict = {param: cp.asarray(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        else:  # jax
            noise_segment_dict = {param: jnp.array(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        
        # Use subset of particles for lyapunov computation to save memory
        if lyap_tracker is not None:
            lyap_particles_x = particles_x_gpu[:LYAP_TRACK_PARTICLES] 
            lyap_particles_y = particles_y_gpu[:LYAP_TRACK_PARTICLES]
            evolve_particles_with_lyap_option(lyap_particles_x, lyap_particles_y, 
                                            noise_segment_dict, batch_size, lyap_tracker)
            # Copy evolved lyap particles back
            particles_x_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_x
            particles_y_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_y
            # Evolve remaining particles normally
            if LYAP_TRACK_PARTICLES < total_particles:
                evolve_particles(particles_x_gpu[LYAP_TRACK_PARTICLES:], 
                               particles_y_gpu[LYAP_TRACK_PARTICLES:], 
                               noise_segment_dict, batch_size)
        else:
            evolve_particles(particles_x_gpu, particles_y_gpu, noise_segment_dict, batch_size)
        
        current_noise_idx += batch_size
        current_step = i + batch_size
        
        # Check if we need to capture snapshots in this range
        while (next_snapshot_idx < len(PULLBACK_SNAPSHOT_STEPS) and 
               PULLBACK_SNAPSHOT_STEPS[next_snapshot_idx] <= current_step):
            
            target_step = PULLBACK_SNAPSHOT_STEPS[next_snapshot_idx]
            
            if target_step <= current_step:  # We've passed or reached this snapshot point
                if BACKEND == 'cuda':
                    snap_x = cp.asnumpy(particles_x_gpu)
                    snap_y = cp.asnumpy(particles_y_gpu)
                else:  # jax
                    snap_x = np.array(particles_x_gpu)
                    snap_y = np.array(particles_y_gpu)
                
                hist_snap, _, _ = np.histogram2d(snap_x, snap_y, bins=(WIDTH, HEIGHT), 
                                               range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]])
                snapshot_images.append(hist_snap.T)
                print(f"Pullback snapshot {len(snapshot_images)}/{len(PULLBACK_SNAPSHOT_STEPS)} at step {target_step}")
                next_snapshot_idx += 1
        
        # Log lyapunov progress during pullback
        if lyap_tracker is not None and i % (LYAP_SAVE_INTERVAL * 10) == 0:
            current_lyap = lyap_tracker.save_snapshot()
            print(f"Pullback {i}/{PULLBACK_STEPS} - Lyapunov exp: {current_lyap}")
        elif i % 1000 == 0 and i > 0:  # Don't double-print with snapshot msg
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
    
    if lyap_tracker is not None:
        final_pullback_lyap = lyap_tracker.get_current_exponents()
        print(f"Final pullback lyapunov exponents: {final_pullback_lyap}")
    
    # --- Intermediate Phase ---
    print("Intermediate evolution phase...")
    intermediate_interval = INTERMEDIATE_STEPS // INTERMEDIATE_SNAPSHOTS
    
    for i in range(0, INTERMEDIATE_STEPS, intermediate_interval):
        batch_size = intermediate_interval
        
        if BACKEND == 'cuda':
            noise_segment_dict = {param: cp.asarray(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        else:  # jax
            noise_segment_dict = {param: jnp.array(noise_dict[param][current_noise_idx : current_noise_idx + batch_size]) for param in param_names}
        
        # Evolve with lyapunov tracking
        if lyap_tracker is not None:
            lyap_particles_x = particles_x_gpu[:LYAP_TRACK_PARTICLES] 
            lyap_particles_y = particles_y_gpu[:LYAP_TRACK_PARTICLES]
            evolve_particles_with_lyap_option(lyap_particles_x, lyap_particles_y, 
                                            noise_segment_dict, batch_size, lyap_tracker)
            particles_x_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_x
            particles_y_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_y
            if LYAP_TRACK_PARTICLES < total_particles:
                evolve_particles(particles_x_gpu[LYAP_TRACK_PARTICLES:], 
                               particles_y_gpu[LYAP_TRACK_PARTICLES:], 
                               noise_segment_dict, batch_size)
        else:
            evolve_particles(particles_x_gpu, particles_y_gpu, noise_segment_dict, batch_size)
        
        current_noise_idx += batch_size
        
        # Capture snapshot
        if BACKEND == 'cuda':
            snap_x = cp.asnumpy(particles_x_gpu)
            snap_y = cp.asnumpy(particles_y_gpu)
        else:  # jax
            snap_x = np.array(particles_x_gpu)
            snap_y = np.array(particles_y_gpu)
        
        hist_snap, _, _ = np.histogram2d(snap_x, snap_y, bins=(WIDTH, HEIGHT), 
                                       range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]])
        snapshot_images.append(hist_snap.T)
        
        if lyap_tracker is not None:
            current_lyap = lyap_tracker.save_snapshot()
            print(f"Intermediate snapshot {len(snapshot_images) - len(PULLBACK_SNAPSHOT_STEPS)}/{INTERMEDIATE_SNAPSHOTS} - Lyapunov exp: {current_lyap}")
        else:
            print(f"Intermediate snapshot {len(snapshot_images) - len(PULLBACK_SNAPSHOT_STEPS)}/{INTERMEDIATE_SNAPSHOTS}")
    
    # --- Main Snapshot Generation Phase ---
    for snap_idx in range(NUM_SNAPSHOTS):
        print(f"Generating main snapshot {snap_idx + 1}/{NUM_SNAPSHOTS}")
        
        # Get noise segment for this snapshot
        if BACKEND == 'cuda':
            noise_segment_dict = {param: cp.asarray(noise_dict[param][current_noise_idx : current_noise_idx + STEPS_PER_SNAPSHOT]) for param in param_names}
        else:  # jax
            noise_segment_dict = {param: jnp.array(noise_dict[param][current_noise_idx : current_noise_idx + STEPS_PER_SNAPSHOT]) for param in param_names}
        
        if any(len(noise_segment) < STEPS_PER_SNAPSHOT for noise_segment in noise_segment_dict.values()):
            print("Ran out of noise sequence!")
            break
        
        # Evolve particles for this snapshot with lyapunov tracking
        if lyap_tracker is not None:
            lyap_particles_x = particles_x_gpu[:LYAP_TRACK_PARTICLES] 
            lyap_particles_y = particles_y_gpu[:LYAP_TRACK_PARTICLES]
            evolve_particles_with_lyap_option(lyap_particles_x, lyap_particles_y, 
                                            noise_segment_dict, STEPS_PER_SNAPSHOT, lyap_tracker)
            particles_x_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_x
            particles_y_gpu[:LYAP_TRACK_PARTICLES] = lyap_particles_y
            # Evolve remaining particles normally
            if LYAP_TRACK_PARTICLES < total_particles:
                evolve_particles(particles_x_gpu[LYAP_TRACK_PARTICLES:], 
                               particles_y_gpu[LYAP_TRACK_PARTICLES:], 
                               noise_segment_dict, STEPS_PER_SNAPSHOT)
        else:
            evolve_particles(particles_x_gpu, particles_y_gpu, noise_segment_dict, STEPS_PER_SNAPSHOT)
        
        if BACKEND == 'cuda':
            cp.cuda.stream.get_current_stream().synchronize()
        current_noise_idx += STEPS_PER_SNAPSHOT
        
        # Save lyapunov snapshot periodically
        if lyap_tracker is not None and snap_idx % (LYAP_SAVE_INTERVAL // STEPS_PER_SNAPSHOT) == 0:
            current_lyap = lyap_tracker.save_snapshot()
            print(f"Main snapshot {snap_idx} - Lyapunov exp: {current_lyap}")
        
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
        
        # Update time indicator based on which phase we're in
        num_pullback_frames = len(PULLBACK_SNAPSHOT_STEPS)
        
        if frame < num_pullback_frames:
            # Pullback phase - use actual step from our non-uniform array
            actual_step = PULLBACK_SNAPSHOT_STEPS[frame]
            time_text.set_text(f'Pullback phase - Step: {actual_step}')
        elif frame < num_pullback_frames + INTERMEDIATE_SNAPSHOTS:
            # Intermediate phase
            intermediate_frame = frame - num_pullback_frames
            actual_step = PULLBACK_STEPS + intermediate_frame * intermediate_interval
            time_text.set_text(f'Intermediate phase - Step: {actual_step}')
        else:
            # Main phase
            main_frame = frame - num_pullback_frames - INTERMEDIATE_SNAPSHOTS
            actual_step = PULLBACK_STEPS + INTERMEDIATE_STEPS + main_frame * STEPS_PER_SNAPSHOT
            time_text.set_text(f'Main phase - Step: {actual_step}')
        
        return [im, time_text]
    
    print(f"Creating animation with {len(snapshot_images)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(snapshot_images), 
                        interval=1000/FPS, blit=True, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=FPS)
    anim.save(GIF_FILENAME, writer=writer)
    print(f"Saved morphing {ATTRACTOR_NAME} attractor animation to {GIF_FILENAME}")
    
    # --- Lyapunov Analysis ---
    if lyap_tracker is not None and len(lyap_tracker.history) > 0:
        print("\n=== LYAPUNOV EXPONENT ANALYSIS ===")
        
        # Extract time series
        times = [entry['time'] for entry in lyap_tracker.history]
        mean_lyap1_values = [entry['mean_lyap1'] for entry in lyap_tracker.history]
        mean_lyap2_values = [entry['mean_lyap2'] for entry in lyap_tracker.history]
        max_lyap1_values = [entry['max_lyap1'] for entry in lyap_tracker.history]
        max_lyap2_values = [entry['max_lyap2'] for entry in lyap_tracker.history]
        mean_ky_values = [entry['mean_ky_dim'] for entry in lyap_tracker.history]
        max_ky_values = [entry['max_ky_dim'] for entry in lyap_tracker.history]
        
        # Filter out infinities and nans for plotting
        def clean_data(data):
            return np.array([x for x in data if np.isfinite(x)])
        
        mean_lyap1_clean = clean_data(mean_lyap1_values)
        mean_lyap2_clean = clean_data(mean_lyap2_values)
        max_lyap1_clean = clean_data(max_lyap1_values)
        max_lyap2_clean = clean_data(max_lyap2_values)
        
        # Recompute statistics with clean data
        mean_of_means1 = np.mean(mean_lyap1_clean)
        std_of_means1 = np.std(mean_lyap1_clean)
        mean_of_means2 = np.mean(mean_lyap2_clean)
        std_of_means2 = np.std(mean_lyap2_clean)
        
        mean_of_maxs1 = np.mean(max_lyap1_clean)
        std_of_maxs1 = np.std(max_lyap1_clean)
        mean_of_maxs2 = np.mean(max_lyap2_clean)
        std_of_maxs2 = np.std(max_lyap2_clean)

        # Create enhanced visualization with 4 subplots
        plt.style.use('dark_background')
        fig_lyap, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time series plot - both lyapunov exponents
        ax1.plot(times, mean_lyap1_clean, 'cyan', linewidth=1.5, alpha=0.8, label='Mean L1')
        ax1.plot(times, mean_lyap2_clean, 'magenta', linewidth=1.5, alpha=0.8, label='Mean L2')
        ax1.plot(times, max_lyap1_clean, 'orange', linewidth=1.5, alpha=0.8, label='Max L1')
        ax1.plot(times, max_lyap2_clean, 'red', linewidth=1.5, alpha=0.8, label='Max L2')
        ax1.axhline(y=0, color='yellow', linestyle='--', alpha=0.7, label='Neutral')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Lyapunov Exponents')
        ax1.set_title(f'{ATTRACTOR_NAME} Attractor - Lyapunov Spectrum Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Kaplan-Yorke dimension time series
        valid_mean_ky = [x for x in mean_ky_values if x is not None and np.isfinite(x)]
        valid_max_ky = [x for x in max_ky_values if x is not None and np.isfinite(x)]
        ax2.plot(times[:len(valid_mean_ky)], valid_mean_ky, 'cyan', linewidth=1.5, alpha=0.8, label='Mean KY')
        ax2.plot(times[:len(valid_max_ky)], valid_max_ky, 'orange', linewidth=1.5, alpha=0.8, label='Max KY')
        ax2.axhline(y=1.0, color='yellow', linestyle='--', alpha=0.7, label='D=1')
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='D=2')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Kaplan-Yorke Dimension')
        ax2.set_title('Kaplan-Yorke Dimension Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Histogram for mean lyapunov spectrum
        ax3.hist(mean_lyap1_clean, bins=30, color='cyan', alpha=0.5, label='Mean L1')
        ax3.hist(mean_lyap2_clean, bins=30, color='magenta', alpha=0.5, label='Mean L2')
        ax3.axvline(x=mean_of_means1, color='cyan', linestyle='-', linewidth=2)
        ax3.axvline(x=mean_of_means2, color='magenta', linestyle='-', linewidth=2)
        ax3.axvline(x=0, color='yellow', linestyle='--', alpha=0.7, label='Neutral')
        ax3.set_xlabel('Lyapunov Exponents')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Mean Lyapunov Spectrum Distribution')
        ax3.legend()
        
        # Histogram for Kaplan-Yorke dimension
        ax4.hist(valid_mean_ky, bins=30, color='cyan', alpha=0.5, label='Mean KY')
        ax4.hist(valid_max_ky, bins=30, color='orange', alpha=0.5, label='Max KY')
        if valid_mean_ky:
            ax4.axvline(x=np.mean(valid_mean_ky), color='cyan', linestyle='-', linewidth=2)
        if valid_max_ky:
            ax4.axvline(x=np.mean(valid_max_ky), color='orange', linestyle='-', linewidth=2)
        ax4.set_xlabel('Kaplan-Yorke Dimension')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Kaplan-Yorke Dimension Distribution')
        ax4.legend()
        
        plt.tight_layout()
        lyap_plot_filename = os.path.join(RUN_DIR, f"lyapunov_analysis_{ATTRACTOR_NAME.lower()}_{TIMESTAMP}.png")
        plt.savefig(lyap_plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved lyapunov analysis plot to {lyap_plot_filename}")
        
        # Print statistics with clean data
        print(f"Mean lyapunov spectrum: L1={mean_of_means1:.6f}±{std_of_means1:.6f}, L2={mean_of_means2:.6f}±{std_of_means2:.6f}")
        print(f"Max lyapunov spectrum: L1={mean_of_maxs1:.6f}±{std_of_maxs1:.6f}, L2={mean_of_maxs2:.6f}±{std_of_maxs2:.6f}")
        
        mean_ky = np.mean(valid_mean_ky) if valid_mean_ky else float('nan')
        max_ky = np.mean(valid_max_ky) if valid_max_ky else float('nan')
        print(f"Mean Kaplan-Yorke dimension: {mean_ky:.6f}")
        print(f"Max Kaplan-Yorke dimension: {max_ky:.6f}")
        
        # Print noise configuration
        noise_config = config['noise']
        if noise_config['type'] == NoiseType.GAUSSIAN:
            print(f"Noise type: Gaussian (μ={noise_config['params']['mean']}, σ={noise_config['params']['std']})")
        elif noise_config['type'] == NoiseType.UNIFORM:
            print(f"Noise type: Uniform [{noise_config['params']['low']}, {noise_config['params']['high']}]")
        else:
            print("Noise type: None")

        # Stability analysis based on maximal lyapunov (more standard)
        if mean_of_maxs1 > 0.01:
            print("DIAGNOSIS: Strong chaotic behavior (positive maximal lyapunov exp)")
        elif mean_of_maxs1 > -0.01:
            print("DIAGNOSIS: Near-neutral stability (maximal lyapunov exp ≈ 0)")
        else:
            print("DIAGNOSIS: Stable/periodic behavior (negative maximal lyapunov exp)")
        
        # Save lyapunov data with clean values
        lyap_filename = os.path.join(RUN_DIR, f"lyapunov_{ATTRACTOR_NAME.lower()}_{TIMESTAMP}.txt")
        with open(lyap_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Lyapunov exponent analysis for {ATTRACTOR_NAME} attractor\n")
            
            # Write noise configuration
            if noise_config['type'] == NoiseType.GAUSSIAN:
                f.write(f"# Noise: Gaussian (mean={noise_config['params']['mean']}, std={noise_config['params']['std']})\n")
            elif noise_config['type'] == NoiseType.UNIFORM:
                f.write(f"# Noise: Uniform [{noise_config['params']['low']}, {noise_config['params']['high']}]\n")
            else:
                f.write("# Noise: None\n")
            
            f.write(f"# Mean lyapunov spectrum: L1={mean_of_means1:.6f}±{std_of_means1:.6f}, L2={mean_of_means2:.6f}±{std_of_means2:.6f}\n")
            f.write(f"# Max lyapunov spectrum: L1={mean_of_maxs1:.6f}±{std_of_maxs1:.6f}, L2={mean_of_maxs2:.6f}±{std_of_maxs2:.6f}\n")
            f.write("# Time\tMean_L1\tMean_L2\tMax_L1\tMax_L2\tMean_KY\tMax_KY\n")
            
            # Write only finite values
            for t, ml1, ml2, mxl1, mxl2, mky, mxky in zip(times, 
                                                          mean_lyap1_values, mean_lyap2_values,
                                                          max_lyap1_values, max_lyap2_values,
                                                          mean_ky_values, max_ky_values):
                # Replace inf/nan with string representation
                def fmt(x):
                    if x is None or not np.isfinite(x):
                        return "nan"
                    return f"{x:.8f}"
                
                f.write(f"{t}\t{fmt(ml1)}\t{fmt(ml2)}\t{fmt(mxl1)}\t{fmt(mxl2)}\t{fmt(mky)}\t{fmt(mxky)}\n")
        
        print(f"Saved lyapunov data to {lyap_filename}")
        plt.show()
    else:
        plt.show()