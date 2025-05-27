# Morphing Stochastic Attractors

A high-performance cross-platform implementation for visualizing evolving chaotic attractors with stochastic parameter perturbations. Generates mesmerizing animations of attractors morphing through phase space as their parameters drift under noise.

## What It Does

This script implements the concept of **random attractors** from dynamical systems theory - chaotic systems where parameters slowly evolve over time due to noise, causing the attractor's geometry to continuously morph. You'll see beautiful fractal structures breathing, twisting, and evolving as the underlying mathematics shifts.

The implementation follows the framework from Chekroun et al. for studying random attractors A(θ_t^ω), where:

- θ_t^ω represents time-evolving random parameters
- Each frame shows the instantaneous attractor geometry
- The morphing comes from accumulated parameter history

## Supported Attractors

1. **Hénon Attractor**: Classic quadratic map showing period-doubling routes to chaos
2. **Clifford Attractor**: Trigonometric strange attractor creating delicate fractal patterns
3. **Ikeda Attractor**: Complex attractor from laser dynamics with rich spiral structures

## Platform Support

### Windows (NVIDIA GPU)

- **Backend**: CUDA with Numba
- **Requirements**: NVIDIA GPU with CUDA support
- **Performance**: ~500k particles at 15+ FPS

### macOS (Apple Silicon)

- **Backend**: JAX with Metal Performance Shaders
- **Requirements**: M1/M2/M3 Mac
- **Performance**: ~80-95% of CUDA performance (can be much faster)

## Installation

### Windows (NVIDIA)

```bash
# Install CUDA toolkit first from NVIDIA
pip install cupy numba matplotlib pillow numpy
```

### macOS (Apple Silicon)

```bash
pip install jax[metal] matplotlib pillow numpy
```

## Usage

```bash
python stochastic_attractors.py
```

The script will:

1. Auto-detect your platform (Windows/Mac)
2. Check for GPU availability
3. Prompt you to choose an attractor (1-3)
4. Generate noise sequences
5. Perform pullback to settle on the random attractor
6. Create animation frames showing morphing dynamics
7. Save as animated GIF and display live preview

## Output

- **Live Animation**: Real-time preview during generation
- **Saved GIF**: `results_morphing_attractors/morphing_[attractor]_[timestamp].gif`
- **Debug Info**: Particle bounds and statistics

## Technical Details

### Algorithm

1. **Noise Generation**: Pre-generate correlated noise sequences for all parameters
2. **Pullback Phase**: Evolve particle ensemble to settle on the random attractor
3. **Snapshot Evolution**: Capture frames as particles evolve along noise path
4. **Histogram Rendering**: Convert particle clouds to density images

### Performance Optimizations

- **GPU Parallelization**: All particle evolution on GPU
- **Batched Processing**: Prevents kernel timeouts on long runs
- **Memory Efficient**: Streaming computation without storing full trajectories
- **JIT Compilation**: Both CUDA and JAX use just-in-time compilation

### Mathematical Implementation

**Hénon Map:**

```
x_{n+1} = 1 - (a + ξ_a)x_n² + y_n
y_{n+1} = (b + ξ_b)x_n
```

**Clifford Map:**

```
x_{n+1} = sin((a + ξ_a)y_n) + (c + ξ_c)cos((a + ξ_a)x_n)
y_{n+1} = sin((b + ξ_b)x_n) + (d + ξ_d)cos((b + ξ_b)y_n)
```

**Ikeda Map:**

```
t = 0.4 - 6/(1 + x_n² + y_n²)
x_{n+1} = 1 + (u + ξ_u)(x_n cos(t) - y_n sin(t))
y_{n+1} = (u + ξ_u)(x_n sin(t) + y_n cos(t))
```

Where ξ terms are uniform noise: ξ ~ U[-ε, ε]

## Configuration

### Attractor Parameters

Located in the `ATTRACTORS` dictionary:

- `params`: Base parameter values
- `noise_range`: Noise amplitude (ε)
- `bounds`: Viewing window [x_min, x_max, y_min, y_max]
- `init_bounds`: Initial particle distribution

### Simulation Settings

```python
NUM_BLOCKS = 2048          # GPU blocks (particles = blocks × threads)
NUM_THREADS_PER_BLOCK = 256
PULLBACK_STEPS = 4000      # Settling time
STEPS_PER_SNAPSHOT = 30    # Evolution between frames
NUM_SNAPSHOTS = 120        # Total animation frames
```

## Troubleshooting

### "No CUDA GPU detected"

- Ensure NVIDIA GPU with CUDA support
- Install CUDA toolkit from NVIDIA
- Check `nvidia-smi` works

### "Metal backend not detected"

- Install: `pip install jax[metal]`
- Restart terminal
- Verify M-series Mac

### White/Empty Frames

- Particles escaped bounds → increase viewing bounds
- Too much noise → reduce `noise_range`
- Check debug output for particle positions

### Performance Issues

- Reduce `NUM_BLOCKS` or `NUM_THREADS_PER_BLOCK`
- Decrease `NUM_SNAPSHOTS` for shorter videos
- Lower resolution with smaller `WIDTH`/`HEIGHT`

## Customization

### Adding New Attractors

1. Add entry to `ATTRACTORS` dictionary
2. Implement CUDA kernel (Windows) and JAX function (Mac)
3. Add dispatch case in `evolve_particles()`

### Parameter Exploration

- Try different base parameters from attractor galleries
- Adjust `noise_range` for more/less morphing
- Modify `STEPS_PER_SNAPSHOT` for smoother/jumpier transitions

### Visual Tweaks

- Change colormap in matplotlib calls
- Adjust `bounds` for different viewing windows
- Modify `WIDTH`/`HEIGHT` for resolution

## Theory Background

This implementation visualizes **random attractors** - a generalization of classical strange attractors where parameters evolve stochastically. Unlike static attractors, these show how chaotic systems respond to continuous parameter drift.

The key insight is that even small parameter perturbations can dramatically alter attractor geometry over time, creating the morphing effect. This has applications in climate modeling, neuroscience, and any system where parameters aren't perfectly constant.
