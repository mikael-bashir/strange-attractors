# Morphing Stochastic Attractors

GPU-accelerated visualization of chaotic attractors evolving under parameter noise, with real-time lyapunov exponent tracking.

## Features

- **3 Attractors**: Hénon, Clifford, Ikeda maps with stochastic perturbations
- **GPU Acceleration**: CUDA (Windows) / JAX Metal (Mac) backends
- **Lyapunov Tracking**: Real-time stability analysis (Hénon/Clifford)
- **Formation Dynamics**: See attractors emerge from random scatter

## Installation

**Windows (NVIDIA):**

```bash
pip install cupy numba matplotlib pillow numpy
```

**macOS (Apple Silicon):**

```bash
pip install jax[metal] matplotlib pillow numpy
```

## Usage

```bash
python stochastic_attractors.py
```

Choose attractor (1-3) → watch formation → get lyapunov analysis

## Output

Each run creates: `results/TIMESTAMP_attractor/`

- `morphing_[attractor].gif` - Animation
- `lyapunov_[attractor]_[time].txt` - Time series data
- `lyapunov_analysis_[attractor]_[time].png` - Analysis plots

## Configuration

Key parameters in `ATTRACTORS` dict:

- `noise_range` - Perturbation amplitude
- `bounds` - Viewing window
- `params` - Base parameter values

Simulation settings:

```python
LYAP_COMPUTATION = True    # Enable lyapunov tracking
LYAP_TRACK_PARTICLES = 100 # Particles for lyap computation
NUM_SNAPSHOTS = 120        # Animation length
```

## Lyapunov Features

**Supported:** Hénon (full theory), Clifford (jacobian-based)

**Analysis:**

- Real-time exponent monitoring
- Stability classification (chaotic/periodic/fixed point)
- Time series plots and statistics

**Theory Note:** For Hénon with uniform noise: λ₁ + λ₂ = 𝔼[ln|b_k|]

## Troubleshooting

- **No GPU**: Install CUDA toolkit (Windows) or JAX Metal (Mac)
- **Empty frames**: Reduce `noise_range` or increase `bounds`
- **Performance**: Lower `NUM_BLOCKS` or disable lyapunov
- **Lyap errors**: Adjust `LYAP_RENORM_INTERVAL` (5-20)
