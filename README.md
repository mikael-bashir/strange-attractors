# Strange Attractors

A high-performance CUDA-accelerated simulation framework for studying stochastic strange attractors.

## Features

- Henon & Clifford maps
- Uniform noise by default
- Regime analysis via deterministic & confidence intervals
- Noise-range sweep & phase diagram
- Precise critical-noise boundary detection
- Kaplan-Yorke dimension tracking
- Attractor visualization & animation

## Install

```
pip install -r requirements.txt
```

## Usage

```bash
# cd into src first
# single-point regime analysis
python -m main regime -a henon -n 0.001 -p 10000 -o results

# noise sweep
python -m main sweep -a henon -r 0.0001 0.01 -n 20 -o sweeps

# find critical boundary
python -m main find-critical -a henon -r 0.001 0.1 -e 1e-5 -v -o critical

# visualization
python -m main visualize -a henon -n 0.001 -p 10000 -o figures  # both animation & analysis
python -m main visualize -a henon -n 0.001 --animation-only     # just the animation
python -m main visualize -a henon -n 0.001 --analysis-only      # just analysis plots
```

## Common Options

```
-a, --attractor      attractor type (henon, clifford)
-n, --noise         noise amplitude
-p, --params        custom attractor parameters (JSON)
-t, --trajectories  number of trajectories
-P, --particles     particles per trajectory
-L, --lyap          particles for lyapunov tracking
-o, --output        output directory
-s, --save-data     save detailed trajectory data
```

## Output

Each command writes results to `<output-dir>_<timestamp>/`:

### Regime Analysis

- `regime_analysis.json`: classification & confidence intervals

### Noise Sweep

- `noise_sweep_results.json`: phase diagram & transitions

### Critical Boundary

- `critical_noise.json`: precise boundary location

### Visualization

- `morphing_<attractor>.gif`: attractor evolution animation
- `lyapunov_analysis_<attractor>.png`: analysis plots
- `lyapunov_<attractor>.txt`: numerical data (with -s flag)

## License

MIT
