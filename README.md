# Strange Attractors

A high-performance CUDA-accelerated simulation framework for studying stochastic strange attractors.

## Features

- Multiple attractor maps (Hénon, Clifford, Ikeda)
- Stochastic parameter evolution with configurable noise
- Lyapunov exponent tracking and analysis
- Visualization tools for attractor morphing
- CUDA GPU acceleration

## Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Requires NVIDIA GPU with CUDA support and compatible drivers.

## Usage

Basic usage with default parameters:

```bash
python -m src.main
# defaults: --num-blocks 2048, --threads-per-block 256 → num-particles=524288
```

Full options:

```bash
python -m src.main --attractor [henon|clifford|ikeda] \
                   --num-particles 10000 \
                   --num-steps 1000 \
                   --lyap-particles 1000 \
                   --num-blocks 2048 \
                   --threads-per-block 256 \
                   --output-dir results \
                   --save-animation \
                   --save-lyapunov
```

The program will interactively prompt for noise configuration:

1. Choose noise type (uniform/gaussian/none)
2. Enter noise parameters (bounds or mean/std)

## Output

- Attractor evolution animations (GIF) saved to `results/<timestamp>_<attractor>_attractor/`
- Lyapunov spectrum analysis plots saved to `results/<timestamp>_<attractor>_attractor/`
- Raw data files for further analysis saved to `results/<timestamp>_<attractor>_attractor/`

## Examples

Hénon map with uniform noise:

```bash
python -m src.main --attractor henon --save-animation --save-lyapunov
# Choose: 1 (uniform noise)
# Enter: -0.1, 0.1 (noise bounds)
```

Clifford attractor with Gaussian noise:

```bash
python -m src.main --attractor clifford --save-animation
# Choose: 2 (gaussian noise)
# Enter: 0.0, 0.05 (mean, std)
```

## License

MIT
