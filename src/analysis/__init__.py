"""
Analysis and tracking modules
"""

from .lyapunov import LyapunovTracker

# comprehensive regime analysis system
# 
# theory.md: confidence intervals vs deterministic intervals
# statistical bounds that are 800x+ tighter than worst-case guarantees
# enabling actionable regime classification instead of useless "uncertain" results

# regime analysis - the new generative kernel
from .regime_analyzer import RegimeAnalyzer, RegimeConfig, RegimeClassification

from .noise_sweep import NoiseSweepRunner, NoiseSweepConfig, NoisePointResult
# results analysis
from .noise_sweep import NoiseSweepResults

# convenience functions for quick analysis
from .noise_sweep import quick_noise_sweep, find_chaos_sync_boundary

__all__ = ['LyapunovTracker'] 
