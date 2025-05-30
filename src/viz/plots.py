"""
Analysis plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os
import time

def plot_lyapunov_analysis(
    history: List[Dict],
    attractor_name: str,
    noise_config: Dict,
    output_dir: str
) -> str:
    """Create enhanced visualization of Lyapunov analysis"""
    
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract time series
    times = [entry['time'] for entry in history]
    mean_lyap1_values = [entry['mean_lyap1'] for entry in history]
    mean_lyap2_values = [entry['mean_lyap2'] for entry in history]
    max_lyap1_values = [entry['max_lyap1'] for entry in history]
    max_lyap2_values = [entry['max_lyap2'] for entry in history]
    mean_ky_values = [entry['mean_ky_dim'] for entry in history]
    max_ky_values = [entry['max_ky_dim'] for entry in history]
    
    # Clean data
    def clean_data(data):
        return np.array([x for x in data if np.isfinite(x)])
    
    mean_lyap1_clean = clean_data(mean_lyap1_values)
    mean_lyap2_clean = clean_data(mean_lyap2_values)
    max_lyap1_clean = clean_data(max_lyap1_values)
    max_lyap2_clean = clean_data(max_lyap2_values)
    
    # Time series plot
    ax1.plot(times, mean_lyap1_clean, 'cyan', linewidth=1.5, alpha=0.8, label='Mean L1')
    ax1.plot(times, mean_lyap2_clean, 'magenta', linewidth=1.5, alpha=0.8, label='Mean L2')
    ax1.plot(times, max_lyap1_clean, 'orange', linewidth=1.5, alpha=0.8, label='Max L1')
    ax1.plot(times, max_lyap2_clean, 'red', linewidth=1.5, alpha=0.8, label='Max L2')
    ax1.axhline(y=0, color='yellow', linestyle='--', alpha=0.7, label='Neutral')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Lyapunov Exponents')
    ax1.set_title(f'{attractor_name} Attractor - Lyapunov Spectrum Evolution')
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
    
    # Histogram for mean Lyapunov spectrum
    ax3.hist(mean_lyap1_clean, bins=30, color='cyan', alpha=0.5, label='Mean L1')
    ax3.hist(mean_lyap2_clean, bins=30, color='magenta', alpha=0.5, label='Mean L2')
    ax3.axvline(x=np.mean(mean_lyap1_clean), color='cyan', linestyle='-', linewidth=2)
    ax3.axvline(x=np.mean(mean_lyap2_clean), color='magenta', linestyle='-', linewidth=2)
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
    
    # Save plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, 
                           f"lyapunov_analysis_{attractor_name.lower()}_{timestamp}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def save_lyapunov_data(
    history: List[Dict],
    attractor_name: str,
    noise_config: Dict,
    output_dir: str
) -> str:
    """Save Lyapunov analysis data to file"""
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, 
                           f"lyapunov_{attractor_name.lower()}_{timestamp}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Lyapunov exponent analysis for {attractor_name} attractor\n")
        
        # Write noise configuration
        if noise_config['type'] == "gaussian":
            f.write(f"# Noise: Gaussian (mean={noise_config['params']['mean']}, "
                   f"std={noise_config['params']['std']})\n")
        elif noise_config['type'] == "uniform":
            f.write(f"# Noise: Uniform [{noise_config['params']['low']}, "
                   f"{noise_config['params']['high']}]\n")
        else:
            f.write("# Noise: None\n")
        
        # Write statistics
        mean_lyap1 = np.mean([h['mean_lyap1'] for h in history if np.isfinite(h['mean_lyap1'])])
        mean_lyap2 = np.mean([h['mean_lyap2'] for h in history if np.isfinite(h['mean_lyap2'])])
        std_lyap1 = np.std([h['mean_lyap1'] for h in history if np.isfinite(h['mean_lyap1'])])
        std_lyap2 = np.std([h['mean_lyap2'] for h in history if np.isfinite(h['mean_lyap2'])])
        
        f.write(f"# Mean Lyapunov spectrum: L1={mean_lyap1:.6f}±{std_lyap1:.6f}, "
               f"L2={mean_lyap2:.6f}±{std_lyap2:.6f}\n")
        
        # Write header
        f.write("# Time\tMean_L1\tMean_L2\tMax_L1\tMax_L2\tMean_KY\tMax_KY\n")
        
        # Write data
        for entry in history:
            def fmt(x):
                if x is None or not np.isfinite(x):
                    return "nan"
                return f"{x:.8f}"
            
            f.write(f"{entry['time']}\t"
                   f"{fmt(entry['mean_lyap1'])}\t"
                   f"{fmt(entry['mean_lyap2'])}\t"
                   f"{fmt(entry['max_lyap1'])}\t"
                   f"{fmt(entry['max_lyap2'])}\t"
                   f"{fmt(entry['mean_ky_dim'])}\t"
                   f"{fmt(entry['max_ky_dim'])}\n")
    
    return filename 