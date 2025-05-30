"""
Animation generation for strange attractors
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Tuple

WIDTH = 600
HEIGHT = 600
FPS = 15

def create_morphing_animation(
    snapshot_images: List[np.ndarray],
    attractor_name: str,
    bounds: Dict[str, Tuple[float, float]],
    output_dir: str,
    fps: int = FPS,
    pullback_steps: List[int] = None,
    intermediate_interval: int = None
) -> str:
    """
    Create a morphing animation from snapshot images.
    
    Args:
        snapshot_images: List of 2D histograms
        attractor_name: Name of the attractor
        bounds: Dictionary with 'x' and 'y' bounds
        output_dir: Directory to save the animation
        fps: Frames per second
        pullback_steps: List of steps where pullback snapshots were taken
        intermediate_interval: Steps between intermediate snapshots
    
    Returns:
        Path to the saved animation file
    """
    print(f"Creating morphing {attractor_name} attractor animation...")
    
    # Setup single plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plt.style.use('dark_background')
    
    # Create initial plot
    first_img = (snapshot_images[0] > 0).astype(float)
    im = ax.imshow(1 - first_img, cmap='gray', vmin=0, vmax=1,
                   origin='lower', interpolation='nearest',
                   extent=[bounds['x'][0], bounds['x'][1], 
                          bounds['y'][0], bounds['y'][1]])
    
    ax.set_xlim(bounds['x'])
    ax.set_ylim(bounds['y'])
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'Morphing Stochastic {attractor_name} Attractor', 
                 fontsize=16, color='white')
    
    # Add time indicator
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, color='white', verticalalignment='top')
    
    def animate(frame):
        # Update image data
        img_data = (snapshot_images[frame] > 0).astype(float)
        im.set_array(1 - img_data)
        
        # Update time indicator based on phase
        if pullback_steps and frame < len(pullback_steps):
            # Pullback phase
            actual_step = pullback_steps[frame]
            time_text.set_text(f'Pullback phase - Step: {actual_step}')
        elif intermediate_interval and frame < len(pullback_steps) + 5:
            # Intermediate phase
            intermediate_frame = frame - len(pullback_steps)
            actual_step = pullback_steps[-1] + intermediate_frame * intermediate_interval
            time_text.set_text(f'Intermediate phase - Step: {actual_step}')
        else:
            # Main phase
            main_frame = frame - len(pullback_steps) - 5
            actual_step = pullback_steps[-1] + 1000 + main_frame * 30
            time_text.set_text(f'Main phase - Step: {actual_step}')
        
        return [im, time_text]
    
    print(f"Creating animation with {len(snapshot_images)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(snapshot_images),
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save as GIF
    filename = os.path.join(output_dir, f"morphing_{attractor_name.lower()}.gif")
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    print(f"Saved animation to: {filename}")
    
    plt.close()
    return filename 