"""
CUDA backend for high-performance strange attractor computation
"""

import numpy as np
import cupy as cp
from numba import cuda
import math
from typing import Dict, Any, Tuple, Optional
from attractors.base import Attractor

class CUDAKernels:
    """optimized CUDA kernel collection"""
    
    @staticmethod
    @cuda.jit
    def henon_evolution_kernel(particles_x, particles_y, noise_a, noise_b, 
                              num_steps, base_a, base_b):
        thread_id = cuda.grid(1)
        stride = cuda.gridsize(1)
        
        for idx in range(thread_id, particles_x.shape[0], stride):
            x, y = particles_x[idx], particles_y[idx]
            
            for step in range(num_steps):
                if not (math.isfinite(x) and math.isfinite(y)):
                    x, y = 0.0, 0.0
                    continue
                    
                a = base_a + noise_a[step]
                b = base_b + noise_b[step]
                
                x_new = 1.0 - a * x * x + y
                y_new = b * x
                x, y = x_new, y_new
            
            particles_x[idx] = x
            particles_y[idx] = y
    
    @staticmethod
    @cuda.jit
    def henon_fused_kernel(particles_x, particles_y,
                          tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                          noise_a, noise_b, num_steps, base_a, base_b,
                          lyap_sums1, lyap_sums2, step_count,
                          renorm_interval):
        """Fused henon evolution + lyapunov tracking kernel for massive speedup"""
        thread_id = cuda.grid(1)
        stride = cuda.gridsize(1)
        
        for idx in range(thread_id, particles_x.shape[0], stride):
            # Load particle state
            x, y = particles_x[idx], particles_y[idx]
            txx, txy = tangent_xx[idx], tangent_xy[idx]
            tyx, tyy = tangent_yx[idx], tangent_yy[idx]
            
            # Local accumulators for lyapunov sums
            local_lyap_sum1 = 0.0
            local_lyap_sum2 = 0.0
            local_step_count = 0
            
            # Main evolution + lyapunov loop
            for step in range(num_steps):
                # Stability check
                if not (math.isfinite(x) and math.isfinite(y)):
                    x, y = 0.0, 0.0
                    txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                    continue
                
                # Get noisy parameters
                a = base_a + noise_a[step]
                b = base_b + noise_b[step]
                
                # Henon map evolution
                x_new = 1.0 - a * x * x + y
                y_new = b * x
                
                # Henon jacobian
                j11, j12, j21, j22 = -2.0 * a * x, 1.0, b, 0.0
                
                # New tangent = J * T
                txx_new = j11 * txx + j12 * tyx
                txy_new = j11 * txy + j12 * tyy
                tyx_new = j21 * txx + j22 * tyx
                tyy_new = j21 * txy + j22 * tyy
                
                # Stability check
                if not (math.isfinite(txx_new) and math.isfinite(txy_new) and 
                        math.isfinite(tyx_new) and math.isfinite(tyy_new)):
                    txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                    x, y = x_new, y_new
                    continue
                
                # Gram-Schmidt renormalization (every renorm_interval steps)
                if (step + 1) % renorm_interval == 0:
                    # First vector normalization
                    norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                    if norm1 > 1e-12:
                        txx_new /= norm1
                        tyx_new /= norm1
                        if math.isfinite(math.log(norm1)):
                            local_lyap_sum1 += math.log(norm1)
                    else:
                        txx_new, tyx_new = 1.0, 0.0
                    
                    # Gram-Schmidt orthogonalization of second vector
                    dot = txy_new * txx_new + tyy_new * tyx_new
                    txy_new -= dot * txx_new
                    tyy_new -= dot * tyx_new
                    
                    # Second vector normalization
                    norm2 = math.sqrt(txy_new * txy_new + tyy_new * tyy_new)
                    if norm2 > 1e-12:
                        txy_new /= norm2
                        tyy_new /= norm2
                        if math.isfinite(math.log(norm2)):
                            local_lyap_sum2 += math.log(norm2)
                    else:
                        txy_new, tyy_new = 0.0, 1.0
                    
                    local_step_count += 1
                
                # Update state
                x, y = x_new, y_new
                txx, txy = txx_new, txy_new
                tyx, tyy = tyx_new, tyy_new
            
            # Store results back to global memory
            particles_x[idx] = x
            particles_y[idx] = y
            tangent_xx[idx] = txx
            tangent_xy[idx] = txy
            tangent_yx[idx] = tyx
            tangent_yy[idx] = tyy
            lyap_sums1[idx] += local_lyap_sum1
            lyap_sums2[idx] += local_lyap_sum2
            step_count[idx] += local_step_count
    
    @staticmethod
    @cuda.jit
    def clifford_evolution_kernel(particles_x, particles_y, 
                                 noise_a, noise_b, noise_c, noise_d,
                                 num_steps, base_a, base_b, base_c, base_d):
        thread_id = cuda.grid(1)
        stride = cuda.gridsize(1)
        
        for idx in range(thread_id, particles_x.shape[0], stride):
            x, y = particles_x[idx], particles_y[idx]
            
            for step in range(num_steps):
                if not (math.isfinite(x) and math.isfinite(y)):
                    x, y = 0.0, 0.0
                    continue
                    
                a = base_a + noise_a[step]
                b = base_b + noise_b[step]
                c = base_c + noise_c[step]
                d = base_d + noise_d[step]
                
                x_new = math.sin(a * y) + c * math.cos(a * x)
                y_new = math.sin(b * x) + d * math.cos(b * y)
                x, y = x_new, y_new
            
            particles_x[idx] = x
            particles_y[idx] = y

    @staticmethod
    @cuda.jit
    def clifford_fused_kernel(particles_x, particles_y,
                             tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                             noise_a, noise_b, noise_c, noise_d, num_steps,
                             base_a, base_b, base_c, base_d,
                             lyap_sums1, lyap_sums2, step_count,
                             renorm_interval):
        """Fused kernel for henon + clifford"""
        thread_id = cuda.grid(1)
        stride = cuda.gridsize(1)
        
        for idx in range(thread_id, particles_x.shape[0], stride):
            # Load particle state
            x, y = particles_x[idx], particles_y[idx]
            txx, txy = tangent_xx[idx], tangent_xy[idx]
            tyx, tyy = tangent_yx[idx], tangent_yy[idx]
            
            # Local accumulators for lyapunov sums
            local_lyap_sum1 = 0.0
            local_lyap_sum2 = 0.0
            local_step_count = 0
            
            # Main evolution + lyapunov loop
            for step in range(num_steps):
                # Stability check
                if not (math.isfinite(x) and math.isfinite(y)):
                    x, y = 0.0, 0.0
                    txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                    continue
                
                # Noisy params
                a = base_a + noise_a[step]
                b = base_b + noise_b[step]
                c = base_c + noise_c[step]
                d = base_d + noise_d[step]
                

                x_new = math.sin(a * y) + c * math.cos(a * x)
                y_new = math.sin(b * x) + d * math.cos(b * y)
                
                # Clifford jacobian
                j11 = -c * a * math.sin(a * x)
                j12 = a * math.cos(a * y)
                j21 = b * math.cos(b * x)
                j22 = -d * b * math.sin(b * y)
                
                txx_new = j11 * txx + j12 * tyx
                txy_new = j11 * txy + j12 * tyy
                tyx_new = j21 * txx + j22 * tyx
                tyy_new = j21 * txy + j22 * tyy
                
                # Another stability check
                if not (math.isfinite(txx_new) and math.isfinite(txy_new) and 
                        math.isfinite(tyx_new) and math.isfinite(tyy_new)):
                    txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                    x, y = x_new, y_new
                    continue
                
                if (step + 1) % renorm_interval == 0:
                    norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                    if norm1 > 1e-12:
                        txx_new /= norm1
                        tyx_new /= norm1
                        if math.isfinite(math.log(norm1)):
                            local_lyap_sum1 += math.log(norm1)
                    else:
                        txx_new, tyx_new = 1.0, 0.0
                    
                    dot = txy_new * txx_new + tyy_new * tyx_new
                    txy_new -= dot * txx_new
                    tyy_new -= dot * tyx_new
                    
                    norm2 = math.sqrt(txy_new * txy_new + tyy_new * tyy_new)
                    if norm2 > 1e-12:
                        txy_new /= norm2
                        tyy_new /= norm2
                        if math.isfinite(math.log(norm2)):
                            local_lyap_sum2 += math.log(norm2)
                    else:
                        txy_new, tyy_new = 0.0, 1.0
                    
                    local_step_count += 1
                
                x, y = x_new, y_new
                txx, txy = txx_new, txy_new
                tyx, tyy = tyx_new, tyy_new
            
            # Store results back to global memory
            particles_x[idx] = x
            particles_y[idx] = y
            tangent_xx[idx] = txx
            tangent_xy[idx] = txy
            tangent_yx[idx] = tyx
            tangent_yy[idx] = tyy
            lyap_sums1[idx] += local_lyap_sum1
            lyap_sums2[idx] += local_lyap_sum2
            step_count[idx] += local_step_count


class CUDABackend:
    """optimized CUDA backend"""
    
    def __init__(self):
        if not cuda.is_available():
            raise RuntimeError("No CUDA GPU detected!")
        
        device = cuda.get_current_device()
        print(f"CUDA GPU detected: {device.name.decode()}")
        print(f"Compute capability: {device.compute_capability}")
        
        # optimized launch config for my RTX 4050 (20 SMs, 2560 cores)
        self.threads_per_block = 512
        self.max_blocks = device.MULTIPROCESSOR_COUNT * 4
    
    def _get_launch_config(self, num_particles: int) -> Tuple[int, int]:
        blocks = min((num_particles + self.threads_per_block - 1) // self.threads_per_block, 
                    self.max_blocks)
        return (blocks, self.threads_per_block)
    
    def evolve_ensemble(self, attractor: Attractor, 
                       particles_x: cp.ndarray, particles_y: cp.ndarray,
                       noise_params: Dict[str, cp.ndarray], num_steps: int,
                       lyap_tracker: Optional[object] = None,
                       renorm_interval: int = 10) -> None:
        
        blocks, tpb = self._get_launch_config(len(particles_x))

        # some if statements to check if the attractor is henon or clifford
        
        if lyap_tracker is not None and attractor.name == "Hénon":
            lyap_blocks, lyap_tpb = self._get_launch_config(lyap_tracker.num_particles)
            CUDAKernels.henon_fused_kernel[lyap_blocks, lyap_tpb](
                particles_x[:lyap_tracker.num_particles],
                particles_y[:lyap_tracker.num_particles],
                lyap_tracker.tangent_xx, lyap_tracker.tangent_xy,
                lyap_tracker.tangent_yx, lyap_tracker.tangent_yy,
                noise_params['a'], noise_params['b'], num_steps,
                np.float32(attractor.params['a']), np.float32(attractor.params['b']),
                lyap_tracker.lyap_sums1, lyap_tracker.lyap_sums2,
                lyap_tracker.step_count, renorm_interval
            )
            
            if lyap_tracker.num_particles < len(particles_x):
                remain_blocks, remain_tpb = self._get_launch_config(
                    len(particles_x) - lyap_tracker.num_particles
                )
                CUDAKernels.henon_evolution_kernel[remain_blocks, remain_tpb](
                    particles_x[lyap_tracker.num_particles:],
                    particles_y[lyap_tracker.num_particles:],
                    noise_params['a'], noise_params['b'], num_steps,
                    np.float32(attractor.params['a']), np.float32(attractor.params['b'])
                )
        
        elif lyap_tracker is not None and attractor.name == "Clifford":
            lyap_blocks, lyap_tpb = self._get_launch_config(lyap_tracker.num_particles)
            CUDAKernels.clifford_fused_kernel[lyap_blocks, lyap_tpb](
                particles_x[:lyap_tracker.num_particles],
                particles_y[:lyap_tracker.num_particles],
                lyap_tracker.tangent_xx, lyap_tracker.tangent_xy,
                lyap_tracker.tangent_yx, lyap_tracker.tangent_yy,
                noise_params['a'], noise_params['b'],
                noise_params['c'], noise_params['d'], num_steps,
                np.float32(attractor.params['a']), np.float32(attractor.params['b']),
                np.float32(attractor.params['c']), np.float32(attractor.params['d']),
                lyap_tracker.lyap_sums1, lyap_tracker.lyap_sums2,
                lyap_tracker.step_count, renorm_interval
            )
            
            if lyap_tracker.num_particles < len(particles_x):
                remain_blocks, remain_tpb = self._get_launch_config(
                    len(particles_x) - lyap_tracker.num_particles
                )
                CUDAKernels.clifford_evolution_kernel[remain_blocks, remain_tpb](
                    particles_x[lyap_tracker.num_particles:],
                    particles_y[lyap_tracker.num_particles:],
                    noise_params['a'], noise_params['b'],
                    noise_params['c'], noise_params['d'], num_steps,
                    np.float32(attractor.params['a']), np.float32(attractor.params['b']),
                    np.float32(attractor.params['c']), np.float32(attractor.params['d'])
                )
        
        elif attractor.name == "Hénon":
            CUDAKernels.henon_evolution_kernel[blocks, tpb](
                particles_x, particles_y,
                noise_params['a'], noise_params['b'], num_steps,
                np.float32(attractor.params['a']), np.float32(attractor.params['b'])
            )
        
        elif attractor.name == "Clifford":
            CUDAKernels.clifford_evolution_kernel[blocks, tpb](
                particles_x, particles_y,
                noise_params['a'], noise_params['b'],
                noise_params['c'], noise_params['d'], num_steps,
                np.float32(attractor.params['a']), np.float32(attractor.params['b']),
                np.float32(attractor.params['c']), np.float32(attractor.params['d'])
            )
        
        else:
            raise ValueError(f"No CUDA kernel for: {attractor.name}") 
