"""
CUDA backend for accelerated attractor evolution, will be heavily optimised later
"""

import cupy as cp
from numba import cuda
import math
from typing import Dict, Optional
from ..attractors.base import Attractor

class CUDAKernels:
    """CUDA kernel collection for different attractors"""
    
    @staticmethod
    @cuda.jit
    def henon_evolution_kernel(particles_x, particles_y, noise_a, noise_b, 
                              num_steps, base_a, base_b):
        """CUDA kernel for Hénon map evolution"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return
        
        x, y = particles_x[thread_id], particles_y[thread_id]
        
        for step in range(num_steps):
            if not (math.isfinite(x) and math.isfinite(y)):
                x, y = 0.0, 0.0
                continue
                
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            
            x_new = 1.0 - a * x * x + y
            y_new = b * x
            x, y = x_new, y_new
        
        particles_x[thread_id] = x
        particles_y[thread_id] = y
    
    @staticmethod
    @cuda.jit
    def henon_lyapunov_kernel(particles_x, particles_y,
                             tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                             noise_a, noise_b, num_steps, base_a, base_b,
                             lyap_sums1, lyap_sums2, step_count,
                             renorm_interval):
        """CUDA kernel for Hénon with Lyapunov tracking"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return
        
        x, y = particles_x[thread_id], particles_y[thread_id]
        txx, txy = tangent_xx[thread_id], tangent_xy[thread_id]
        tyx, tyy = tangent_yx[thread_id], tangent_yy[thread_id]
        
        local_lyap_sum1 = 0.0
        local_lyap_sum2 = 0.0
        local_step_count = 0
        
        for step in range(num_steps):
            if not (math.isfinite(x) and math.isfinite(y)):
                x, y = 0.0, 0.0
                txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                continue
            
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            
            # Evolve point
            x_new = 1.0 - a * x * x + y
            y_new = b * x
            
            # Jacobian elements for Hénon
            j11, j12, j21, j22 = -2.0 * a * x, 1.0, b, 0.0
            
            # Evolve tangent vectors
            txx_new = j11 * txx + j12 * tyx
            txy_new = j11 * txy + j12 * tyy
            tyx_new = j21 * txx + j22 * tyx
            tyy_new = j21 * txy + j22 * tyy
            
            if not (math.isfinite(txx_new) and math.isfinite(txy_new) and 
                    math.isfinite(tyx_new) and math.isfinite(tyy_new)):
                txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                x, y = x_new, y_new
                continue
            
            # Gram-Schmidt orthonormalization
            if (step + 1) % renorm_interval == 0:
                # First vector
                norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                if norm1 > 1e-12:
                    txx_new /= norm1
                    tyx_new /= norm1
                    if math.isfinite(math.log(norm1)):
                        local_lyap_sum1 += math.log(norm1)
                else:
                    txx_new, tyx_new = 1.0, 0.0
                
                # Second vector (orthogonalize)
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
    
    @staticmethod
    @cuda.jit
    def clifford_lyapunov_kernel(particles_x, particles_y,
                                tangent_xx, tangent_xy, tangent_yx, tangent_yy,
                                noise_a, noise_b, noise_c, noise_d, num_steps,
                                base_a, base_b, base_c, base_d,
                                lyap_sums1, lyap_sums2, step_count,
                                renorm_interval):
        """CUDA kernel for Clifford with Lyapunov tracking"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return
        
        x, y = particles_x[thread_id], particles_y[thread_id]
        txx, txy = tangent_xx[thread_id], tangent_xy[thread_id]
        tyx, tyy = tangent_yx[thread_id], tangent_yy[thread_id]
        
        local_lyap_sum1 = 0.0
        local_lyap_sum2 = 0.0
        local_step_count = 0
        
        for step in range(num_steps):
            if not (math.isfinite(x) and math.isfinite(y)):
                x, y = 0.0, 0.0
                txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                continue
            
            a = base_a + noise_a[step]
            b = base_b + noise_b[step]
            c = base_c + noise_c[step]
            d = base_d + noise_d[step]
            
            # Evolve point
            x_new = math.sin(a * y) + c * math.cos(a * x)
            y_new = math.sin(b * x) + d * math.cos(b * y)
            
            # Jacobian elements for Clifford
            j11 = -c * a * math.sin(a * x)
            j12 = a * math.cos(a * y)
            j21 = b * math.cos(b * x)
            j22 = -d * b * math.sin(b * y)
            
            # Evolve tangent vectors
            txx_new = j11 * txx + j12 * tyx
            txy_new = j11 * txy + j12 * tyy
            tyx_new = j21 * txx + j22 * tyx
            tyy_new = j21 * txy + j22 * tyy
            
            if not (math.isfinite(txx_new) and math.isfinite(txy_new) and 
                    math.isfinite(tyx_new) and math.isfinite(tyy_new)):
                txx, txy, tyx, tyy = 1.0, 0.0, 0.0, 1.0
                x, y = x_new, y_new
                continue
            
            # Gram-Schmidt orthonormalization
            if (step + 1) % renorm_interval == 0:
                # First vector
                norm1 = math.sqrt(txx_new * txx_new + tyx_new * tyx_new)
                if norm1 > 1e-12:
                    txx_new /= norm1
                    tyx_new /= norm1
                    if math.isfinite(math.log(norm1)):
                        local_lyap_sum1 += math.log(norm1)
                else:
                    txx_new, tyx_new = 1.0, 0.0
                
                # Second vector (orthogonalize)
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
    
    @staticmethod
    @cuda.jit
    def clifford_evolution_kernel(particles_x, particles_y, 
                                 noise_a, noise_b, noise_c, noise_d,
                                 num_steps, base_a, base_b, base_c, base_d):
        """CUDA kernel for Clifford map evolution"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return
        
        x, y = particles_x[thread_id], particles_y[thread_id]
        
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
        
        particles_x[thread_id] = x
        particles_y[thread_id] = y

    @staticmethod
    @cuda.jit
    def ikeda_evolution_kernel(particles_x, particles_y, noise_u,
                              num_steps, base_u):
        """CUDA kernel for Ikeda map evolution"""
        thread_id = cuda.grid(1)
        if thread_id >= particles_x.shape[0]:
            return
        
        x, y = particles_x[thread_id], particles_y[thread_id]
        
        for step in range(num_steps):
            if not (math.isfinite(x) and math.isfinite(y)):
                x, y = 0.0, 0.0
                continue
                
            u = base_u + noise_u[step]
            t = 0.4 - 6.0 / (1.0 + x*x + y*y)
            
            x_new = 1.0 + u * (x * math.cos(t) - y * math.sin(t))
            y_new = u * (x * math.sin(t) + y * math.cos(t))
            x, y = x_new, y_new
        
        particles_x[thread_id] = x
        particles_y[thread_id] = y


class CUDABackend:
    """High-performance CUDA backend for attractor simulation"""
    
    def __init__(self, num_blocks: int = 256, threads_per_block: int = 256):
        self.num_blocks = num_blocks
        self.threads_per_block = threads_per_block
        
        if not cuda.is_available():
            raise RuntimeError("No CUDA GPU detected!")
        
        device = cuda.get_current_device()
        print(f"CUDA GPU detected: {device.name.decode()}")
        print(f"Compute capability: {device.compute_capability}")
    
    def evolve_ensemble(self, attractor: Attractor, 
                       particles_x: cp.ndarray, particles_y: cp.ndarray,
                       noise_params: Dict[str, cp.ndarray], num_steps: int,
                       lyap_tracker: Optional[object] = None,
                       renorm_interval: int = 10) -> None:
        """Evolve particle ensemble using optimized CUDA kernels"""
        
        if lyap_tracker is not None and attractor.name == "Hénon":
            # Use Lyapunov tracking kernel for Hénon
            CUDAKernels.henon_lyapunov_kernel[self.num_blocks, self.threads_per_block](
                particles_x[:lyap_tracker.num_particles],
                particles_y[:lyap_tracker.num_particles],
                lyap_tracker.tangent_xx,
                lyap_tracker.tangent_xy,
                lyap_tracker.tangent_yx,
                lyap_tracker.tangent_yy,
                noise_params['a'],
                noise_params['b'],
                num_steps,
                attractor.params['a'],
                attractor.params['b'],
                lyap_tracker.lyap_sums1,
                lyap_tracker.lyap_sums2,
                lyap_tracker.step_count,
                renorm_interval
            )
            
            # Evolve remaining particles without Lyapunov tracking
            if lyap_tracker.num_particles < len(particles_x):
                CUDAKernels.henon_evolution_kernel[self.num_blocks, self.threads_per_block](
                    particles_x[lyap_tracker.num_particles:],
                    particles_y[lyap_tracker.num_particles:],
                    noise_params['a'],
                    noise_params['b'],
                    num_steps,
                    attractor.params['a'],
                    attractor.params['b']
                )
        
        elif lyap_tracker is not None and attractor.name == "Clifford":
            # Use Lyapunov tracking kernel for Clifford
            CUDAKernels.clifford_lyapunov_kernel[self.num_blocks, self.threads_per_block](
                particles_x[:lyap_tracker.num_particles],
                particles_y[:lyap_tracker.num_particles],
                lyap_tracker.tangent_xx,
                lyap_tracker.tangent_xy,
                lyap_tracker.tangent_yx,
                lyap_tracker.tangent_yy,
                noise_params['a'],
                noise_params['b'],
                noise_params['c'],
                noise_params['d'],
                num_steps,
                attractor.params['a'],
                attractor.params['b'],
                attractor.params['c'],
                attractor.params['d'],
                lyap_tracker.lyap_sums1,
                lyap_tracker.lyap_sums2,
                lyap_tracker.step_count,
                renorm_interval
            )
            
            # Evolve remaining particles without Lyapunov tracking
            if lyap_tracker.num_particles < len(particles_x):
                CUDAKernels.clifford_evolution_kernel[self.num_blocks, self.threads_per_block](
                    particles_x[lyap_tracker.num_particles:],
                    particles_y[lyap_tracker.num_particles:],
                    noise_params['a'],
                    noise_params['b'],
                    noise_params['c'],
                    noise_params['d'],
                    num_steps,
                    attractor.params['a'],
                    attractor.params['b'],
                    attractor.params['c'],
                    attractor.params['d']
                )
        
        elif attractor.name == "Hénon":
            CUDAKernels.henon_evolution_kernel[self.num_blocks, self.threads_per_block](
                particles_x, particles_y,
                noise_params['a'], noise_params['b'],
                num_steps,
                attractor.params['a'], attractor.params['b']
            )
        
        elif attractor.name == "Clifford":
            CUDAKernels.clifford_evolution_kernel[self.num_blocks, self.threads_per_block](
                particles_x, particles_y,
                noise_params['a'], noise_params['b'],
                noise_params['c'], noise_params['d'],
                num_steps,
                attractor.params['a'], attractor.params['b'],
                attractor.params['c'], attractor.params['d']
            )
        
        elif attractor.name == "Ikeda":
            CUDAKernels.ikeda_evolution_kernel[self.num_blocks, self.threads_per_block](
                particles_x, particles_y,
                noise_params['u'],
                num_steps,
                attractor.params['u']
            )
        
        else:
            raise ValueError(f"No CUDA kernel available for attractor: {attractor.name}")
        
        # Synchronize to ensure completion
        cuda.synchronize() 