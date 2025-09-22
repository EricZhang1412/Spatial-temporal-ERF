import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

def compute_st_erf(model, input_size=65, time_steps=10, time_delay=5, num_runs=20):
    """
    Compute the Spatiotemporal Effective Receptive Field for Spiking Neural Networks.
    
    Args:
        model: The SNN model
        input_size: Spatial dimension of the input
        time_steps: Total number of time steps to simulate
        time_delay: Temporal delay n in ST-ERF = ∂y_{0,0}[t]/∂x_{i,j}[t-n]
        num_runs: Number of runs to average over for stability
        
    Returns:
        st_erf: The spatiotemporal effective receptive field
    """
    device = next(model.parameters()).device
    gradients = []
    
    for _ in range(num_runs):
        # Create random spatiotemporal input
        # Shape: [1, time_steps, 1, input_size, input_size]
        x = torch.randn(time_steps, 1, 1, input_size, input_size).to(device)
        x.requires_grad_(True)
        
        # Forward pass through the SNN model
        # Assuming model can handle input of shape [B, T, C, H, W]
        output = model(x)
        
        # Create gradient signal (1 at center point in space and specific time)
        grad_output = torch.zeros_like(output)
        center = input_size // 2
        target_time = time_steps - 1  # Output at final time step
        grad_output[target_time, 0, 0, center, center] = 1.0
        
        # Backward pass
        output.backward(grad_output)
        
        # Get input gradients at the specific time delay
        # This captures ∂y_{0,0}[t]/∂x_{i,j}[t-n]
        source_time = target_time - time_delay
        if source_time >= 0:
            gradient = x.grad[source_time, 0, 0].abs().cpu().numpy()
            gradients.append(gradient)
    
    # Average gradients across runs
    st_erf = np.mean(gradients, axis=0)
    return st_erf

def compute_st_erf_all_delays(model, input_size=65, time_steps=10, num_runs=20):
    """
    Compute ST-ERF for all possible time delays.
    
    Returns:
        st_erf_volume: A 3D volume of ST-ERF values, where the first dimension
                      represents the time delay (n in the ST-ERF definition)
    """
    device = next(model.parameters()).device
    st_erf_volume = []
    
    for time_delay in range(1, time_steps):
        st_erf = compute_st_erf(model, input_size, time_steps, time_delay, num_runs)
        st_erf_volume.append(st_erf)
    
    return np.array(st_erf_volume)