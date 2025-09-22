import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

def compute_spatial_erf(model, input_size=65, num_runs=20):
    """Compute the Effective Receptive Field."""
    device = next(model.parameters()).device
    gradients = []
    
    for _ in range(num_runs):
        # Create random input
        x = torch.randn(1, 1, input_size, input_size).to(device)
        x.requires_grad_(True)
        
        # Forward pass
        output = model(x)
        
        # Create gradient signal (1 at center, 0 elsewhere)
        grad_output = torch.zeros_like(output)
        center = input_size // 2
        grad_output[0, 0, center, center] = 1.0
        
        # Backward pass
        output.backward(grad_output, retain_graph=True)
        
        # Get input gradients
        gradient = x.grad.abs().cpu().numpy()[0, 0]
        gradients.append(gradient)
    
    # Average gradients across runs
    return np.mean(gradients, axis=0)

def compute_temporal_erf(model, input_size=65, num_runs=20):
    """Compute the Effective Receptive Field."""
    device = next(model.parameters()).device
    gradients = []
    
    for _ in range(num_runs):
        # Create random input
        x = torch.randn(4, 1, 1, input_size, input_size).to(device)
        x.requires_grad_(True)
        
        # Forward pass
        output, v_mem = model(x)
        
        grad_output = torch.zeros_like(v_mem)
        center = input_size // 2
        grad_output[-1, 0, 0, center, center] = 1.0 # at last time step

        # Backward pass
        v_mem.backward(grad_output)

        # Get input gradients
        # gradient = x.grad.abs().cpu().numpy()[0, 0]
        gradient = rearrange(x.grad.abs().cpu(), 'T B C H W -> B C T (H W)').numpy()[0, 0]
        # print(gradient.shape)
        gradients.append(gradient)

    # Average gradients across runs
    return np.mean(gradients, axis=1)

def compute_temporal_erf_2(model, input_size=65, time_steps=10, tau_range=None, num_runs=20):
    """
    Compute the Temporal Effective Receptive Field.
    
    Args:
        model
        input_size
        time_steps
        tau_range
        num_runs
    
    Returns:
        Dictionary mapping tau values to their corresponding ERF values
    """
    device = next(model.parameters()).device

    if tau_range is None:
        tau_range = range(time_steps - 1)

    erf_results = {tau: [] for tau in tau_range}
    
    for _ in range(num_runs):

        x = torch.randn(time_steps, 1, 1, input_size, input_size).to(device)
        x.requires_grad_(True)

        output, v_mem = model(x)

        center = input_size // 2
        T = time_steps - 1 

        grad_output = torch.zeros_like(v_mem)
        grad_output[T, 0, 0, center, center] = 1.0

        v_mem.backward(grad_output)

        input_grad = x.grad.abs().cpu()
        

        for tau in tau_range:
            t_idx = T - tau
            if t_idx < 0:  
                continue
           
            spatial_grad_sum = input_grad[t_idx, 0, 0].sum().item()
            erf_results[tau].append(spatial_grad_sum)

    return {tau: np.mean(values) for tau, values in erf_results.items()}