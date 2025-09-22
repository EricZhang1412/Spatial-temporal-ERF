import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.spikingcnn import ConvNet, ConvNetDynamic
from functions.erf import compute_spatial_erf, compute_temporal_erf
from functions.dynamic_erf import *

def visualize_st_erf(models_config, input_size=65, time_steps=10, num_runs=20):
    """
    Visualize Spatiotemporal ERF for different model configurations.

    Args:
        models_config: List of dictionaries containing model configurations
        input_size: Spatial dimension of the input
        time_steps: Total number of time steps to simulate
        num_runs: Number of runs to average over for stability
    
    Returns:
        fig: Matplotlib figure containing the visualizations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_configs = len(models_config)
    
    # Calculate grid dimensions
    num_time_delays = time_steps - 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_configs, num_time_delays, 
                            figsize=(5 * num_time_delays, 3 * num_configs))
    
    # Handle the case of a single row (one model config)
    if num_configs == 1:
        axes = [axes]

    cmap_bw = LinearSegmentedColormap.from_list('BlackToWhite', [(0, 0, 0), (1, 1, 1)])
        
    for config_idx, config in enumerate(models_config):
        # Create model based on configuration
        model = ConvNetDynamic(
            num_layers=config['layers'],
            kernel_size=config['kernel_size'],
            weight_type=config['weight_type'],
            tau=config['tau'],
            Vth=config['Vth'],
            surrogate_mode=config['surrogate_mode'],
            alpha=config['alpha'],
        ).to(device)
        
        # Compute ST-ERF for all time delays
        st_erf_volume = compute_st_erf_all_delays(model, input_size, time_steps, num_runs)
        
        # Normalize values for better visualization
        max_val = st_erf_volume.max()
        norm_st_erf = st_erf_volume / max_val if max_val > 0 else st_erf_volume

        config_title = f"Layers: {config.get('layers', 'N/A')}, "
        config_title += f"Kernel: {config.get('kernel_size', 'N/A')}, "
        config_title += f"τ: {config.get('tau', 'N/A')}, "
        config_title += f"{config.get('surrogate_mode', 'N/A')}"
        
        # Plot each time delay for this configuration
        for delay_idx in range(num_time_delays):
            # Get current axis
            if num_time_delays == 1:
                ax = axes[config_idx]
            else:
                if num_configs == 1:
                    ax = axes[delay_idx]
                else:
                    ax = axes[config_idx, delay_idx]
            
            im = ax.imshow(norm_st_erf[delay_idx], cmap=cmap_bw, vmin=0, vmax=1)
            
            # Add titles and annotations
            if delay_idx == 0:
                ax.set_ylabel(f"Config {config_idx+1}", fontsize=12)
            
            ax.set_title(f"Delay: {delay_idx+1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for the last column
            if delay_idx == num_time_delays - 1:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Normalized Gradient Magnitude')
    
    # Add super title with visualization details
    plt.suptitle("Spatiotemporal Effective Receptive Field (ST-ERF)", fontsize=16)
    
    # Add a text box with configuration details
    config_text = "\n".join([
        f"Config {i+1}: {cfg.get('layers', 'N/A')} layers, "
        f"kernel={cfg.get('kernel_size', 'N/A')}, "
        f"τ={cfg.get('tau', 'N/A')}, "
        f"{cfg.get('surrogate_mode', 'N/A')}"
        for i, cfg in enumerate(models_config)
    ])
    
    fig.text(0.02, 0.02, config_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for suptitle and bottom text
    return fig


def visualize_st_erf_comparison(models_config, time_delay=1, input_size=65, time_steps=10, num_runs=20):
    """
    Visualize and compare ST-ERF at a specific time delay across different model configurations.
    
    Args:
        models_config: List of dictionaries containing model configurations
        time_delay: The specific time delay to visualize
        input_size: Spatial dimension of the input
        time_steps: Total number of time steps to simulate
        num_runs: Number of runs to average over for stability
    
    Returns:
        fig: Matplotlib figure containing the comparison visualizations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_configs = len(models_config)
    
    # Calculate grid dimensions
    cols = min(4, num_configs)
    rows = (num_configs + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    cmap_bw = LinearSegmentedColormap.from_list('BlackToWhite', [(0, 0, 0), (1, 1, 1)])
    
    # For storing max value for normalization
    all_erfs = []
    
    # First pass: compute all ERFs
    for config_idx, config in enumerate(models_config):
        # Create model based on configuration
        model = ConvNetDynamic(
            num_layers=config['layers'],
            kernel_size=config['kernel_size'],
            weight_type=config['weight_type'],
            tau=config['tau'],
            Vth=config['Vth'],
            surrogate_mode=config['surrogate_mode'],
            alpha=config['alpha'],
        ).to(device)
        
        # Compute ST-ERF for the specific time delay
        st_erf = compute_st_erf(model, input_size, time_steps, time_delay, num_runs)
        all_erfs.append(st_erf)
    
    # Find global max for consistent colormap
    global_max = np.max([erf.max() for erf in all_erfs])
    
    # Second pass: plot with consistent normalization
    for config_idx, (config, st_erf) in enumerate(zip(models_config, all_erfs)):
        if config_idx < len(axes):
            ax = axes[config_idx]
            
            im = ax.imshow(st_erf / global_max, cmap=cmap_bw, vmin=0, vmax=1)
            
            # Create title for this configuration
            config_title = f"Layers: {config.get('layers', 'N/A')}, "
            config_title += f"Kernel: {config.get('kernel_size', 'N/A')}\n"
            config_title += f"τ: {config.get('tau', 'N/A')}, "
            config_title += f"{config.get('surrogate_mode', 'N/A')}"
            
            ax.set_title(config_title)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(config_idx + 1, len(axes)):
        axes[i].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes, label='Normalized Gradient Magnitude')
    
    plt.suptitle(f'ST-ERF Comparison at Time Delay = {time_delay}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    dynamic_models_config = [
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig1 = visualize_st_erf(dynamic_models_config, input_size=65, time_steps=4, num_runs=20)
    fig1.savefig("st_erf_visualization_bw.pdf", dpi=300, bbox_inches='tight')
    
    
    
