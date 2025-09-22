import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec

from models.spikingcnn import ConvNet, ConvNetDynamic
from functions.erf import compute_spatial_erf, compute_temporal_erf, compute_temporal_erf_2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

def gaussian(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def visualize_erf(models_config, input_size=14, num_runs=50):
    """Visualize ERF for different model configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_configs = len(models_config)
    # fig, axes = plt.subplots(6, num_configs//6, figsize=(15, 8))
    fig, axes = plt.subplots(1, num_configs, figsize=(2*num_configs, 3))
    if num_configs == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for idx, config in enumerate(models_config):
        model = ConvNet(
            num_layers=config['layers'],
            kernel_size=config['kernel_size'],
            weight_type=config['weight_type'],
            activation=config['activation']
        ).to(device)
        print(model)
        
        # Compute ERF
        erf = compute_spatial_erf(model, input_size, num_runs)

        # Plot
        ax = axes[idx]
        im = ax.imshow(erf, cmap='gray')
        ax.set_title(f"{config['layers']} layers\n{config['weight_type']}\n{config['activation']}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_erf_fit(models_config, input_size=14, num_runs=50, single_width=5):
    """Visualize square ERF maps with aligned profile plots.
    
    Parameters:
    -----------
    models_config : list of dict
        Model configurations to visualize
    input_size : int
        Spatial size of input stimulus
    num_runs : int
        Number of trials for ERF computation
    single_width : float
        Base width per model column (inches)
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_configs = len(models_config)
    
    # Calculate figure dimensions
    profile_height = single_width * 0.4  # Profile plot height relative to image
    fig_width = single_width * num_configs
    fig_height = single_width + 2 * profile_height  # Image + two profile rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        3, num_configs,
        height_ratios=[single_width, profile_height, profile_height],  # Enforce square image
        width_ratios=[1]*num_configs,
        wspace=0.3, hspace=0.5
    )
    
    for idx, config in enumerate(models_config):
        model = ConvNet(
            num_layers=config['layers'],
            kernel_size=config['kernel_size'],
            weight_type=config['weight_type'],
            activation=config['activation']
        ).to(device)
        erf = compute_spatial_erf(model, input_size, num_runs)
        
        # --- Square ERF Image ---
        ax_img = fig.add_subplot(gs[0, idx])
        
        # Key modification: Adjust image display to enforce square aspect
        im = ax_img.imshow(erf, cmap='gray', extent=[0, 1, 0, 1])  # Normalized coordinates
        ax_img.set_aspect('equal')  # Critical for square display
        ax_img.set_title(f"{config['layers']}L/{config['kernel_size']}K\n{config['activation']}", pad=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # --- Profiles ---
        h, w = erf.shape
        _plot_profile(
            fig.add_subplot(gs[1, idx]),
            np.arange(w), erf[h//2, :], 'Horizontal'
        )
        _plot_profile(
            fig.add_subplot(gs[2, idx]),
            np.arange(h), erf[:, w//2], 'Vertical'
        )
    
    plt.tight_layout()
    return fig

def _plot_profile(ax, x, profile, direction):
    """Standardized profile plotting with consistent scaling."""
    try:
        params, _ = curve_fit(gaussian, x, profile, 
                            p0=[profile.max(), len(x)/2, len(x)/4])
        ax.plot(x, profile, 'b-', label='Data')
        ax.plot(x, gaussian(x, *params), 'r--', label=f'Fit (σ={params[2]:.1f})')
    except RuntimeError:
        ax.plot(x, profile, 'b-', label='Data (Fit Failed)')
    
    # Uniform styling
    ax.set_xlim(x[0]-0.5, x[-1]+0.5)  # Add slight padding
    ax.set_title(direction, pad=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)



def visualize_temporal_erf(models_config, input_size=32, num_runs=20):
    """
    Visualize temporal ERF for different LIF configurations.
    
    Args:
        models_config: List of dictionaries containing model configurations
                      Each dict should have: {'tau': float, 'Vth': float, 
                                            'activation': str, 'title': str}
        input_size: Size of the spatial input dimension (H=W)
        T: Number of time steps
        num_runs: Number of random runs for averaging
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_configs = len(models_config)
    
    # Create subplot grid
    fig, axes = plt.subplots(7, num_configs//7, figsize=(15, 8))
    if num_configs <= 2:
        axes = np.array([axes])  # Ensure axes is 2D
    axes = axes.flatten()
    
    for idx, config in enumerate(models_config):
        # Create model with current configuration
        model = ConvNetDynamic(
            num_layers=config['layers'],
            kernel_size=config['kernel_size'],
            weight_type=config['weight_type'],
            tau=config['tau'],
            Vth=config['Vth'],
            surrogate_mode=config['surrogate_mode'],
            alpha=config['alpha'],
        ).to(device)

        print(model)

    # for idx, config in enumerate(models_config):
    #     model = ConvNet(
    #         num_layers=config['layers'],
    #         kernel_size=config['kernel_size'],
    #         weight_type=config['weight_type'],
    #         activation=config['activation']
    #     ).to(device)
        
        # Compute temporal ERF
        avg_temporal_grad = compute_temporal_erf_2(model, input_size, num_runs) # [T, (H*W)]
        # print(avg_temporal_grad.shape) 
        
        # Plot
        ax = axes[idx]
        time_steps = np.arange(len(avg_temporal_grad))
        ax.plot(time_steps, avg_temporal_grad, '-o', linewidth=2, markersize=4)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Average Gradient')
        ax.set_title(f"{config['surrogate_mode']}\n tau={config['tau']}")
        ax.grid(True, linestyle='--', alpha=0.7)
        

    plt.tight_layout()
    return fig



if __name__ == "__main__":
    static_models_config = [
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        # {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},

        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        # {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},

        # {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
        # {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_sigmoid', 'tau': 2.0, 'Vth': 1.0, 'alpha': 2.0},
        {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_sigmoid', 'tau': 2.0, 'Vth': 1.0, 'alpha': 4.0},
        {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_sigmoid', 'tau': 2.0, 'Vth': 1.0, 'alpha': 8.0},
        {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},

        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        # {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},

        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},   
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        # {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},

        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
        # {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'},
    ]
    ################for static processing################
    # plt.figure(figsize=(15, 8))
    # fig = visualize_erf(
    #     models_config=static_models_config,
    #     input_size=56,
    #     num_runs=50)
    # plt.show()
    # # save figure
    # fig.savefig('static_erf_visualization.pdf')

    dynamic_models_config = [
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 1.1, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 2.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},

        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 4.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},
    
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 8.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 8.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 8.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 8.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},
    
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 16.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 16.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 16.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 16.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},
    
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 32.0, 'Vth': 1.0, 'surrogate_mode': 'triangle', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 32.0, 'Vth': 1.0, 'surrogate_mode': 'sigmoid', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 32.0, 'Vth': 1.0, 'surrogate_mode': 'arctan', 'alpha': 2.0},
        {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'tau': 32.0, 'Vth': 1.0, 'surrogate_mode': 'rectangle', 'alpha': 2.0},
    ]

    # dynamic_models_config = [
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 5, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},

    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},

    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 20, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},

    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 40, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},

    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 100, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},

    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'relu'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'tanh'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'sigmoid'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'MultispikeNorm4'},
    #     {'layers': 200, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'Multispike4'},
    # ]

    plt.figure(figsize=(15, 8))
    fig = visualize_erf_fit(
        models_config=static_models_config,
        input_size=32,
        num_runs=50)
    plt.show()

    # save figure
    fig.savefig('visualization_gm.pdf')
    
    
