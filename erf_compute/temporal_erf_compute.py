import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase

class Quant(torch.autograd.Function):
    @staticmethod
    # @torch.cuda.amp.custom_fwd
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None
    
class MultiSpike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x): # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)

@torch.jit.script
def heaviside(x: torch.Tensor):
    '''
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: 输入tensor
    :return: 输出tensor

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    * :ref:`中文API <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    '''
    return (x >= 0).to(x)

@torch.jit.script
def rect_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * (x.abs() < 0.5 / alpha).to(x) * grad_output, None


class rect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return rect_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Rect(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rect.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.clamp(alpha * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def backward(grad_output, x, alpha):
        return rect_backward(grad_output, x, alpha)[0]

@torch.jit.script
def polynomial_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    """
    Backward function for the polynomial surrogate function
    
    According to the table:
    h'(x) = {
        0,                |x| > 1/alpha
        -alpha^2 |x| + alpha,  |x| <= 1/alpha
    }
    """
    mask_small = (x.abs() <= 1.0 / alpha)
    # Compute gradient based on the formula: -alpha^2 |x| + alpha when |x| <= 1/alpha
    abs_x = x.abs()
    grad = torch.zeros_like(x)
    grad[mask_small] = -alpha * alpha * abs_x[mask_small] + alpha
    
    return grad * grad_output, None


class polynomial(torch.autograd.Function):
    """
    Polynomial surrogate function
    
    Forward: Heaviside step function
    Backward: Defined by polynomial_backward
    """
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return polynomial_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Polynomial(nn.Module):
    """
    Polynomial surrogate gradient function
    
    h(x) = {
        0,                           x < -1/alpha
        -0.5*alpha^2*x|x| + alpha*x + 0.5,  |x| <= 1/alpha
        1,                           x > 1/alpha
    }
    """
    def __init__(self, alpha=1.0):
        super(Polynomial, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return polynomial.apply(x, self.alpha)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        """
        Primitive function (antiderivative) for the polynomial surrogate
        
        h(x) = {
            0,                           x < -1/alpha
            -0.5*alpha^2*x|x| + alpha*x + 0.5,  |x| <= 1/alpha
            1,                           x > 1/alpha
        }
        """
        result = torch.zeros_like(x)
        
        # Case: x < -1/alpha
        mask_neg = (x < -1.0 / alpha)
        result[mask_neg] = 0.0
        
        # Case: |x| <= 1/alpha
        mask_mid = (x.abs() <= 1.0 / alpha)
        abs_x = x.abs()
        result[mask_mid] = -0.5 * alpha * alpha * x[mask_mid] * abs_x[mask_mid] + alpha * x[mask_mid] + 0.5
        
        # Case: x > 1/alpha
        mask_pos = (x > 1.0 / alpha)
        result[mask_pos] = 1.0
        
        return result
    
    @staticmethod
    def backward(grad_output, x, alpha):
        return polynomial_backward(grad_output, x, alpha)[0]
    
class ERFNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 num_layers=20, 
                 kernel_size=3, 
                 activation='relu',
                 surrogate_hyperparameters=2.0,
                 tau=2.0,
                 use_bn=False):
        super(ERFNet, self).__init__()
        self.T = 16  
        
        if activation == 'ifnode':
            act_fn = neuron.IFNode(surrogate_function=surrogate.ATan(alpha=surrogate_hyperparameters), step_mode='m', v_threshold=50.0)
        elif activation == 'lifnode':
            act_fn = neuron.LIFNode(surrogate_function=surrogate.ATan(alpha=surrogate_hyperparameters), tau=tau, step_mode='m', v_threshold=1.0)
            # act_fn = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(alpha=surrogate_hyperparameters), tau=tau, step_mode='m', v_threshold=1.0)
            # act_fn = neuron.LIFNode(surrogate_function=Rect(alpha=surrogate_hyperparameters), tau=tau, step_mode='m', v_threshold=1.0)
            # act_fn = neuron.LIFNode(surrogate_function=Polynomial(alpha=surrogate_hyperparameters), tau=tau, step_mode='m', v_threshold=1.0)
        elif activation == 'ilif':
            act_fn = MultiSpike(min_value=0, max_value=self.T, Norm=1.0)
        elif activation == 'izh':
            act_fn = neuron.IzhikevichNode(surrogate_function=surrogate.ATan(alpha=surrogate_hyperparameters), tau=tau, step_mode='m', v_threshold=5.0)
        elif activation == 'plif':
            act_fn = neuron.ParametricLIFNode(
                surrogate_function=surrogate.ATan(alpha=surrogate_hyperparameters), 
                step_mode='m', 
                v_threshold=1.0
            )
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        # 构建网络层
        layers = []
        
        # 第一层
        hidden_channels = 1  
        padding = kernel_size // 2 
        
        layers.append(
            layer.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding)
        )
        layers.append(act_fn)
        
        # 中间层
        for _ in range(num_layers - 1):
            if use_bn:
                layers.append(layer.BatchNorm2d(hidden_channels))
            layers.append(
                layer.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
            )
            layers.append(act_fn)
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.network(x)
        return x
    
    def initialize_weights(self, mode="gaussian", mean=0.0, std=0.01):
        if mode == "gaussian":
            for m in self.modules():
                if isinstance(m, layer.Conv2d):
                    nn.init.normal_(m.weight, mean=mean, std=std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        if mode == "uniform":
            for m in self.modules():
                if isinstance(m, layer.Conv2d):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


def safe_normalize(erf_map):
    erf_map = np.nan_to_num(erf_map, nan=0.0, posinf=1.0, neginf=0.0)
    
    max_val = np.max(erf_map)
    if np.isfinite(max_val) and max_val > 0:
        erf_map = erf_map / max_val
    else:
        erf_map = np.zeros_like(erf_map)
    
    return erf_map

def randn_range(shape, a, b, device=None):
    x = torch.randn(shape, device=device)
    return (b - a) * x + a

def compute_erf(model, input_size=(1, 64, 64), center_pos=None, device="cuda", tau=1):
    """
        Calculate the effective receptive field

    Parameter:
        - model: the trained model
        - input_size: Enter image size (C, H, W)
        - center_pos: The center point position (y, x), which defaults to the center of the image
        - device: computing device
        - temporal: Whether to calculate the ERF of the time dimension, which is set to False by default (calculates the spatial ERF)
        - tau: The time interval parameter τ, which is used to calculate the time ERF

    Return:
        - erf_map: Active Sensory Wild Diagram (normalized to [0,1])
    """
    T = 16
    model.to(device)

    if center_pos is None:
        center_pos = (input_size[1] // 2, input_size[2] // 2)
    
    input_tensor = torch.randn((1,) + input_size, device=device)
    input_seq = input_tensor.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    input_seq.requires_grad_(True)

    def create_grad_monitors(model):
        spike_grad_monitor = None
        input_grad_monitor = None

        for module in model.modules():
            if isinstance(module, neuron.IFNode):
                spike_grad_monitor = monitor.GradOutputMonitor(model, neuron.IFNode)
                input_grad_monitor = monitor.GradInputMonitor(model, neuron.IFNode)
                break
            elif isinstance(module, neuron.LIFNode):
                spike_grad_monitor = monitor.GradOutputMonitor(model, neuron.LIFNode)
                input_grad_monitor = monitor.GradInputMonitor(model, neuron.LIFNode)
                break
            elif isinstance(module, neuron.ParametricLIFNode):
                spike_grad_monitor = monitor.GradOutputMonitor(model, neuron.ParametricLIFNode)
                input_grad_monitor = monitor.GradInputMonitor(model, neuron.ParametricLIFNode)
                break
            elif isinstance(module, neuron.IzhikevichNode):
                spike_grad_monitor = monitor.GradOutputMonitor(model, neuron.IzhikevichNode)
                input_grad_monitor = monitor.GradInputMonitor(model, neuron.IzhikevichNode)
                break

        return spike_grad_monitor, input_grad_monitor

    spike_seq_grad_monitor, input_seq_monitor = create_grad_monitors(model)

    functional.set_step_mode(model, step_mode='m')
    
    functional.reset_net(model)
    output_seq = model(input_seq)
    fr = output_seq.mean()

    grad_output = torch.zeros_like(output_seq, device=device)
    grad_output[-1, :, :, :, :] = 1.0

    output_seq.backward(gradient=grad_output)

    gradient = input_seq.grad.detach()
    
    erf_map = gradient.abs().sum(dim=(1,2)).squeeze().cpu().numpy()

    def safe_normalize(arr):
        arr_max = np.max(arr)
        if arr_max > 0:
            return arr / arr_max
        return arr
    
    spike_seq_grad_monitor.records.clear()
    input_seq_monitor.records.clear()

    erf_map = safe_normalize(erf_map)
    return erf_map


def gaussian(x, amplitude, mean, sigma):
    """Gaussian function for curve fitting"""
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

def visualize_erf_comparison(erf_maps, titles, time_indices=None, figsize=(45, 15), save_path=None):
    """
    Visualize multiple ERF maps with temporal dimension (T,H,W) with separate horizontal and vertical profile curves showing Gaussian fits.
    
    Parameters:
    -----------
    erf_maps : list of 3D arrays (T,H,W)
        The ERF maps to visualize, each with temporal dimension
    titles : list of str
        Titles for each ERF map
    time_indices : list of int, optional
        Specific time indices to visualize. If None, visualizes all time points
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str or list of str, optional
        Path(s) to save the figure(s). If a string, '_t{time_index}' will be appended before the file extension
        
    Returns:
    --------
    figs : list of matplotlib.figure.Figure
        The figure objects
    """

    n = len(erf_maps)
    figs = []
    
    # Check if all erf_maps have the same temporal dimension
    time_dims = [erf_map.shape[0] for erf_map in erf_maps]
    
    # Use specified time indices or all available
    if time_indices is None:
        time_indices = list(range(max(time_dims)))
    
    # Process each time point
    for t_idx in time_indices:
        # Skip if t_idx is out of range for any map
        if any(t_idx >= t_dim for t_dim in time_dims):
            continue
            
        # Create a new figure for this time point
        # Modified layout with 3 rows:
        # - Top row: ERF maps
        # - Middle row: Horizontal profiles
        # - Bottom row: Vertical profiles
        width_ratios = [2] * n + [0.05]
        
        # Create a GridSpec layout with 3 rows
        gs = gridspec.GridSpec(3, n+1, 
                              width_ratios=width_ratios, 
                              height_ratios=[2, 1, 1],  # ERF maps take more vertical space than profiles
                              wspace=0.3, 
                              hspace=0.4)
        
        fig = plt.figure(figsize=figsize)
        ims = []
        
        # Draw each ERF map at the current time point in the top row
        for i, (erf_map, title) in enumerate(zip(erf_maps, titles)):
            # Extract the 2D map at the current time point
            current_map = erf_map[t_idx]
            
            ax_erf = fig.add_subplot(gs[0, i])
            im = ax_erf.imshow(current_map, cmap='gray')
            # Add time index to the title
            ax_erf.set_title(f"{title} (t={t_idx})")
            ax_erf.axis('off')
            ims.append(im)
            
            # Extract horizontal and vertical profiles through the center
            h, w = current_map.shape
            center_h = h // 2
            center_w = w // 2
            
            # Get horizontal and vertical profiles
            h_profile = current_map[center_h, :]
            v_profile = current_map[:, center_w]
            
            # Create x-axis for profiles
            h_x = np.arange(w)
            v_x = np.arange(h)
            
            # Create horizontal profile subplot in the middle row
            ax_h_profile = fig.add_subplot(gs[1, i])
            
            # Create vertical profile subplot in the bottom row
            ax_v_profile = fig.add_subplot(gs[2, i])
            
            # Fit Gaussian to horizontal profile
            try:
                # Initial parameter guesses: [amplitude, mean, sigma]
                p0 = [np.max(h_profile), center_w, w/10]
                h_params, _ = curve_fit(gaussian, h_x, h_profile, p0=p0)
                h_mean, h_sigma = h_params[1], h_params[2]
                
                # Create horizontal fitted curve
                h_fit = gaussian(h_x, *h_params)
                
                # Plot horizontal profile and fit
                ax_h_profile.plot(h_x, h_profile, 'b-', label='Profile')
                ax_h_profile.plot(h_x, h_fit, 'r--', alpha=0.7, label='Gaussian Fit')
                
                # Add text with mean and variance (σ²)
                ax_h_profile.text(0.05, 0.95, f'μ={h_mean:.1f}, σ²={h_sigma**2:.1f}', 
                             transform=ax_h_profile.transAxes, 
                             verticalalignment='top', 
                             color='black',
                             fontsize=10)
                
                ax_h_profile.set_title('Horizontal Profile')
                ax_h_profile.legend(loc='upper right', fontsize='small')
                ax_h_profile.grid(True, alpha=0.3)
                
                # Fit Gaussian to vertical profile
                p0 = [np.max(v_profile), center_h, h/10]
                v_params, _ = curve_fit(gaussian, v_x, v_profile, p0=p0)
                v_mean, v_sigma = v_params[1], v_params[2]
                
                # Create vertical fitted curve
                v_fit = gaussian(v_x, *v_params)
                
                # Plot vertical profile and fit
                ax_v_profile.plot(v_x, v_profile, 'b-', label='Profile')
                ax_v_profile.plot(v_x, v_fit, 'r--', alpha=0.7, label='Gaussian Fit')
                
                # Add text with mean and variance (σ²)
                ax_v_profile.text(0.05, 0.95, f'μ={v_mean:.1f}, σ²={v_sigma**2:.1f}', 
                             transform=ax_v_profile.transAxes, 
                             verticalalignment='top', 
                             color='black',
                             fontsize=10)
                
                ax_v_profile.set_title('Vertical Profile')
                ax_v_profile.legend(loc='upper right', fontsize='small')
                ax_v_profile.grid(True, alpha=0.3)
                
            except:
                # If fitting fails, just plot the profiles
                ax_h_profile.plot(h_x, h_profile, 'b-', label='Profile')
                ax_h_profile.text(0.05, 0.95, 'Gaussian fit failed', 
                             transform=ax_h_profile.transAxes, 
                             verticalalignment='top')
                ax_h_profile.set_title('Horizontal Profile')
                ax_h_profile.grid(True, alpha=0.3)
                
                ax_v_profile.plot(v_x, v_profile, 'b-', label='Profile')
                ax_v_profile.text(0.05, 0.95, 'Gaussian fit failed', 
                             transform=ax_v_profile.transAxes, 
                             verticalalignment='top')
                ax_v_profile.set_title('Vertical Profile')
                ax_v_profile.grid(True, alpha=0.3)
            
        # Add colorbar in the last column of the top row
        cbar_ax = fig.add_subplot(gs[0, n])
        fig.colorbar(ims[0], cax=cbar_ax)
        
        # Add a main title showing the time index
        fig.suptitle(f"Time Index: {t_idx}", fontsize=16)
        
        # Save the figure if requested
        if save_path:
            if isinstance(save_path, list) and len(save_path) > t_idx:
                plt.savefig(save_path[t_idx], dpi=800, bbox_inches='tight')
            elif isinstance(save_path, str):
                # Insert time index before file extension
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, '')
                save_file = f"{base}_t{t_idx}.{ext}" if ext else f"{base}_t{t_idx}"
                plt.savefig(save_file, dpi=800, bbox_inches='tight')
        
        plt.tight_layout()
        rect = [0, 0.03, 1, 0.95]  # [left, bottom, right, top]
        plt.subplots_adjust(top=rect[3])
        
        figs.append(fig)
    
    return figs


def visualize_erf_temporal_with_shaded_area(erf_maps_list, titles, figsize=(12, 8), save_path=None, fit_curves=False, fit_type='exp_x'):
    """
    Create a line plot showing multiple ERF sequences over time, with shaded areas indicating variability, and showing the mean curve.
    Optionally, curve fitting is performed for monotonically increasing intervals.

    Parameter:
    -----------
    erf_maps_list : A list of lists
    Each internal list contains an ERF map for a set of experiments (each ERF map is shaped [T,H,W])
    titles: A list of strings
    The title of each ERF group
    figsize : tuple, optional
    Image size (width, height)
    save_path : String, optional
    The path to save the image
    fit_curves : Boolean, optional
    Whether or not to fit the monotonic increase interval of the mean curve
    fit_type : String, optional
    The type of curve fitted

    Return:
    --------
    fig : matplotlib.figure.Figure
    Image objects
    fit_results : Dictionary
    A dictionary containing fitting parameters for each ERF group
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
    import matplotlib.ticker as ticker
    
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'  

    fig, ax = plt.subplots(figsize=figsize)
    fit_results = {}
    
    # 定义颜色列表，确保每组有不同的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 定义拟合函数
    def exp_x_func(x, a, b, c):
        # y = a * b^x + c 
        return a * np.power(b, x) + c
    
    for idx, (erf_maps, title) in enumerate(zip(erf_maps_list, titles)):
        color = colors[idx % len(colors)]
        alpha_value = 0.2

        time_steps = erf_maps[0].shape[0]

        all_sums = []
        for erf_map in erf_maps:
            total_sum_per_time = np.array([np.sum(erf_map[t]) for t in range(time_steps)])
            all_sums.append(total_sum_per_time)
        
        all_sums = np.array(all_sums)
        mean_sums = np.mean(all_sums, axis=0)
        std_sums = np.std(all_sums, axis=0)

        time_axis = np.arange(time_steps)

        ax.fill_between(time_axis, mean_sums - std_sums, mean_sums + std_sums, 
                        color=color, alpha=alpha_value, label=f"{title} (pm std)")
        
        ax.plot(time_axis, mean_sums, '-', color=color, linewidth=2, label=f"{title} (mean)")
        

        if fit_curves:
            try:
                diff = np.diff(mean_sums)
                monotonic_mask = diff > 0

                max_length = 0
                max_start = 0
                max_end = 0
                current_start = 0
                current_length = 0
                
                for i, is_increasing in enumerate(monotonic_mask):
                    if is_increasing:
                        if current_length == 0:
                            current_start = i
                        current_length += 1
                    else:
                        if current_length > max_length:
                            max_length = current_length
                            max_start = current_start
                            max_end = current_start + current_length
                        current_length = 0

                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = current_start + current_length

                if max_length >= 2:

                    fit_time_range = time_axis[max_start:max_end+1]
                    fit_mean_range = mean_sums[max_start:max_end+1]

                    x_smooth = np.linspace(fit_time_range[0], fit_time_range[-1], num=100)
                    
                    if fit_type == 'exp_x':
                        rel_fit_time = fit_time_range - fit_time_range[0]
                        
                        p0 = [1, 1.1, min(fit_mean_range)]  # init a=1, b=1.1, c=min value
                        popt, pcov = curve_fit(exp_x_func, rel_fit_time, fit_mean_range, p0=p0, maxfev=10000)

                        rel_x_smooth = x_smooth - fit_time_range[0]
                        y_fit = exp_x_func(rel_x_smooth, *popt)

                        ax.plot(x_smooth, y_fit, '--', color=color, linewidth=1.5, 
                                label=f"{title} fit (t={fit_time_range[0]}-{fit_time_range[-1]}): {popt[0]:.4f} * {popt[1]:.4f}^x + {popt[2]:.4f}")
                        
                        fit_results[title] = {
                            'type': 'exp_x',
                            'fit_range': (fit_time_range[0], fit_time_range[-1]),
                            'equation': f'{popt[0]:.4f} * {popt[1]:.4f}^x + {popt[2]:.4f}',
                            'parameters': popt,
                            'covariance': pcov
                        }
                    else:
                        print(f"Warning: Currently, only the 'exp_x' fit type is supported. Skip other types of curve fittings. ")
                else:
                    print(f"Warning: {title} The monotonic increase interval is too short to fit in")
                    fit_results[title] = {'error': 'The monotonic increase interval is too short to fit in'}
            
            except Exception as e:
                print(f"{title} Failed fitting: {str(e)}")
                fit_results[title] = {'error': str(e)}
    ax.grid(True, linestyle='--', alpha=0.7)

    max_time_step = max([erf_maps[0].shape[0] for erf_maps in erf_maps_list])
    ax.set_xticks(np.arange(0, max_time_step, 1)) 

    ax.tick_params(axis='both', which='major', labelsize=20)
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if "mean" in l or "fit:" in l or "±std" in l:
            filtered_handles.append(h)
            filtered_labels.append(l)

    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if "mean" in l or "fit:" in l or "±std" in l:
            filtered_handles.append(h)
            filtered_labels.append(l)
    

    ax.legend(filtered_handles, filtered_labels, fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, fit_results

def main():
    # Optional: Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    input_size = (1, 64, 64)  
    center_pos = (input_size[1] // 2, input_size[2] // 2)
    
    num_layers = 5  
    N_runs = 20      
    
    lifnode_net_tau8_list    = []
    lifnode_net_tau2_list    = []
    lifnode_net_tau4_list    = []
    lifnode_net_tau6_list    = []
    
    
    for run in range(N_runs):
        print(f"Run {run+1}/{N_runs}")
        lifnode_net_tau8 = ERFNet(activation='plif', num_layers=num_layers, in_channels=input_size[0], surrogate_hyperparameters=0.01, tau=8.0)
        lifnode_net_tau2 = ERFNet(activation='plif', num_layers=num_layers, in_channels=input_size[0], surrogate_hyperparameters=0.01, tau=2.0)
        lifnode_net_tau4 = ERFNet(activation='plif', num_layers=num_layers, in_channels=input_size[0], surrogate_hyperparameters=0.01, tau=4.0)
        lifnode_net_tau6 = ERFNet(activation='plif', num_layers=num_layers, in_channels=input_size[0], surrogate_hyperparameters=0.01 , tau=6.0)

        lifnode_net_tau8.initialize_weights("gaussian", mean=0.0, std=0.05)
        lifnode_net_tau2.initialize_weights("gaussian", mean=0.0, std=0.05)
        lifnode_net_tau4.initialize_weights("gaussian", mean=0.0, std=0.05)
        lifnode_net_tau6.initialize_weights("gaussian", mean=0.0, std=0.05)
        
        
        # ifnode_erfs    = compute_erf(ifnode_net, input_size=input_size, center_pos=center_pos, device=device)
        lifnode_erfs_tau1    = compute_erf(lifnode_net_tau8, input_size=input_size, center_pos=center_pos, device=device)
        lifnode_erfs_tau2    = compute_erf(lifnode_net_tau2, input_size=input_size, center_pos=center_pos, device=device)
        lifnode_erfs_tau4    = compute_erf(lifnode_net_tau4, input_size=input_size, center_pos=center_pos, device=device)
        lifnode_erfs_tau6    = compute_erf(lifnode_net_tau6, input_size=input_size, center_pos=center_pos, device=device)

        lifnode_net_tau8_list.append(lifnode_erfs_tau1)
        lifnode_net_tau2_list.append(lifnode_erfs_tau2)
        lifnode_net_tau4_list.append(lifnode_erfs_tau4)
        lifnode_net_tau6_list.append(lifnode_erfs_tau6)

    lifnode_erfs_avg_tau1    = np.mean(lifnode_net_tau8_list, axis=0)
    lifnode_erfs_avg_tau2    = np.mean(lifnode_net_tau2_list, axis=0)
    lifnode_erfs_avg_tau4    = np.mean(lifnode_net_tau4_list, axis=0)
    lifnode_erfs_avg_tau6    = np.mean(lifnode_net_tau6_list, axis=0)  

    erf_maps = [lifnode_erfs_avg_tau1, lifnode_erfs_avg_tau2, lifnode_erfs_avg_tau4, lifnode_erfs_avg_tau6]
    erf_maps_lists = [lifnode_net_tau8_list, lifnode_net_tau2_list, lifnode_net_tau4_list, lifnode_net_tau6_list]
    titles   = ['LIFNode (tau=8)', 'LIFNode (tau=2)', 'LIFNode (tau=4)', 'LIFNode (tau=6)']

    fig1, fit_results = visualize_erf_temporal_with_shaded_area(
        erf_maps_lists, 
        titles, 
        figsize=(14, 8),
        save_path='erf_t_comparison_with_distribution.png', 
        fit_curves=True, 
        fit_type='exp_x'
    )
    print("\nFitting Results:")
    for title, res in fit_results.items():
        print(f"{title}: {res.get('equation', 'Fitting failed')}")
    
    plt.close(fig1)


if __name__ == "__main__":
    main()