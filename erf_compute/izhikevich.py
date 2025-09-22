import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from spikingjelly.activation_based import base, surrogate, neuron

class IzhikevichNode(neuron.BaseNode):
    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0,
                 tau_inv: float = 250.0, sq: float = 0.04, mn: float = 5.0, bias: float = 140.0,
                 v_threshold: float = 30.0, v_reset: float = None,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode: str = 's',
                 backend: str = 'torch', store_v_seq: bool = False):
        """
        Izhikevich神经元模型
        
        参数:
        a: 控制恢复变量u时间尺度的参数 (默认: 0.02)
        b: 控制u对v变化敏感性的参数 (默认: 0.2)
        c: 脉冲后膜电压重置值 (默认: -65.0)
        d: 脉冲后恢复变量u的增量 (默认: 8.0)
        tau_inv: 时间常数倒数 (默认: 250.0)
        sq: 二次项系数 (默认: 0.04)
        mn: 线性项系数 (默认: 5.0)
        bias: 偏置项 (默认: 140.0)
        v_threshold: 阈值电压 (默认: 30.0)
        v_reset: 重置电压 (Izhikevich使用固定重置值c, 默认为None表示不使用重置)
        surrogate_function: 反向传播使用的替代函数 (默认: Sigmoid)
        detach_reset: 是否在反向传播时分离重置操作 (默认: False)
        step_mode: 步进模式 ('s' 单步, 'm' 多步) (默认: 's')
        backend: 后端实现 ('torch', 'cupy') (默认: 'torch')
        store_v_seq: 是否存储电压序列 (默认: False)
        """
        assert isinstance(a, float) and a != 0
        assert isinstance(tau_inv, float) and tau_inv > 0
        
        # v_reset 在Izhikevich神经元中是固定的c值，所以这里重写为None
        super().__init__(v_threshold, v_reset, surrogate_function, 
                         detach_reset, step_mode, backend, store_v_seq)
        
        # Izhikevich神经元参数
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tau_inv = tau_inv
        self.sq = sq
        self.mn = mn
        self.bias = bias
        
        # 初始化恢复变量u
        self.register_memory('u', None)
        
    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch',)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")
            
    def extra_repr(self):
        return (f"a={self.a}, b={self.b}, c={self.c}, d={self.d}, "
                f"tau_inv={self.tau_inv}, sq={self.sq}, mn={self.mn}, "
                f"bias={self.bias}, " + super().extra_repr())
    
    def neuronal_charge(self, x: torch.Tensor):
        """
        神经元充电过程 - Izhikevich模型
        """
        if self.u is None:
            # 初始化解状态
            self.u = self.b * self.v.clone()
        
        # 更新膜电位
        dv = self.tau_inv * (self.sq * self.v * self.v + self.mn * self.v + 
                             self.bias - self.u + x)
        self.v = self.v + dv
        
        # 更新恢复变量
        du = self.a * (self.b * self.v - self.u)
        self.u = self.u + du
    
    def neuronal_reset(self, spike):
        """
        神经元重置过程
        """
        super().neuronal_reset(spike)
        
        # Izhikevich特有的重置：脉冲后膜电位设为c，恢复变量u增加d
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        if self.v_reset is not None:
            # Izhikevich使用固定的重置值c
            self.v = self.c * spike_d + (1 - spike_d) * self.v
        
        # 恢复变量u增加d
        self.u = self.u + self.d * spike_d
    
    def reset(self):
        """
        重置神经元状态
        """
        super().reset()
        
        # 重置恢复变量u
        if self.u is not None:
            if isinstance(self.u, torch.Tensor):
                self.u.fill_(self.b * self.c)
            else:
                self.u = self.b * self.c
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward(
        x: torch.Tensor, v: torch.Tensor, u: torch.Tensor,
        a: float, b: float, c: float, d: float,
        tau_inv: float, sq: float, mn: float, bias: float,
        v_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步前向传播的JIT实现
        """
        # 更新膜电位
        dv = tau_inv * (sq * v * v + mn * v + bias - u + x)
        v = v + dv
        
        # 更新恢复变量
        du = a * (b * v - u)
        u = u + du
        
        # 判断是否发放脉冲
        spike = (v >= v_threshold).to(x)
        
        # 脉冲后重置
        v = c * spike + (1 - spike) * v
        u = u + d * spike
        
        return spike, v, u
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward(
        x_seq: torch.Tensor, v: torch.Tensor, u: torch.Tensor,
        a: float, b: float, c: float, d: float,
        tau_inv: float, sq: float, mn: float, bias: float,
        v_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        多步前向传播的JIT实现
        """
        T = x_seq.shape[0]
        spike_seq = torch.zeros_like(x_seq)
        
        for t in range(T):
            # 计算当前时间步
            dv = tau_inv * (sq * v * v + mn * v + bias - u + x_seq[t])
            v = v + dv
            du = a * (b * v - u)
            u = u + du
            
            # 发放判断
            spike = (v >= v_threshold).to(x_seq)
            v = c * spike + (1 - spike) * v
            u = u + d * spike
            
            spike_seq[t] = spike
        
        return spike_seq, v, u
    
    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            # 训练模式使用自动微分
            self.v_float_to_tensor(x)
            self.u_float_to_tensor(x)
            
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            
            return spike
        else:
            # 推理模式使用JIT优化
            if self.u is None:
                self.u = self.b * self.c * torch.ones_like(self.v)
                
            spike, self.v, self.u = self.jit_eval_single_step_forward(
                x, self.v, self.u,
                self.a, self.b, self.c, self.d,
                self.tau_inv, self.sq, self.mn, self.bias,
                self.v_threshold
            )
            return spike
    
    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            # 训练模式
            T = x_seq.shape[0]
            y_seq = []
            
            if self.store_v_seq:
                v_seq = []
            
            for t in range(T):
                y = self.single_step_forward(x_seq[t])
                y_seq.append(y)
                
                if self.store_v_seq:
                    if isinstance(self.v, torch.Tensor):
                        v_seq.append(self.v.clone())
                    else:
                        v_seq.append(self.v)
            
            if self.store_v_seq:
                self.v_seq = torch.stack(v_seq)
                
            return torch.stack(y_seq)
        else:
            # 推理模式使用JIT优化
            if self.u is None:
                self.u = self.b * self.c * torch.ones_like(self.v)
                
            spike_seq, self.v, self.u = self.jit_eval_multi_step_forward(
                x_seq, self.v, self.u,
                self.a, self.b, self.c, self.d,
                self.tau_inv, self.sq, self.mn, self.bias,
                self.v_threshold
            )
            
            if self.store_v_seq:
                # 未在JIT版本中存储v_seq，可以扩展如果需要
                pass
                
            return spike_seq
    
    def u_float_to_tensor(self, x: torch.Tensor):
        """
        确保u是Tensor并与输入x的数据类型和位置一致
        """
        if isinstance(self.u, float):
            self.u = torch.tensor(
                self.u, 
                dtype=x.dtype, 
                device=x.device
            )
        elif self.u is None:
            self.u = self.b * self.c * torch.ones_like(self.v)
    
    def reset_parameters(self):
        """
        重置神经元参数（如果需要）
        """
        # 可以添加参数重置逻辑
        super().reset()

def plot_v_u_trajectory(neuron, v_seq, u_seq, spike_seq):
    """绘制膜电位v和恢复变量u随时间变化的轨迹"""
    plt.figure(figsize=(14, 8))
    
    # 绘制膜电位v
    plt.subplot(3, 1, 1)
    plt.plot(v_seq, label='Membrane Potential (v)')
    plt.plot(np.where(spike_seq > 0)[0], 
             [v if v != 0 else None for v, s in zip(v_seq, spike_seq) if s > 0], 
             'ro', markersize=4, label='Spike')
    plt.axhline(y=neuron.v_threshold, color='r', linestyle='--', alpha=0.3)
    plt.ylabel('Membrane Potential')
    plt.title('Izhikevich Neuron Dynamics')
    plt.legend()
    
    # 绘制恢复变量u
    plt.subplot(3, 1, 2)
    plt.plot(u_seq, label='Recovery Variable (u)', color='orange')
    plt.ylabel('Recovery Variable')
    plt.legend()
    
    # 绘制输入电流
    plt.subplot(3, 1, 3)
    plt.plot(input_seq, label='Input Current', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Input Current')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('izhikevich_neuron_behavior.png', dpi=200)
    # plt.show()

def visualize_erf_temporal_with_shaded_area(erf_maps_list, titles, figsize=(12, 8), save_path=None, fit_curves=False, 
                                           fit_type='exp_x', show_equation=True, equation_pos='upper left'):
    """
    Create a line plot showing multiple ERF sequences over time, with shaded areas indicating variability, 
    showing the mean curve, and displaying fitted equations.
    
    主要改进：
    - 在图表上显示拟合曲线的数学公式
    - 提供公式位置的灵活选择
    - 改进图例显示和整体美观度
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
    import matplotlib.font_manager as fm
    
    # 设置专业科技论文风格的字体和样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    fit_results = {}
    
    # 定义专业的颜色方案
    colors = plt.cm.viridis(np.linspace(0, 1, len(erf_maps_list)))
    
    # 定义拟合函数
    def exp_x_func(x, a, b, c):
        return a * np.power(b, x) + c
    
    # 用于存储方程文本对象
    equation_texts = []
    
    for idx, (erf_maps, title) in enumerate(zip(erf_maps_list, titles)):
        color = colors[idx]
        alpha_value = 0.2 + 0.3 * (1 - idx/len(erf_maps_list))  # 不同曲线使用不同透明度
        time_steps = erf_maps[0].shape[0]
        
        # 计算所有试验的总和
        all_sums = []
        for erf_map in erf_maps:
            # 更稳健的ERF计算：考虑平均值而非总和
            total_per_time = np.array([np.mean(erf_map[t][erf_map[t] > 0]) if np.any(erf_map[t] > 0) else 0 
                                     for t in range(time_steps)])
            all_sums.append(total_per_time)
        
        all_sums = np.array(all_sums)
        mean_sums = np.mean(all_sums, axis=0)
        std_sums = np.std(all_sums, axis=0)
        time_axis = np.arange(time_steps)
        # 创建主曲线：均值线
        if len(erf_maps_list) > 1:
            # 多组数据时只显示均值线
            line_mean = ax.plot(time_axis, mean_sums, '-', color=color, linewidth=2.5, alpha=0.9, 
                               label=f"{title}", zorder=5)[0]
        else:
            # 单组数据时显示均值和标准差
            ax.fill_between(time_axis, mean_sums - std_sums, mean_sums + std_sums, 
                               color=color, alpha=alpha_value, label=f"{title} ± std")
            line_mean = ax.plot(time_axis, mean_sums, '-', color=color, linewidth=2.5, 
                                label=f"{title} mean", zorder=5)[0]
        equation = None
        
        # 曲线拟合和显示
        if fit_curves:
            try:
                diff = np.diff(mean_sums)
                monotonic_mask = diff > 0
                
                # 自动检测主要上升区间
                max_length = 0
                max_start = 0
                max_end = 0
                
                for start in np.where(monotonic_mask)[0]:
                    end = start
                    while end < len(diff) and monotonic_mask[end]:
                        end += 1
                    length = end - start
                    if length > max_length:
                        max_length, max_start, max_end = length, start, end
                
                if max_length >= 2:
                    fit_time_range = time_axis[max_start:min(max_end+1, len(mean_sums))]  # 避免越界
                    if len(fit_time_range) < 2:  # 确保至少有两个点
                        fit_time_range = time_axis
                        fit_mean_range = mean_sums
                    else:
                        fit_mean_range = mean_sums[fit_time_range]
                    
                    # 创建更平滑的曲线用于显示
                    x_smooth = np.linspace(fit_time_range[0], fit_time_range[-1], num=100)
                    
                    if fit_type == 'exp_x' and len(fit_time_range) > 2:
                        rel_fit_time = fit_time_range - fit_time_range[0]
                        rel_x_smooth = x_smooth - fit_time_range[0]
                        
                        # 使用加权最小二乘法更稳健的拟合
                        weights = np.linspace(1, 3, len(fit_mean_range))  # 给予后期数据更多权重
                        p0 = [0.1, 1.1, min(fit_mean_range)]  # 初始猜测
                        
                        popt, pcov = curve_fit(exp_x_func, rel_fit_time, fit_mean_range, 
                                              p0=p0, bounds=([0, 0.1, -np.inf], [np.inf, 10, np.inf]),
                                              maxfev=5000)
                        
                        a, b, c = popt
                        a = np.clip(a, 1e-6, None)  # 防止过小值
                        b = np.clip(b, 1.001, None)  # 防止过小值
                        
                        y_fit = exp_x_func(rel_x_smooth, popt)
                        
                        # 绘制拟合曲线（更细，不同样式）
                        ax.plot(x_smooth, y_fit, '--', color=color, linewidth=1.8, alpha=0.8,
                                zorder=3, label=f"{title} fit ({popt[1]:.3f}$^x$)")
                        
                        # 添加方程文本
                        if show_equation:
                            eq_text = f"$f(x) = {a:.3f} \\times {b:.3f}^{{x}} + {c:.3f}$"
                            
                            # 确定公式位置
                            y_pos = fit_mean_range[int(len(fit_mean_range)*0.7)]
                            text_obj = ax.text(x_smooth[90], y_pos, eq_text, 
                                              fontsize=14, color=color, fontweight='bold',
                                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                            
                            # 添加指向拟合曲线的箭头
                            ax.annotate('', xy=(x_smooth[80], y_fit[80]), 
                                        xytext=(x_smooth[90]+1.5, y_pos-0.05),
                                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.8))
                            
                            equation_texts.append(text_obj)
                            
                            fit_results[title] = {
                                'type': 'exp_x',
                                'equation': eq_text,
                                'parameters': popt,
                                'fit_range': (fit_time_range[0], fit_time_range[-1])
                            }
                
                else:
                    print(f"无适当单调上升区间用于 {title}")
                    
            except Exception as e:
                print(f"{title} 拟合失败: {str(e)}")
                fit_results[title] = {'error': str(e)}
    
    # 设置轴标签和标题
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Time Step', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('ERF Value', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title('Event Related Field over Time', fontsize=18, pad=15)
    
    # 创建紧凑图例在右上角
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # 只在存在图例项时添加
        ax.legend(handles, labels, fontsize=12, loc='upper right', 
                  frameon=True, framealpha=0.9, shadow=True)
    
    # 添加网格和美化
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # 自动调整坐标轴范围
    all_data = np.concatenate([np.array([np.mean(erf_map[t]) for erf_map in erf_maps_list[0] for t in range(erf_map.shape[0])])
                               for erf_map_list in erf_maps_list])
    y_range = np.nanmin(all_data), np.nanmax(all_data) * 1.1
    ax.set_ylim(y_range)
    
    # 保存高质量图像
    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight', transparent=False)
    
    # 添加版权注释
    ax.text(0.98, 0.02, '© Neuroscience Analysis Tool v1.0', 
           transform=ax.transAxes, fontsize=10, ha='right', va='bottom', alpha=0.7)
    
    plt.tight_layout()
    
    return fig, fit_results, equation_texts

if __name__ == "__main__":
    neuron = IzhikevichNode(
        a=0.02, b=0.2, c=-65.0, d=6.0, tau_inv=0.002, v_threshold=2.3,
        surrogate_function=surrogate.ATan(alpha=2.0),
        store_v_seq=True
    )
    erf_map_list = []
    trial_steps = 60
    for _ in range(trial_steps):
        input_seq = torch.randn(24)  # 模拟输入电流
        input_seq.requires_grad = True

        # 前向传播
        spike_seq = neuron.multi_step_forward(input_seq)
        print(f"Spike Sequence: {spike_seq}")
        grad_output = torch.zeros_like(spike_seq)
        grad_output[-1] = 1.0
        spike_seq.backward(gradient=grad_output, retain_graph=True)
        gradient = input_seq.grad.detach()
        erf_map = gradient.abs().squeeze().cpu().numpy()
        erf_map_list.append(erf_map)

        # Clear computation graph
        input_seq.grad = None

    
    def safe_normalize(arr):
        arr_max = np.max(arr)
        if arr_max > 0:
            return arr / arr_max
        return arr

    erf_map = safe_normalize(erf_map)
    print(f"ERF Map: {erf_map}")
    # # 定义拟合函数 - 尝试多种形式
    def exp_decay(x, a, b, c):
        """指数衰减拟合"""
        return a * np.exp(-b * x) + c
    
    visualize_erf_temporal_with_shaded_area(
        [erf_map_list],
        titles=['ERF Map'],
        figsize=(12, 8),
        save_path='erf_map_plot.png',
        fit_curves=True,
        fit_type='exp_x'
    )
    # def polynomial(x, a, b, c):
    #     """二次多项式拟合"""
    #     return a*x**2 + b*x + c
        
    
    # def gaussian(x, a, b, c):
    #     """高斯函数拟合"""
    #     return a * np.exp(-(x-b)**2/(2*c**2))
    
    # # 1. 尝试指数衰减拟合
    # try:
    #     popt_exp, pcov_exp = curve_fit(exp_decay, time_steps, grad_data, p0=[0.1, 0.1, 0.1])
    #     exp_fit = exp_decay(time_steps, *popt_exp)
    #     exp_r_squared = 1 - np.sum((grad_data - exp_fit)**2) / np.sum((grad_data - np.mean(grad_data))**2)
    # except:
    #     exp_fit = None
    
    # # 2. 尝试多项式拟合
    # try:
    #     popt_poly, pcov_poly = curve_fit(polynomial, time_steps, grad_data, p0=[0, 0, 0])
    #     poly_fit = polynomial(time_steps, *popt_poly)
    #     poly_r_squared = 1 - np.sum((grad_data - poly_fit)**2) / np.sum((grad_data - np.mean(grad_data))**2)
    # except:
    #     poly_fit = None

    # best_fit = None
    # best_eq = None
    # best_r_squared = -np.inf
    
    # if exp_fit is not None and exp_r_squared > best_r_squared:
    #     best_fit = exp_fit
    #     best_r_squared = exp_r_squared
    #     best_eq = r"${:.4f} e^{{{:.4f} x}} + {:.4f}$".format(popt_exp[0], -popt_exp[1], popt_exp[2])
    
    # if poly_fit is not None and poly_r_squared > best_r_squared:
    #     best_fit = poly_fit
    #     best_r_squared = poly_r_squared
    #     best_eq = r"${:.4f}x^2 + {:.4f}x + {:.4f}$".format(popt_poly[0], popt_poly[1], popt_poly[2])
    
    # # 绘制结果
    # plt.figure(figsize=(12, 8))
    
    # # 原始梯度数据
    # plt.plot(time_steps, grad_data, 'bo-', label='original', linewidth=2, markersize=8)
    
    # # 绘制所有拟合曲线（透明度较低）
    # legends = ['original']
    
    # if exp_fit is not None:
    #     plt.plot(time_steps, exp_fit, 'r--', alpha=0.4, linewidth=1.5)
    #     legends.append(f'exponential fit (R²={exp_r_squared:.3f})')
    
    # if poly_fit is not None:
    #     plt.plot(time_steps, poly_fit, 'g--', alpha=0.4, linewidth=1.5)
    #     legends.append(f'polynomial fit (R²={poly_r_squared:.3f})')
    
    
    # # 突出显示最佳拟合
    # if best_fit is not None and best_r_squared > 0.5:  # 仅当拟合较好时显示
    #     plt.plot(time_steps, best_fit, 'k-', label='best fit', linewidth=2.5)
    #     legends.append('best fit')
    
    # plt.title('', fontsize=15)
    # plt.xlabel('T', fontsize=13)
    # plt.ylabel('Grad', fontsize=13)
    # plt.grid(True, alpha=0.3)
    
    # # 添加图例
    # plt.legend(legends, fontsize=10)
    
    # # 添加拟合方程信息
    # if best_fit is not None and best_r_squared > 0.5:
    #     plt.text(0.05, 0.95, f"Bset fitting: {best_eq}\n (R²) = {best_r_squared:.4f}", 
    #              transform=plt.gca().transAxes, fontsize=12,
    #              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # plt.tight_layout()
    # plt.savefig('gradient_distribution_with_fit.png', dpi=200, bbox_inches='tight')
    # plt.show()
