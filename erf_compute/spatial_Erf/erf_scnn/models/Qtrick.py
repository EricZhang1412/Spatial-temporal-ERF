import torch
import torch.nn as nn
import math

quant = 4
T = quant


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, i, min_value=0, max_value=quant):
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


class MultiSpike_norm4(nn.Module):
    def __init__(
            self,
            Vth=1.0,
            T=T,
    ):
        super().__init__()
        self.spike = Quant()
        self.Vth = Vth
        self.T = T

    def forward(self, x):
        return self.spike.apply(x) / self.T

class MultiSpike_4(nn.Module):
    def __init__(
            self,
            T=T, 
    ):
        super().__init__()
        self.spike = Quant()
        self.T = T

    def forward(self, x):
        return self.spike.apply(x)

class SurrogateFunction:
    @staticmethod
    def triangle(x, alpha=2.0):
        return torch.where(abs(x) <= 1.0, alpha * (1.0 - abs(x)), torch.zeros_like(x))
    
    @staticmethod
    def sigmoid(x, alpha=4.0):
        return torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))
    
    @staticmethod
    def arctan(x, alpha=2.0):
        return alpha / (1.0 + (math.pi / 2 * alpha * x).pow_(2))

    @staticmethod
    def rectangle(x, width=1.0):
        return torch.where(abs(x) <= width, torch.ones_like(x), torch.zeros_like(x))

class QuantWithSurrogate(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, i, min_value=0, max_value=quant, grad_mode='triangle', alpha=2.0):
        ctx.min = min_value
        ctx.max = max_value
        ctx.grad_mode = grad_mode
        ctx.alpha = alpha
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        
        if ctx.grad_mode == 'triangle':
            grad_scale = SurrogateFunction.triangle(i - ctx.min, ctx.alpha)
        elif ctx.grad_mode == 'sigmoid':
            grad_scale = SurrogateFunction.sigmoid(i - ctx.min, ctx.alpha)
        elif ctx.grad_mode == 'arctan':
            grad_scale = SurrogateFunction.arctan(i - ctx.min, ctx.alpha)
        elif ctx.grad_mode == 'rectangle':
            grad_scale = SurrogateFunction.rectangle(i - ctx.min, ctx.alpha)
        else:
            # 默认使用原始的硬截断
            grad_input[i < ctx.min] = 0
            grad_input[i > ctx.max] = 0
            return grad_input, None, None, None, None

        return grad_input * grad_scale, None, None, None, None

class LIF(nn.Module):
    def __init__(
            self,
            tau=2.0,
            Vth=1.0,
            detach_reset=True,
            surrogate_mode='triangle',  
            alpha=2.0  
    ):
        super().__init__()
        self.tau = tau
        self.Vth = Vth
        self.detach_reset = detach_reset
        self.spike = QuantWithSurrogate
        self.surrogate_mode = surrogate_mode
        self.alpha = alpha

        self.T = T
        
    def forward(self, x):
        # x shape: (T, Batch, Channel, Height, Width)
        device = x.device
        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
            x = x.unsqueeze(0)
            x = x.expand(self.T, -1, -1, -1, -1)
        else:
            T, batch_size, channels, height, width = x.shape
        
        mem = torch.zeros(batch_size, channels, height, width, device=device)
        spikes = []
        
        for t in range(self.T):
            mem = mem / self.tau
            mem = mem + x[t]
            
            spike = self.spike.apply(
                mem / self.Vth, 
                0, 1,  # min_value, max_value
                self.surrogate_mode,
                self.alpha
            )
            spikes.append(spike)
            
            if self.detach_reset:
                mem = mem - (spike * self.Vth).detach()
            else:
                mem = mem - spike * self.Vth
        
        return torch.stack(spikes, dim=0)

class LIF_with_Temporal_Gradient(nn.Module):
    def __init__(
            self,
            tau=2.0,
            Vth=1.0,
            detach_reset=True,
            surrogate_mode='triangle',
            alpha=2.0
    ):
        super().__init__()
        self.tau = tau
        self.Vth = Vth
        self.detach_reset = detach_reset
        self.spike = QuantWithSurrogate
        self.surrogate_mode = surrogate_mode
        self.alpha = alpha
        self.T = T
        
    def forward(self, x):
        device = x.device

        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
            x = x.unsqueeze(0)
            x = x.expand(self.T, -1, -1, -1, -1)
        else:
            T, batch_size, channels, height, width = x.shape

        v_history = torch.zeros(self.T, batch_size, channels, height, width, device=device)
        spike_history = torch.zeros(self.T, batch_size, channels, height, width, device=device)

        mem = torch.zeros(batch_size, channels, height, width, device=device)

        for t in range(self.T):
            mem = mem / self.tau
            mem = mem + x[t]

            spike = self.spike.apply(
                mem / self.Vth,
                0, 1,  # min_value, max_value
                self.surrogate_mode,
                self.alpha
            )

            if self.detach_reset:
                mem = mem - (spike * self.Vth).detach()
            else:
                mem = mem - spike * self.Vth

            v_history[t] = mem.clone()
            spike_history[t] = spike.clone()
        
        return spike_history, v_history