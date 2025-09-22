import torch
import pytest

import torch.nn as nn
import torch.nn.functional as F
from norse.torch.module.izhikevich import (
    IzhikevichCell,
    IzhikevichRecurrentCell,
    Izhikevich,
    IzhikevichRecurrent,
)
from norse.torch.functional import izhikevich
from norse.torch.functional.izhikevich import IzhikevichSpikingBehavior


# list_method = [
#     izhikevich.tonic_spiking,
#     izhikevich.phasic_spiking,
#     izhikevich.tonic_bursting,
#     izhikevich.phasic_bursting,
#     izhikevich.mixed_mode,
#     izhikevich.spike_frequency_adaptation,
#     izhikevich.class_1_exc,
#     izhikevich.class_2_exc,
#     izhikevich.spike_latency,
#     izhikevich.subthreshold_oscillation,
#     izhikevich.resonator,
#     izhikevich.integrator,
#     izhikevich.rebound_spike,
#     izhikevich.rebound_burst,
#     izhikevich.threshold_variability,
#     izhikevich.bistability,
#     izhikevich.dap,
#     izhikevich.accomodation,
#     izhikevich.inhibition_induced_spiking,
#     izhikevich.inhibition_induced_bursting,
# ]

def randn_range(size, a, b):
    x = torch.randn(size)
    return (b-a) * (x - x.min()) / (x.max() - x.min()) + a

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor): # TBCHW
        if isinstance(x_seq, tuple):
            x_seq = x_seq[0]  
        y_shape = [x_seq.shape[0], x_seq.shape[1]]  #T*B,C,H,W
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)
    
# class SNNetwork(torch.nn.Module):
#     def __init__(self, spiking_method: IzhikevichSpikingBehavior):
#         super(SNNetwork, self).__init__()
#         self.spiking_method = spiking_method
#         self.l0 = Izhikevich(spiking_method)
#         self.l1 = Izhikevich(spiking_method)
#         self.s0 = self.s1 = None

#     def forward(self, spikes):
#         spikes, self.s0 = self.l0(spikes, self.s0)
#         spike, self.s1 = self.l1(spikes, self.s1)
#         return (spike, self.s1)

class ConvNetIzhikevich(nn.Module):
    def __init__(self, spiking_method: IzhikevichSpikingBehavior, num_layers, kernel_size=3, weight_type='random'):
        super(ConvNetIzhikevich, self).__init__()
        self.layers = nn.ModuleList()
        self.spiking_method = spiking_method

        # Create convolutional layers
        for _ in range(num_layers):
            conv = SeqToANNContainer(
                nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
            )

            # Initialize weights based on type
            if weight_type == 'uniform':
                conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))
            else:  # random
                nn.init.xavier_normal_(conv.module.weight)
            self.layers.append(conv)

            # Set activation function
        self.activation = Izhikevich(spiking_method)
            

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x
if __name__ == "__main__":

    size = (500, 3, 3)
    x = randn_range(size, 500000, 5000000)
    model = ConvNetIzhikevich(
        spiking_method=izhikevich.spike_frequency_adaptation,
        num_layers=3,
        kernel_size=3,
        weight_type='random'
    )
    print(model)
    y = model(x)
    print(y.shape)
    
