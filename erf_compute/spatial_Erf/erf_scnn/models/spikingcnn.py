import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .Qtrick import *
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.surrogate import ATan, Sigmoid, Rect

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

class ConvNet(nn.Module):
    def __init__(self, num_layers, kernel_size=3, weight_type='random', activation='none', 
                 tau=2.0, Vth=1.0, alpha=2.0):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create convolutional layers
        for _ in range(num_layers):
            conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
            
            # Initialize weights based on type
            if weight_type == 'uniform':
                conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))
            else:  # random
                nn.init.xavier_normal_(conv.weight)
                # gaussian mask
                weights = conv.weight
                gaussian_mask = torch.randn_like(weights) 
                conv.weight.data *= gaussian_mask
            # print(conv)
            # print(conv.weight)
            # conv.weight = conv.weight * nn.Parameter(10)
            # print(conv.weight)
            self.layers.append(conv)
            
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'lif_atan':
            self.activation = LIFNode(tau=tau, v_threshold=Vth, surrogate_function=ATan(alpha=alpha), step_mode='m')
        elif activation == 'lif_sigmoid':
            self.activation = LIFNode(tau=tau, v_threshold=Vth, surrogate_function=Sigmoid(alpha=alpha), step_mode='m')
        elif activation == 'lif_rect':
            self.activation = LIFNode(tau=tau, v_threshold=Vth, surrogate_function=Rect(alpha=alpha), step_mode='m')
        elif activation == 'MultispikeNorm4':
            self.activation = MultiSpike_norm4()
        elif activation == 'Multispike4':
            self.activation = MultiSpike_4()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

class ConvNetDynamic(nn.Module):
    def __init__(self, num_layers, kernel_size=3, weight_type='random', tau=2.0, Vth=1.0, surrogate_mode='triangle', alpha=2.0):
        super(ConvNetDynamic, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create convolutional layers
        for _ in range(num_layers):
            # conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
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
        self.activation = LIF_with_Temporal_Gradient(tau=tau, Vth=Vth, surrogate_mode=surrogate_mode, alpha=alpha)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x, _ = self.activation(x)
        return x

if __name__ == '__main__':
    # Create a random input tensor
    size = (1, 1, 14, 14)
    x = randn_range(size, -5, 5)
    
    

    # Create a convolutional network with 3 layers
    # model = ConvNet(
    #     num_layers=5, 
    #     kernel_size=3, 
    #     weight_type=None, 
    #     activation='relu',
    #     )
    model = ConvNetDynamic(
        num_layers=5, 
        kernel_size=3, 
        weight_type=None, 
        tau=2.0, 
        Vth=1.0, 
        surrogate_mode='triangle', 
        alpha=2.0
        )
    
    # Forward pass
    y, v = model(x)
    
    # Print output shape
    print(y.shape)
    print(v.shape)

    # Print output
    print(y)