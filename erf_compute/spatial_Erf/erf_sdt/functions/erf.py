import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from spikingjelly.clock_driven import functional



def compute_erf(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    features_dict = {}

    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    feature_names = [f'B{i}_attn_proj' for i in range(12)]
    handles_proj = []
    for i in range(12):
        handle = model.blocks[i].attn.proj.register_forward_hook(
            get_features(feature_names[i])
        )
        handles_proj.append(handle)
    all_layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):
        x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
        x.requires_grad_(True)
        _ = model(x)

        for name in feature_names:
            if name not in features_dict:
                raise ValueError(f"未能获取到 {name} 的特征")

        accumulated_grad = None
        for name in feature_names:
            layer_features = features_dict[name]

            layer_features = layer_features[:, 1:]  # ignore <cls> token
            # layer_features = layer_features[:, :-1]  # ignore <distill> token if using DeiT, remenber to uncomment this line when using DeiT/Q-ViT.
            layer_features = rearrange(layer_features, 'b n c -> b c n')
            B, C, N = layer_features.shape
            H = int(np.sqrt(N))
            W = H
            layer_features = layer_features.view(B, C, H, W)
            feature_size = layer_features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            # input gradient
            grad_output = torch.zeros_like(layer_features)
            grad_output[0, :, center_h, center_w] = 1.0

            layer_features.backward(grad_output, retain_graph=True)

            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()

            all_layer_gradients[name].append(gradient)

            x.grad.zero_()
        features_dict.clear()

    for handle in handles_proj:
        handle.remove()

    avg_layer_gradients = {}
    for name in feature_names:
        avg_gradient = np.mean(all_layer_gradients[name], axis=0)
        avg_gradient = np.maximum(0, avg_gradient)  # 忽略负值
        avg_layer_gradients[name] = avg_gradient
    
    return avg_layer_gradients

def compute_erf_hf(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()

    features_dict = {}

    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    feature_names = [f'B{i}_attn_proj' for i in range(12)]
    handles_proj = []
    for i in range(12):
        handle = model.vit.encoder.layer[i].attention.output.register_forward_hook(
            get_features(feature_names[i])
        )
        handles_proj.append(handle)

    all_layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):
        x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
        x.requires_grad_(True)
        _ = model(x)

        for name in feature_names:
            if name not in features_dict:
                raise ValueError(f"未能获取到 {name} 的特征")
        accumulated_grad = None
        for name in feature_names:
            layer_features = features_dict[name]
            
            layer_features = layer_features[:, 1:]  # ignore <cls> token
            layer_features = rearrange(layer_features, 'b n c -> b c n')
            B, C, N = layer_features.shape
            H = int(np.sqrt(N))
            W = H
            layer_features = layer_features.view(B, C, H, W)
            feature_size = layer_features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            # input gradient
            grad_output = torch.zeros_like(layer_features)
            grad_output[0, :, center_h, center_w] = 1.0

            layer_features.backward(grad_output, retain_graph=True)

            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()

            all_layer_gradients[name].append(gradient)

            x.grad.zero_()
        features_dict.clear()

    for handle in handles_proj:
        handle.remove()
    avg_layer_gradients = {}
    for name in feature_names:
        avg_gradient = np.mean(all_layer_gradients[name], axis=0)
        avg_gradient = np.maximum(0, avg_gradient)  # 忽略负值
        avg_layer_gradients[name] = avg_gradient
    
    return avg_layer_gradients

def compute_erf_swin(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    
    features_dict = {}

    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    feature_names = []
    handles_proj = []

    stage_block_counts = {0: 2, 1: 2, 2: 6, 3: 2}  # stage0有2blocks, stage1有2blocks, stage2有6blocks, stage3有2blocks
    
    for stage in range(4):  # Swin通常有4个stage
        for block in range(stage_block_counts[stage]):
            feature_name = f"S{stage}B{block}_attn_proj"
            feature_names.append(feature_name)

            target_layer = model.layers[stage].blocks[block].attn.proj

            handle = target_layer.register_forward_hook(get_features(feature_name))
            handles_proj.append(handle)
    all_layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):
        x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
        x.requires_grad_(True)
        _ = model(x)
        for name in feature_names:
            if name not in features_dict:
                raise ValueError(f"未能获取到 {name} 的特征")
        accumulated_grad = None
        for name in feature_names:
            layer_features = features_dict[name]
            print(f"{name} 特征形状: {layer_features.shape}")
            layer_features = rearrange(layer_features, 'b n c -> b c n')
            B, C, N = layer_features.shape
            H = int(np.sqrt(N))
            W = H
            layer_features = layer_features.view(B, C, H, W)
            feature_size = layer_features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            # input gradient
            grad_output = torch.zeros_like(layer_features)
            grad_output[0, :, center_h, center_w] = 1.0

            layer_features.backward(grad_output, retain_graph=True)

            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()

            all_layer_gradients[name].append(gradient)

            x.grad.zero_()
        features_dict.clear()

    for handle in handles_proj:
        handle.remove()


    avg_layer_gradients = {}
    for name in feature_names:
        avg_gradient = np.mean(all_layer_gradients[name], axis=0)
        avg_gradient = np.maximum(0, avg_gradient)  # 忽略负值
        avg_layer_gradients[name] = avg_gradient
    
    return avg_layer_gradients


def compute_erf_pvt(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    features_dict = {}
 
    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    feature_names = []
    handles_proj = []
    
    stage_block_counts = {0: 2, 1: 2, 2: 2, 3: 2}  # stage0有2blocks, stage1有2blocks, stage2有6blocks, stage3有2blocks
    
    for stage in range(4):  # Swin通常有4个stage
        for block in range(stage_block_counts[stage]):
            feature_name = f"S{stage}B{block}_attn_proj"
            feature_names.append(feature_name)

            target_layer = model.stages[stage].blocks[block].attn.proj

            handle = target_layer.register_forward_hook(get_features(feature_name))
            handles_proj.append(handle)
    all_layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):
        x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
        x.requires_grad_(True)
        _ = model(x)
        for name in feature_names:
            if name not in features_dict:
                raise ValueError(f"未能获取到 {name} 的特征")
        accumulated_grad = None
        for name in feature_names:
            layer_features = features_dict[name]
            print(f"{name} 特征形状: {layer_features.shape}")
            layer_features = rearrange(layer_features, 'b n c -> b c n')
            B, C, N = layer_features.shape
            H = int(np.sqrt(N))
            W = H
            layer_features = layer_features.view(B, C, H, W)
            feature_size = layer_features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            # input gradient
            grad_output = torch.zeros_like(layer_features)
            grad_output[0, :, center_h, center_w] = 1.0
            
            layer_features.backward(grad_output, retain_graph=True)

            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()
            
            all_layer_gradients[name].append(gradient)
            x.grad.zero_()
        features_dict.clear()

    for handle in handles_proj:
        handle.remove()

    
    # return avg_gradient
    avg_layer_gradients = {}
    for name in feature_names:
        avg_gradient = np.mean(all_layer_gradients[name], axis=0)
        avg_gradient = np.maximum(0, avg_gradient)  # 忽略负值
        avg_layer_gradients[name] = avg_gradient
    
    return avg_layer_gradients
        
    
def compute_erf_sdt(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    # 创建一个字典来存储中间特征
    features_dict = {}
    
    # 定义钩子函数来获取特定层的输出
    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    # 定义要获取的层
    layer_configs = [
        # (model.ConvBlock1_2[0].conv2, 'stage0_conv2'),
        # (model.ConvBlock2_2[0].conv2, 'stage1_conv2'),
        (model.block3, 'block3'),
        (model.block4, 'block4'),
        
    ]

    handles = []
    feature_names = []
    
    for block, block_name in layer_configs:
        for i in range(len(block)):  
            try:
                name = f"{block_name}_{i}_attn_proj_conv"
                handle = block[i].attn.proj_conv.register_forward_hook(get_features(name))
                handles.append(handle)
                feature_names.append(name)
            except (AttributeError, IndexError) as e:
                print(f"cannot register {block_name}_{i} hook: {e}")

    layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):  
        for name in feature_names:
            x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
            x.requires_grad_(True)
            _ = model(x)
            
            if name not in features_dict:
                print(f"Warning: Cannot get feature of {name} , skipping...")
                continue

            features = features_dict[name]

            if len(features.shape) == 3:  # [B, N, C]
                B, N, C = features.shape
                features = features.permute(0, 2, 1)  # [B, C, N]
                H = int(np.sqrt(N))
                W = H
                features = features.view(B, C, H, W)
            elif len(features.shape) == 4:  # [B, C, H, W]
                B, C, H, W = features.shape
            else:
                print(f"Warning: Unexpected shape of {name}: {features.shape}, skipping...")
                continue
            
            feature_size = features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2 + 5
            
            grad_output = torch.zeros_like(features)
            grad_output[0, :, center_h, center_w] = 1.0
            
            features.backward(grad_output, retain_graph=True)
            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()
            
            layer_gradients[name].append(gradient)
            
            x.grad = None
            features_dict.clear()
            try:
                if hasattr(model, 'reset_net'):
                    model.reset_net()
                elif hasattr(functional, 'reset_net'):
                    functional.reset_net(model)
            except:
                pass  
    
    for handle in handles:
        handle.remove()
    
    avg_gradients = {}
    for name in feature_names:
        if layer_gradients[name]: 
            avg_gradient = np.mean(layer_gradients[name], axis=0)
            avg_gradient = np.maximum(0, avg_gradient)
            avg_gradients[name] = avg_gradient
    
    return avg_gradients

def compute_erf_sdtv1(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    features_dict = {}
    
    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    handles = []
    feature_names = []
    
    for i in range(len(model.block)):
        try:
            name = f"B{i}_attn_proj_conv"
            handle = model.block[i].attn.proj_conv.register_forward_hook(get_features(name))
            handles.append(handle)
            feature_names.append(name)
            print(f"Register hook: {name}")
        except (AttributeError, IndexError) as e:
            print(f"Failed to register hook for block[{i}].attn.proj_conv: {e}")

    layer_gradients = {name: [] for name in feature_names}

    for _ in range(num_runs):
        for name in feature_names:
            x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
            x.requires_grad_(True)
            _ = model(x)

            if name not in features_dict:
                print(f"Warning: Cannot get feature of {name}, skipping...")
                continue
            features = features_dict[name]

            print(f"{name}: {features.shape}")

            if len(features.shape) == 3:  # [B, C, N] or [B, N, C]
                B = features.shape[0]
                if features.shape[2] > features.shape[1]:  # [B, N, C]
                    features = features.permute(0, 2, 1)  # [B, C, N]
                
                C, N = features.shape[1], features.shape[2]
                H = int(np.sqrt(N))
                W = H
                features = features.view(B, C, H, W)
            elif len(features.shape) == 4:  # [B, C, H, W]
                pass
            else:
                print(f"Warning: Unexpected shape of {name}: {features.shape}, skipping...")
                continue
            
            feature_size = features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            grad_output = torch.zeros_like(features)
            grad_output[0, :, center_h, center_w] = 1.0

            features.backward(grad_output, retain_graph=True)
            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()
            layer_gradients[name].append(gradient)
            
            x.grad = None
            features_dict.clear()

            try:
                functional.reset_net(model)
            except:
                print("Warning: Failed to reset model, continuing...")

    for handle in handles:
        handle.remove()

    avg_gradients = {}
    for name in feature_names:
        if layer_gradients[name]:  
            avg_gradient = np.mean(layer_gradients[name], axis=0)
            avg_gradient = np.maximum(0, avg_gradient) 
            avg_gradients[name] = avg_gradient
    
    return avg_gradients


def compute_erf_spikformerv1(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()
    features_dict = {}

    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    handle7 = model.block[7].mlp.fc2_conv.register_forward_hook(
        get_features('B7_mlp_fc2_conv')
    )
    
    feature_names = [f'B{i}_attn_proj' for i in range(8)]
    handles_proj = []
    for i in range(8):
        handle = model.block[i].attn.proj_conv.register_forward_hook(
            get_features(feature_names[i])
        )
        handles_proj.append(handle) 

    layer_gradients = {name: [] for name in feature_names}

    for _ in range(num_runs):
        for name in feature_names:
            x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
            # x = x * 1e-6
            x.requires_grad_(True)
            _ = model(x)
            
            if name not in features_dict:
                raise ValueError(f"Cannot get feature of {name}, skipping...")
            
            features = features_dict[name]
            B, C, N = features.shape
            H = int(np.sqrt(N))
            W = H
            features = features.view(B, C, H, W)
            
            feature_size = features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2

            grad_output = torch.zeros_like(features)
            grad_output[0, :, center_h, center_w] = 1.0
            features.backward(grad_output)

            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()
            layer_gradients[name].append(gradient)

            x.grad = None
            features_dict.clear()
            functional.reset_net(model)

    handle7.remove()
    for handle in handles_proj:
        handle.remove()
    
    avg_gradients = {}
    for name in feature_names:
        avg_gradient = np.mean(layer_gradients[name], axis=0)
        avg_gradient = np.maximum(0, avg_gradient)
        avg_gradients[name] = avg_gradient
    
    return avg_gradients
    

def compute_erf_pool(model, image_size=(128, 128), num_runs=20):
    device = next(model.parameters()).device
    model.eval()

    # Create a dictionary to store intermediate features
    features_dict = {}
    
    # Define hook function to get specific layer outputs
    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    # Register hooks for all token_mixer.dwconv layers from stages.0.0 to stages.3.2
    handles = []
    feature_names = []
    
    ###### convformer ######
    # # Define block count for each stage based on your architecture
    # stage_blocks = {
    #     0: 3,  # stage 0 has 3 blocks
    #     1: 3,  # stage 1 has 3 blocks
    #     2: 9,  # stage 2 has 9 blocks
    #     3: 3   # stage 3 has 3 blocks
    # }
    
    # # Dynamically register hooks for all requested layers
    # for i in range(4):  # stages 0-3
    #     for j in range(stage_blocks[i]):
    #         if hasattr(model.stages[i][j], 'token_mixer') and hasattr(model.stages[i][j].token_mixer, 'dwconv'):
    #             layer_name = f'stage{i}_{j}_token_mixer_dwconv'
    #             feature_names.append(layer_name)
    #             handle = model.stages[i][j].token_mixer.dwconv.register_forward_hook(
    #                 get_features(layer_name)
    #             )
    #             handles.append(handle)
    
    # ###### caformer ######
    # # Define block count for each stage based on your architecture
    # stage_blocks = {
    #     0: 3,  # stage 0 has 3 blocks
    #     1: 3,  # stage 1 has 3 blocks
    #     2: 9,  # stage 2 has 9 blocks
    #     3: 3   # stage 3 has 3 blocks
    # }

    # # Dynamically register hooks for all requested layers
    # for i in range(4):  # stages 0-3
    #     for j in range(stage_blocks[i]):
    #         if hasattr(model.stages[i][j], 'token_mixer'):
    #             if i < 2:  # For stage 0 and 1, keep dwconv
    #                 if hasattr(model.stages[i][j].token_mixer, 'dwconv'):
    #                     layer_name = f'stage{i}_{j}_token_mixer_dwconv'
    #                     feature_names.append(layer_name)
    #                     handle = model.stages[i][j].token_mixer.dwconv.register_forward_hook(
    #                         get_features(layer_name)
    #                     )
    #                     handles.append(handle)
    #             else:  # For stage 2 and 3, use proj
    #                 if hasattr(model.stages[i][j].token_mixer, 'proj'):
    #                     layer_name = f'stage{i}_{j}_token_mixer_proj'
    #                     feature_names.append(layer_name)
    #                     handle = model.stages[i][j].token_mixer.proj.register_forward_hook(
    #                         get_features(layer_name)
    #                     )
    #                     handles.append(handle)    

    ###### poolformer ######
    # Define block count for each stage based on your architecture
    stage_blocks = {
        0: 2,  # stage 0 has 2 blocks
        1: 2,  # stage 1 has 2 blocks
        2: 6,  # stage 2 has 6 blocks
        3: 2   # stage 3 has 2 blocks
    }

    # Dynamically register hooks for norm2 in all stages
    feature_names = []
    handles = []

    for i in range(4):  # stages 0-3
        for j in range(stage_blocks[i]):
            if hasattr(model.stages[i][j], 'norm2'):
                layer_name = f'stage{i}_{j}_norm2'
                feature_names.append(layer_name)
                handle = model.stages[i][j].norm2.register_forward_hook(
                    get_features(layer_name)
                )
                handles.append(handle)

    
    
    # Create a dictionary to store gradients for each layer
    layer_gradients = {name: [] for name in feature_names}
    
    for _ in range(num_runs):
        for name in feature_names:
            # Perform separate forward and backward pass for each layer
            x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
            x.requires_grad_(True)
            _ = model(x)
            
            # Check if feature was captured
            if name not in features_dict:
                print(f"Warning: Could not capture features for {name}, skipping...")
                continue
            
            features = features_dict[name]
            
            # Handle different possible shapes of features
            if len(features.shape) == 4:  # B, C, H, W or B, H, W, C
                if features.shape[1] > features.shape[2] and features.shape[1] > features.shape[3]:
                    # Format is likely B, C, H, W
                    B, C, H, W = features.shape
                else:
                    # Format is likely B, H, W, C
                    B, H, W, C = features.shape
                    features = features.permute(0, 3, 1, 2)  # Convert to B, C, H, W
            else:
                # Handle other possible shapes (e.g., B, N, C)
                B, N, C = features.shape
                H = W = int(np.sqrt(N))
                features = features.reshape(B, C, H, W)
            
            # Get center positions
            center_h, center_w = H // 2, W // 2
            
            # Create gradient output
            grad_output = torch.zeros_like(features)
            grad_output[0, :, center_h, center_w] = 1.0
            
            # Compute gradients
            features.backward(grad_output)
            
            # Get gradients
            gradient = x.grad.abs().detach()
            gradient = gradient.mean(dim=1).squeeze(0).cpu().numpy()
            
            # Add to layer's gradient list
            layer_gradients[name].append(gradient)
            
            # Clear gradients and features dictionary
            x.grad = None
            features_dict.clear()
            
            # Reset model if needed (assuming functional.reset_net is defined)
            if 'functional' in globals():
                functional.reset_net(model)
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    # Calculate average gradients for each layer
    avg_gradients = {}
    for name in feature_names:
        if layer_gradients[name]:  # Check if we have gradients for this layer
            avg_gradient = np.mean(layer_gradients[name], axis=0)
            avg_gradient = np.maximum(0, avg_gradient)  # Ignore negative values
            avg_gradients[name] = avg_gradient
    
    return avg_gradients

def compute_erf_qk(model, image_size=(128, 128), num_runs=20, input_path=None):
    device = next(model.parameters()).device
    model.eval()
    # 创建一个字典来存储中间特征
    features_dict = {}
   
    # 定义钩子函数来获取特定层的输出
    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
   
    # 注册所有需要分析的层
    handles = []
    feature_names = []
    
    # 在这里添加所有要分析的层
    target_layers = [
        (model.stage1[0].tssa.proj_conv, 'stage1_0_tssa_proj_conv'),
        (model.stage2[0].tssa.proj_conv, 'stage2_0_tssa_proj_conv'),
        (model.stage2[1].tssa.proj_conv, 'stage2_1_tssa_proj_conv')
    ]
    
    # 添加stage3的所有attn.proj_conv层
    for i in range(7):  # 假设有7个子块，从0到6
        try:
            target_layers.append(
                (model.stage3[i].attn.proj_conv, f'stage3_{i}_attn_proj_conv')
            )
        except (AttributeError, IndexError) as e:
            print(f"无法添加 stage3[{i}].attn.proj_conv: {e}")
    
    # 注册所有钩子
    for module, name in target_layers:
        try:
            handle = module.register_forward_hook(get_features(name))
            handles.append(handle)
            feature_names.append(name)
            print(f"成功注册钩子: {name}")
        except Exception as e:
            print(f"注册钩子失败 {name}: {e}")
   
    # 为每一层创建单独的梯度列表
    layer_gradients = {name: [] for name in feature_names}
   
    for _ in range(num_runs):  # 修正了语法错误：for * in range(num*runs)
        for name in feature_names:
            # 对每一层进行单独的前向和反向传播
            # 准备输入数据
            if input_path is not None:
                # 加载图像
                from PIL import Image  
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((image_size[0], image_size[1])),
                    transforms.ToTensor(),
                ])
                image = Image.open(input_path).convert('RGB')
                x = transform(image)
                x = x.unsqueeze(0)
                x = x.to(device)
            else:
                x = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
                # x = x * 1e-10  # 根据原始代码的设置
           
            x.requires_grad_(True)
            _ = model(x)
            
            # 确保该层的特征已获取
            if name not in features_dict:
                print(f"警告: 未能获取到 {name} 的特征，跳过")
                continue
            
            # 处理当前层的特征
            features = features_dict[name]
            
            # 打印特征形状进行调试
            # print(f"{name} 特征形状: {features.shape}")
            
            # 检查特征的形状
            if len(features.shape) == 3:  # [B, C, N] 或 [B, N, C]
                B = features.shape[0]
            
                C, N = features.shape[1], features.shape[2]
                H = int(np.sqrt(N))
                W = H
                features = features.view(B, C, H, W)
            elif len(features.shape) == 4:  # 已经是 [B, C, H, W]
                # 已经是正确的形状，不需要处理
                pass
            else:
                print(f"警告: {name} 的特征形状异常: {features.shape}，跳过")
                continue
            
            feature_size = features.shape[2:]
            center_h, center_w = feature_size[0] // 2, feature_size[1] // 2
            
            # 创建梯度输出
            grad_output = torch.zeros_like(features)
            grad_output[:, :, center_h, center_w] = 1.0  # 使用原始代码中的值
            
            # 计算梯度
            features.backward(grad_output, retain_graph=True)
            
            # 获取梯度
            gradient = x.grad.abs().detach()
            gradient = gradient.mean(axis=1).squeeze(0).cpu().numpy()
            
            # 添加到该层的梯度列表
            layer_gradients[name].append(gradient)
            
            # 清除梯度和特征字典
            x.grad = None
            features_dict.clear()
            
            # 重置模型
            try:
                functional.reset_net(model)
            except Exception as e:
                print(f"重置模型失败: {e}")
                pass
    
    # 移除所有钩子
    for handle in handles:
        handle.remove()
    
    # 计算每一层的平均梯度
    avg_gradients = {}
    for name in feature_names:
        if layer_gradients[name]:  # 检查是否有梯度数据
            avg_gradient = np.mean(layer_gradients[name], axis=0)
            avg_gradient = np.maximum(0, avg_gradient)  # 忽略负值
            avg_gradients[name] = avg_gradient
    
    return avg_gradients