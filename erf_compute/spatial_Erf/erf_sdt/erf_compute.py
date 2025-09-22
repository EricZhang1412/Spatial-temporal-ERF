import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from models.sdtv3 import sdtv3_s, sdtv3_s_attn, sdtv3_s_channelmlp, sdtv3_s_fullattn, sdtv3_s_splash
from models.vit import VisionTransformer, VisionTransformer_attn, test_vit_attention, vit_tiny_patch16_224, vit_tiny_patch16_224_relu
from models.metaformer import poolformerv2_s12, caformer_s18, convformer_s18
from models.qkformer import QKFormer_10_384
from models.MAE_SDT import spikmae_12_512
from models.sdtv3_large import spikformer12_512
from models.sd_former_v1 import sdt
from models.spikformer import vit_snn

from models.q_vit.quant_vision_transformer import fourbits_deit_small_patch16_224

from functions.erf import compute_erf, compute_erf_sdt, compute_erf_pool, compute_erf_qk, compute_erf_sdtv1, compute_erf_spikformerv1, compute_erf_hf, compute_erf_swin, compute_erf_pvt
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoModel
from transformers.activations import GELUActivation

from spikingjelly.clock_driven import functional


def visualize_erf(erf_map, title="Effective Receptive Field", file_name="erf.pdf"):
    plt.figure(figsize=(10, 8))
    plt.imshow(erf_map, cmap='grey')
    plt.colorbar(label='Gradient Magnitude')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()

    #save
    plt.savefig(file_name, dpi=300)

def visualize_multierf(erf_maps, base_filename="erf"):
    for layer_name, erf_map in erf_maps.items():
        plt.figure(figsize=(10, 8))
        plt.imshow(erf_map, cmap='grey')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f"Effective Receptive Field - {layer_name}")
        plt.axis('off')
        plt.tight_layout()
        file_name = f"{base_filename}_{layer_name}.pdf"
        plt.savefig(file_name, dpi=300)
        plt.close()

    n_layers = len(erf_maps)
    rows = int(np.ceil(n_layers / 2))
    cols = min(n_layers, 2)
    
    plt.figure(figsize=(cols * 5, rows * 4))
    for i, (layer_name, erf_map) in enumerate(erf_maps.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(erf_map, cmap='grey')
        plt.colorbar(label='Magnitude')
        plt.title(layer_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{base_filename}_all_layers.pdf", dpi=300)
    plt.close()

    avg_erf = np.mean([erf_map for erf_map in erf_maps.values()], axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_erf, cmap='grey')
    plt.colorbar(label='Average Gradient Magnitude')
    plt.title("Average Effective Receptive Field Across All Layers")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{base_filename}_average.pdf", dpi=300)
    plt.close()

def visualize_multierf_swin(erf_maps, base_filename="erf", target_size=(128, 128)):
    # Step 1: 确定所有层的最大尺寸（可选自动计算）
    if target_size is None:
        all_sizes = [erf.shape for erf in erf_maps.values()]
        target_size = (max(s[0] for s in all_sizes), max(s[1] for s in all_sizes))
    
    # Step 2: 预处理 - 统一上采样到目标尺寸
    uniform_maps = {}
    for layer_name, erf_map in erf_maps.items():
        # 将numpy数组转为tensor进行上采样
        tensor_map = torch.from_numpy(erf_map).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        
        # 使用双线性插值上采样（保持数值范围）
        resized_map = F.interpolate(
            tensor_map, 
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # 保留原始层名与新尺寸图
        uniform_maps[layer_name] = resized_map
    
    # Step 3: 使用统一尺寸的可视化（原逻辑稍作调整）
    for layer_name, erf_map in uniform_maps.items():
        plt.figure(figsize=(10, 8))
        plt.imshow(erf_map, cmap='grey')  # 建议使用viridis等高对比度色彩
        plt.colorbar(label='Normalized Magnitude')
        plt.title(f"ERF - {layer_name} (resized to {target_size})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{base_filename}_{layer_name}.pdf", bbox_inches='tight', dpi=300)
        plt.close()
    # 组合图绘制（使用统一尺寸）
    n_layers = len(uniform_maps)
    rows = int(np.ceil(n_layers / 2))
    cols = min(n_layers, 2)
    
    plt.figure(figsize=(cols * 6, rows * 5))  # 略微增加画布尺寸
    for i, (layer_name, erf_map) in enumerate(uniform_maps.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(erf_map, cmap='grey')
        plt.colorbar(label='Norm. Magnitude')
        
        # 缩短标题显示（适合多层情况）
        short_name = layer_name.replace('_attn_proj', '')  # 示例：移除冗余信息
        plt.title(f"L{short_name}", pad=10)
        plt.axis('off')
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f"{base_filename}_all_layers.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    # 平均图计算（基于统一尺寸）
    avg_erf = np.mean(list(uniform_maps.values()), axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_erf, cmap='grey')  # 使用不同色系区分
    plt.colorbar(label='Avg. Magnitude (normalized)')
    plt.title(f"Avg. ERF Across {n_layers} Layers\n({target_size} unified size)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{base_filename}_average.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# ############## ERF of ViT-Tiny-16-224 ###############
# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#     processor = ViTImageProcessor.from_pretrained('./vit-tiny-16-224')
#     model = ViTForImageClassification.from_pretrained('./vit-tiny-16-224')

#     ## without pretrained
#     model = vit_tiny_patch16_224(pretrained="/data2/users/zhangjy/erf_sdt/vit_model.pth")
    
#     # model = VisionTransformer_attn(
#     #     img_size = 224,
#     #     patch_size = 16,
#     #     in_chans = 3,
#     #     num_classes = 1000,
#     #     global_pool = 'token',
#     #     embed_dim = 192,
#     #     depth=12, 
#     #     num_heads=3,
#     # ) 
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf(model, image_size=(224, 224), num_runs=20)
#     # visualize_erf(single_erf, title="ERF of ViT-Base-16-224", file_name="erf_vit_tiny_16_224_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_vit_tiny_16_224_w_pretrained")

# ############## ERF of ViT-Tiny-ReluAct-16-224 ###############
# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     import timm

#     def modify_to_relu(model):
#         for name, module in model.named_modules():
#             if isinstance(module, (torch.nn.GELU, GELUActivation)):
#                 parent = model
#                 names = name.split('.')
#                 for n in names[:-1]:
#                     parent = getattr(parent, n)

#                 new_relu = torch.nn.ReLU()
#                 if hasattr(module, 'inplace'):
#                     new_relu.inplace = module.inplace
#                 setattr(parent, names[-1], new_relu)

#         for layer in model.vit.encoder.layer:
#             if hasattr(layer.intermediate, 'intermediate_act_fn'):
#                 layer.intermediate.intermediate_act_fn = torch.nn.ReLU()

#         return model

#     model = ViTForImageClassification.from_pretrained("./vit-tiny-ReLUact-16-224")
#     model = modify_to_relu(model)

#     print(f"Model: {model}")
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf_hf(model, image_size=(224, 224), num_runs=60)
#     # visualize_erf(single_erf, title="ERF of ViT-Base-16-224", file_name="erf_vit_tiny_16_224_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_vit_tiny_relu_16_224_w_pretrained")

# # ############### ERF of Q-ViT-4bit ###############
# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     import timm

#     model = fourbits_deit_small_patch16_224()
#     print(f"Model: {model}")
#     state_dict = torch.load("pretrained/deit_t_best_checkpoint_4bit.pth", map_location='cpu', weights_only=False)
#     model.load_state_dict(state_dict["model"], strict=True)
#     model.eval()
#     # for name, param in model.named_parameters():
#     #     print(name, param.size())
#     # print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     single_erf = compute_erf(model, image_size=(224, 224), num_runs=40)
#     # visualize_erf(single_erf, title="ERF of ViT-Base-16-224", file_name="erf_vit_tiny_16_224_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_qvit4bit_tiny_w_pretrained")

# # ############### ERF of Swin-T ###############
# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     import timm

#     model = timm.create_model('swin_tiny_patch4_window7_224', pretrained='pretrained/swin_tiny_patch4_window7_224_22kto1k_finetune.pth')
#     print(f"Model: {model}")
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf_swin(model, image_size=(224, 224), num_runs=20)
#     # visualize_erf(single_erf, title="ERF of ViT-Base-16-224", file_name="erf_vit_tiny_16_224_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_swin_tiny_patch4_window7_224_w_pretrained")

# # ############### ERF of PVT ###############
# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     import timm

#     model = timm.create_model('pvt_v2_b0', pretrained='pretrained/pvt_v2_b0')
#     print(f"Model: {model}")
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf_pvt(model, image_size=(224, 224), num_runs=20)
#     # visualize_erf(single_erf, title="ERF of ViT-Base-16-224", file_name="erf_vit_tiny_16_224_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_pvt_v2_b0_w_pretrained")


# ############### ERF of SDTV1-8-384 ###############
# if __name__ == "__main__":
    
#     model = sdt(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         in_channels=3,
#         embed_dims=384,
#         num_heads=8,
#         mlp_ratios=4,
#         qkv_bias=True,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#         depths=8,
#         sr_ratios=[8, 4, 2],
#         T=4,
#         pooling_stat="1111",
#         attn_mode="direct_xor",
#         spike_mode="lif",
#     ) 
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())


#     state_dict = torch.load('sd_former_v1_8_384.pth.tar', map_location='cpu')
#     model.load_state_dict(state_dict["state_dict"], strict=False)

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     functional.reset_net(model)

#     single_erf = compute_erf_sdtv1(model, image_size=(224, 224), num_runs=90)
#     # visualize_erf(single_erf, title="ERF of SDTV1-8-384", file_name="erf_sdtv1_s_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="SDTV1-8-384")

# ############### ERF of SpikformerV1-8-512 ###############
# if __name__ == "__main__":
    
#     model = vit_snn() 
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     state_dict = torch.load('spikformer_checkpoint-308.pth.tar', map_location='cpu')
#     model.load_state_dict(state_dict["state_dict"])

#     device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     functional.reset_net(model)

#     single_erf = compute_erf_spikformerv1(model, image_size=(224, 224), num_runs=300)
#     # visualize_erf(single_erf, title="ERF of SDTV1-8-384", file_name="erf_spikformerv1_s_w_pretrained.pdf")
#     visualize_multierf(single_erf, base_filename="erf_spikformerv1_s_w_pretrained")

# ############### ERF of SDTV3-S-16-224 ###############
# if __name__ == "__main__":
    
#     model = sdtv3_s()
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     state_dict = torch.load('pretrained/V3_5.1M_1x4.pth', map_location='cpu', weights_only=False)
#     model.load_state_dict(state_dict['model'], strict=False)

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf_sdt(model, image_size=(224, 224), num_runs=60)
#     visualize_multierf(single_erf, base_filename="erf_sdtv3_s_w_pretrained.pdf")

############### ERF of SDTV3-S-channelmlp/splash-and-reconstruct ###############
if __name__ == "__main__":
    
    model = sdtv3_s_splash()  
    model.eval()
    for name, param in model.named_parameters():
        print(name, param.size())

    state_dict = torch.load('pretrained/sdtv3_s_splash_blr5e-4/checkpoint-199.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model'])

    # device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = model.to(device)

    single_erf = compute_erf_sdt(model, image_size=(224, 224), num_runs=60)
    visualize_multierf(single_erf, base_filename="erf_sdtv3_s_splash_w_pretrained.pdf")


# ############### ERF of SDTV3-MAE-12-512 ###############
# if __name__ == "__main__":
    
#     model = spikformer12_512()
#     model.eval()
#     for name, param in model.named_parameters():
#         print(name, param.size())

#     # state_dict = torch.load('V3_5.1M_1x4.pth', map_location='cpu')
#     # model.load_state_dict(state_dict['model'], strict=False)

#     device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     single_erf = compute_erf_sdt(model, image_size=(224, 224), num_runs=500)
#     visualize_erf(single_erf, title="ERF of SDTV3-S-16-224", file_name="erf_sdtv3_s.pdf")

# ############### ERF of ANN Metaformer ###############
# if __name__ == "__main__":
    
#     model = poolformerv2_s12()
#     model.eval()

#     for name, param in model.named_parameters():
#         print(name, param.size())

#     state_dict = torch.load('poolformerv2_s12.pth', map_location='cpu')
#     model.load_state_dict(state_dict)

#     device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)


#     single_erf = compute_erf_pool(model, image_size=(224, 224), num_runs=60)
#     visualize_multierf(single_erf, base_filename="erf_poolformerv2_s12_w_pretrained")

# ############### ERF of QKFormer ###############
# if __name__ == "__main__":
    
#     model = QKFormer_10_384(T = 4)
#     model.eval()

#     for name, param in model.named_parameters():
#         print(name, param.size())

#     state_dict = torch.load('qk_10_384_checkpoint-199.pth', map_location='cpu')
#     model.load_state_dict(state_dict["model"])
    

#     device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     # input_path = "/data/dataset/ImageNet/val/n01440764/ILSVRC2012_val_00000293.JPEG"
#     single_erf = compute_erf_qk(model, image_size=(224, 224), num_runs=50, input_path=None)
#     visualize_multierf(single_erf, base_filename="erf_qkformer_10_384_w_pretrained.pdf")