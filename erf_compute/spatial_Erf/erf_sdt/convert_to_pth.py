import torch
from transformers import AutoModel

# 1. 从Hugging Face加载模型
model_path = 'vit-tiny-ReLUact-16-224'  # 您的本地Hugging Face模型目录
hf_model = AutoModel.from_pretrained(model_path)

# 2. 获取模型的状态字典
state_dict = hf_model.state_dict()

# 3. 保存为.pth文件
torch.save(state_dict, 'vit_reluact_model.pth')

# # 或者，如果您想保存整个模型而不仅仅是权重
# torch.save(hf_model, 'vit_model_full.pth')

# # 如果您想包含额外信息，可以使用字典格式
# model_info = {
#     'model': state_dict,
#     'config': hf_model.config.to_dict(),
#     'model_type': 'vit'
# }
# torch.save(model_info, 'vit_model_with_config.pth')