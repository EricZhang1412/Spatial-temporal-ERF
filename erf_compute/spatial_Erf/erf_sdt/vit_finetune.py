import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from PIL import Image
from datasets import load_metric
import os
from transformers.activations import GELUActivation

# 参数配置
MODEL_PATH = './vit-tiny-16-224'
DATA_PATH = '/root/autodl-tmp/imagenet'  # 替换为实际路径
OUTPUT_DIR = './relu_vit_tiny_imagenet'
BATCH_SIZE = 680
NUM_EPOCHS = 20

# 1. 数据集加载
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', processor=None):
        self.root = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.JPEG'):
                    samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        else:
            inputs = {'pixel_values': self.transform(image)}
        inputs['labels'] = torch.tensor(label)
        return inputs

# 2. 加载处理工具（自动匹配模型）
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

# 3. 创建数据集
train_dataset = ImageNetDataset(DATA_PATH, 'train', processor)
val_dataset = ImageNetDataset(DATA_PATH, 'val', processor)

# 4. 修改模型激活函数为ReLU
def modify_to_relu(model):
    for name, module in model.named_modules():
        # 处理标准的GELU和HuggingFace的GELUActivation
        if isinstance(module, (torch.nn.GELU, GELUActivation)):
            # 获取父模块和子模块名
            parent = model
            names = name.split('.')
            for n in names[:-1]:
                parent = getattr(parent, n)
            
            # 替换为ReLU并保留原始配置
            new_relu = torch.nn.ReLU()
            if hasattr(module, 'inplace'):
                new_relu.inplace = module.inplace
            setattr(parent, names[-1], new_relu)
    
    # 特别处理中间层的激活函数
    for layer in model.vit.encoder.layer:
        if hasattr(layer.intermediate, 'intermediate_act_fn'):
            layer.intermediate.intermediate_act_fn = torch.nn.ReLU()
    
    return model

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1000,
    ignore_mismatched_sizes=True
)
model = modify_to_relu(model)

print(f"Model: {model}")
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# 5. 训练配置
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    weight_decay=0.01,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.1,
    logging_dir='./logs',
    remove_unused_columns=False,
    fp16=True if torch.cuda.is_available() else False,
    dataloader_num_workers=16,
    report_to="tensorboard"
)

# 6. 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 7. 开始训练
print("Starting training...")
trainer.train()

# 8. 保存最终模型
trainer.save_model(f"{OUTPUT_DIR}/final")
