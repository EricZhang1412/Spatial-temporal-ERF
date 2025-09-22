# Usage
## Spatial ERF Visualization for ViT-tiny-16-224
### 1. Convert `vit-tiny-16-224/model.safetensors` to `.pth` file
**Make sure that you have the correct model directory and have the necessary dependencies installed. Then, use the `convert_to_pth.py` script**:
```bash
python convert_to_pth.py
```
### 2. Visualize!
**Uncomment the code segment spanning L72-L102. Then, use the `erf_compute.py` script**

```bash
python erf_compute.py
```
Then you can get some files named `erf_vit_tiny_16_224_*.pdf` in the current directory.

> Note: The '_w_pretrained_' string indicates that the model was loaded with pretrained weights.

## Spatial ERF Visualization for ViT-tiny-ReLUact-16-224
### 1. (Not necessary) Get a ReLU-activation-based ViT-tiny-16-224 model
**If you have a model with ReLU activation, you can skip this step. Otherwise, you can use the `finetune.sh` script to finetune a model with ReLU activation**:

**First, be sure that you modified `vit_finetune.py`**:
```py
MODEL_PATH = './vit-tiny-16-224' # Original ViT-tiny-16-224 model path
DATA_PATH = '/root/autodl-tmp/imagenet'  # ImageNet dataset path
OUTPUT_DIR = './relu_vit_tiny_imagenet' # Output directory for the finetuned model
BATCH_SIZE = 680 # Batch size (per GPU) for finetuning
NUM_EPOCHS = 20 # Number of epochs for finetuning
```
**Then, run the `finetune.sh` script**:
```bash
bash finetune.sh
```
### 2. Convert `vit-tiny-ReLUact-16-224/model.safetensors` to `.pth` file
**Same as top...**:

### 3. Visualize!

## Other models:

The project support the following models at present:
| Model Name | Model Type |
|------------|------------|
| ViT-tiny-16-224 | ANN |
| ViT-tiny-ReLUact-16-224 | ANN |
| Q-ViT | ANN Quantized |
| Swin-Tiny | ANN |
| PVT-v2-b0 | ANN |
| Spikformer | SNN |
| Spike-driven Transformer v1-v3 | SNN |
| QKFormer | SNN |