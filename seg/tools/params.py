from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
from mmseg.apis import inference_model, init_model
from mmengine import Config
import torch
import sys

import pprint

pprint.pprint(sys.path)
sys.path.append('/data/users/zhouxl/zjy/sdtv3_zjy/SDT_V3/Segmentation/tools/params.py')
sys.path.append('sdtv3_zjy/SDT_V3/Segmentation/configs/C-MLP/c-mlp_512x512_ade20k.py')

 
 
def load_model(config_path, checkpoint_path, resize_height, resize_width, class_num):
    cfg = Config.fromfile(config_path)
    cfg.crop_size = (resize_height, resize_width)
    cfg.data_preprocessor.size = cfg.crop_size
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(resize_width, resize_height), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]
 
    cfg.model.decode_head.num_classes = class_num
    print('class_num=', class_num)
    model = init_model(cfg, checkpoint_path, 'cuda:0')
 
    return model
 
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('==> Building model..')
    config_path = '../configs/C-MLP/c-mlp_512x512_ade20k.py'
    # checkpoint_path = '/data/users/zhouxl/zjy/sdtv3_zjy/SDT_V3/Segmentation/tools/work_dirs/c-mlp_512x512_ade20k/best_mIoU_iter_160000.pth'
    checkpoint_path = None
    # config_path = 'deeplabv3plus_r50b-d8_4xb2-80k_cityscapes-512x1024.py'
    # checkpoint_path = 'deeplabv3plus_r50b-d8_512x1024_80k_cityscapes_20201225_213645-a97e4e43.pth'
    resize_width = 512
    resize_height = 512
    class_num = 150
    net = load_model(config_path, checkpoint_path, resize_height, resize_width, class_num)
    net = net.to(device=device)
    input_shape = (3, resize_height, resize_width)
    outputs = get_model_complexity_info(net, input_shape=input_shape, show_table=False, show_arch=False)
    flops = _format_size(outputs['flops'])
    params = _format_size(outputs['params'])
    print("flops:{}".format(flops))
    print("params:{}".format(params))