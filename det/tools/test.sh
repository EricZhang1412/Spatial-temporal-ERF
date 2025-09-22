CUDA_VISIBLE_DEVICES=4 bash dist_test.sh ..configs/c_mlp_maskrcnn/sr_fpn_1x_coco.py work_dirs/sr_fpn_1x_coco/epoch_1.pth 1
# python get_flops.py ..configs/c_mlp_maskrcnn/sr_fpn_1x_coco.py