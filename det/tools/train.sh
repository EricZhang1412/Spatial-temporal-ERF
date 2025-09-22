# bash dist_train.sh ../configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh ..configs/sdeformer_mask_rcnn/mask-rcnn_sdeformer_fpn_1x_coco.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train.sh ..configs/c_mlp_maskrcnn/sr_fpn_1x_coco.py 4
