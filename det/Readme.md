

### Data preparation
You can download the COCO-2017 [here](https://cocodataset.org) and prepare the COCO follow this format:

```tree data
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```
It is suggested to link the data path as:
```bash
export DETECTRON2_DATASETS=/path/to/data
```


### Training
Download the pretrained model

Pre-trained ckpts 10M: [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).

Pre-trained ckpts 19M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).


Train 19M on 2 GPUs:

- `cd tools`
- `CUDA_VISIBLE_DEVICES=0,1 ./dist_train.sh ../configs\sdeformer_mask_rcnn\mask-rcnn_sdeformer_fpn_1x_coco.py 2`
