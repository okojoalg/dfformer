# FFT-based Dynamic Token Mixer for Vision

This code is the official implementation of [DFFormer and CDFFormer](https://arxiv.org/pdf/2303.03932.pdf).

[FFT-based Dynamic Token Mixer for Vision](https://arxiv.org/pdf/2303.03932.pdf)

## Usage

### Requirements

- torch==1.12.1
- torchvision==0.13.1
- timm==0.5.4
- Pillow
- etc. (see requirements.txt)

### Data preparation

The ImageNet dataset should be downloaded and extracted with a directory
structure as specified.

```
imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Classification Training

**Single-node**

```bash
./distributed_train.sh 8 /path/to/imagenet --model dfformer_s18 -b 128 -j 8 --opt adamw --epochs 300 --sched cosine --native-amp --img-size 224 --drop-path 0.2 --lr 1e-3 --weight-decay 0.05 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --warmup-lr 1e-6 --warmup-epochs 20 --experiment DFFormer --task-name dfformer_s18
```

**Multi-node**

Please use mpirun, depending on your environment.

### Segmentation Training

Using pre-trained weights is necessary, so it must be set in advance.

```bash
bash segmentation/tools/dist_train.sh \
    segmentation/configs/fpn/dfformer_s18_fpn.py 8 \
    --work-dir work_dir/dfformer_s18_fpn --seed 42 --deterministic
```

### Object Detection Training

Using pre-trained weights is necessary, so it must be set in advance.

```bash
FORK_LAST3=1 bash detection/tools/dist_train.sh \
    detection/configs/retinanet/dfformer_m36_retinanet.py 8 \
    --work-dir work_dir/dfformer_m36_retinanet --seed 42 --deterministic
```

## Acknowledgment

Our implementation is based on MetaFormer Baselines for Vision, pytorch-image-models, mmsegmentation, and mmdetection.
