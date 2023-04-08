_base_ = [
    '../_base_/models/fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='SegDFFormerM36',
    ),
    neck=dict(
        in_channels=[96, 192, 384, 576],
    ),
    decode_head=dict(
        num_classes=150,
    ),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

find_unused_parameters = True
