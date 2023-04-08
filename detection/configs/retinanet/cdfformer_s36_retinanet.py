_base_ = [
    '../_base_/models/retinanet.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='DetCDFFormerS36',
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# do not use apex fp16
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# use apex fp16
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

find_unused_parameters = True