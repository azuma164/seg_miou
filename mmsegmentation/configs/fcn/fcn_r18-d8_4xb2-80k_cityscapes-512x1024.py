_base_ = './fcn_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

# test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='work_dirs/format_results')
test_evaluator = dict(
    type='CityscapesMetric',
    # format_only=True,
    keep_results=True,
    output_dir='work_dirs/format_results')
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='../../data/cityscapes/',
        data_prefix=dict(img_path='driving_at_night_0.60_without_controlnet/train',seg_map_path='gtFine/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))