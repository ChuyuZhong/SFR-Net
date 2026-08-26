dataset_type = 'InriaAerialDataset'
data_root = '/mnt/dataset/zhongchuyu/inria_aerial'
crop_size = (512, 512)
local_crop_size = (512, 512)
distances = [1, 3, 10]

train_pipeline = [
    dict(
        type='LoadScaleFrustumFromFile',
        local_crop_size=local_crop_size,
        distances=distances,
        use_global=True,
        load_type='random'),
    dict(type='LoadScaleFrustumAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(
        type='LoadScaleFrustumFromFile',
        local_crop_size=local_crop_size,
        distances=distances,
        use_global=True,
        load_type='center_sfr'),
    dict(type='Resize', scale=local_crop_size, keep_ratio=True),
    dict(type='LoadScaleFrustumAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='Resize', scale=local_crop_size, keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='Label/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='Label/val'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='Label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
