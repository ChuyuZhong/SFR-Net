# dataset settings
dataset_type = 'FBPSDataset'
data_root = 'data/GID'
crop_size = (512,512)
local_crop_size = (512, 512)
distances = [1, 3, 14]
train_pipeline = [
    dict(type='LoadScaleFrustumFromFile', local_crop_size=local_crop_size,
         distances=distances, use_global=True, load_type='random'),
    dict(type='LoadScaleFrustumAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadScaleFrustumFromFile', local_crop_size=local_crop_size,
         distances=distances, use_global=True, load_type='random'),
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
            img_path='Image_train', seg_map_path='annos_train_24l'),
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
            img_path='Image_test',
            seg_map_path='annos_test_24l'),
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
            img_path='Image_test',
            seg_map_path='annos_test_24l'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
