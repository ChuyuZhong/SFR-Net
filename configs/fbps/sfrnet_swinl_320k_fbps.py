_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/fbps.py', '../_base_/schedules/schedule_320k.py']
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53]*3,
    std=[58.395, 57.12, 57.375]*3,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,)
num_classes = 25
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SFRNet',
        depth=[4,18,18],
        local_crop_size=(512,512),),
    neck=dict(
        type='CascadedCrossScaleFusion',
        main_channels=1536,
        sub_channels=[512,512],),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='DiceLoss', use_sigmoid=False, loss_weight=5.0)]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1536,
        in_index=4,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,
        by_epoch=False,
    )
]


train_dataloader = dict(batch_size=4,num_workers=64)
val_dataloader = dict(batch_size=4,num_workers=64)
test_dataloader = val_dataloader
