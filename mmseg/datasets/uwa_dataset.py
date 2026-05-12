# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class GIDDataset(BaseSegDataset):
    """GID dataset.

    please set num_classes=6 and reduce_zero_label=False
    In segmentation map annotation, 0 is the unlabeled class.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('unlabeled', 'built-up', 'farmland', 'forest', 'meadow', 'water'),
        palette=[[0,0,0],[255,0,0],[0,255,0],[0,255,255],[255,255,0],[0,0,255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class FBPSDataset(BaseSegDataset):
    """FBPS dataset.

    please set num_classes=25 and reduce_zero_label=False    
    In segmentation map annotation, 0 is the unlabeled class.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('unlabeled', 'industrial area', 'paddy field', 'irrigated field', 'dry cropland', 'garden land', 'arbor forest', 'shrub forest', 'park', 'natural meadow', 'artificial meadow', 'river', 'urban residential', 'lake', 'pond', 'fish pond', 'snow', 'bareland', 'rural residential', 'stadium', 'square', 'road', 'overpass', 'railway station', 'airport'),
        palette=[[0,0,0], [200,0,0],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],[200,150,200],[250,200,0],[200,200,0],[0,0,200],[250,0,150],[0,150,200],[0,200,250],[150,200,250],[250,250,250],[200,200,200],[200,150,150],[250,200,150],[150,150,0],[250,150,150],[250,150,0],[250,200,250],[200,150,0]])
    

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
