import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load image metadata from an existing ``results['img']`` ndarray."""

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
