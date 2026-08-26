from typing import Sequence

import mmcv
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from numpy import random

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):


    def _flip(self, results: dict) -> None:
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        for key in results.get('seg_fields', []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(
                    results[key], direction=results['flip_direction']).copy()
                results['swap_seg_labels'] = self.swap_seg_labels


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):


    def _resize_seg(self, results: dict) -> None:
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[seg_key] = gt_seg


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):


    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: float = 1,
                beta: float = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def sfr_saturation(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            processed_channels = []
            for i in range(img.shape[2] // 3):
                img_split = mmcv.bgr2hsv(img[:, :, i * 3:(i + 1) * 3])
                img_split[:, :, 1] = self.convert(
                    img_split[:, :, 1],
                    alpha=random.uniform(self.saturation_lower,
                                         self.saturation_upper))
                processed_channels.append(mmcv.hsv2bgr(img_split))
            img = np.concatenate(processed_channels, axis=2)
        return img

    def sfr_hue(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            processed_channels = []
            for i in range(img.shape[2] // 3):
                img_split = mmcv.bgr2hsv(img[:, :, i * 3:(i + 1) * 3])
                img_split[:, :, 0] = (
                    img_split[:, :, 0].astype(int) +
                    random.randint(-self.hue_delta, self.hue_delta)) % 180
                processed_channels.append(mmcv.hsv2bgr(img_split))
            img = np.concatenate(processed_channels, axis=2)
        return img

    def transform(self, results: dict) -> dict:
        img = self.brightness(results['img'])

        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        img = self.sfr_saturation(img)
        img = self.sfr_hue(img)

        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results
