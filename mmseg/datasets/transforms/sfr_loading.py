
import warnings
from pathlib import Path
from typing import Dict, Optional, Union
import time
import mmcv
import random
import mmengine.fileio as fileio
import cv2
import mxnet as mx
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS
from mmseg.utils import datafrombytes

try:
    from osgeo import gdal
except ImportError:
    gdal = None

@TRANSFORMS.register_module()
class LoadScaleFrustumFromFile(BaseTransform):


    def __init__(self,
                 local_crop_size: tuple = (512, 512),
                 distances: list = [1, 3, 6, 10],
                 use_global: bool = True,
                 load_type: str = 'random',
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 ignore_last: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.local_crop_size = local_crop_size
        self.distances = distances
        self.use_global = use_global
        self.load_type = load_type
        self.ignore_empty = ignore_empty
        self.ignore_last = ignore_last
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()


    def transform(self, results: dict) -> Optional[dict]:


        filename = results['img_path']
        start_time = time.time()
        try:
            img = mx.image.imread(filename, flag=1).asnumpy()

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)


        H,W,C = img.shape
        h,w = self.local_crop_size

        assert self.load_type in ['random','origin','global','local','center','center_sfr']

        sfr_windows = []
        if self.load_type == 'random' or self.load_type == 'center_sfr':
            results['prp_h'] = random.randint(0,H-h)
            results['prp_w'] = random.randint(0,W-w)
            if self.load_type == 'center_sfr':
                results['prp_h'] = int(H/2-h/2)
                results['prp_w'] = int(W/2-w/2)

            for i in range(len(self.distances)):
                if i < len(self.distances)-1:
                    window_h = int(h * self.distances[i])
                    window_w = int(w * self.distances[i])
                    start_h = int(results['prp_h']*(self.distances[-1]-self.distances[i])/(self.distances[-1]-1))
                    start_w = int(results['prp_w']*(self.distances[-1]-self.distances[i])/(self.distances[-1]-1))
                    sfr_windows.append(cv2.resize(img[start_h:start_h+window_h,start_w:start_w+window_w,:].copy(),dsize=self.local_crop_size))

                else:
                    if self.ignore_last:
                        break

                    else:
                        if self.use_global:
                            sfr_windows.append(cv2.resize(img.copy(),dsize=self.local_crop_size))
                        else:
                            sfr_windows.append(cv2.resize(img[:min(h * self.distances[-1],H),:min(w * self.distances[-1],W),:].copy(),dsize=self.local_crop_size))

        elif self.load_type == 'origin':
            sfr_windows.append(img)

        elif self.load_type == 'global':
            sfr_windows.append(cv2.resize(img.copy(),dsize=self.local_crop_size))

        elif self.load_type == 'center':
            sfr_windows.append(img[int(H/2-h/2):int(H/2+h/2),int(W/2-w/2):int(W/2+w/2),:].copy())

        elif self.load_type == 'local':
            results['prp_h'] = random.randint(0,H-h)
            results['prp_w'] = random.randint(0,W-w)
            window_h = h
            window_w = w
            start_h = results['prp_h']
            start_w = results['prp_w']
            sfr_windows.append(img[start_h:start_h+window_h,start_w:start_w+window_w,:].copy())

        else :
            raise Exception('Sorry, no such type accomplished.')

        sfr = np.concatenate(sfr_windows,axis=2).squeeze()

        results['img'] = sfr
        results['load_type'] = self.load_type
        results['local_crop_size'] = self.local_crop_size
        results['img_shape'] = sfr.shape[:2]
        results['ori_shape'] = sfr.shape[:2]
        results['distances'] = self.distances
        results['whole_shape'] = img.shape[:2]

        end_time = time.time()
        return results


    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadScaleFrustumAnnotations(MMCV_LoadAnnotations):


    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:


        start_time = time.time()
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)


        if results['load_type'] == 'random' or results['load_type'] == 'local' or results['load_type'] == 'center_sfr':


            gt_semantic_seg = gt_semantic_seg[results['prp_h']:results['prp_h']+results['local_crop_size'][0],\
                                              results['prp_w']:results['prp_w']+results['local_crop_size'][1]]

        elif results['load_type'] == 'global':
            gt_semantic_seg = cv2.resize(gt_semantic_seg,dsize=results['local_crop_size'])

        elif results['load_type'] == 'center':
            gt_semantic_seg = gt_semantic_seg[int(gt_semantic_seg.shape[0]/2-results['local_crop_size'][0]/2):int(gt_semantic_seg.shape[0]/2+results['local_crop_size'][0]/2),\
                                              int(gt_semantic_seg.shape[1]/2-results['local_crop_size'][1]/2):int(gt_semantic_seg.shape[1]/2+results['local_crop_size'][1]/2)]


        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:

            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        if results.get('label_map', None) is not None:


            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        end_time = time.time()


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str