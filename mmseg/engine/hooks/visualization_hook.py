
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class SegVisualizationHook(Hook):


    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:

            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:


        if self.draw is False:
            return


        total_curr_iter = runner.iter + batch_idx


        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}'

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:


        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)
