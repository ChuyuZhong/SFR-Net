import argparse
import os
import time

import cv2
import mxnet as mx
import numpy as np

from mmseg.apis import inference_model, init_model

DATASETS = {
    'gid': {
        'src': '/mnt/dataset/zhongchuyu/GID/Image_test',
        'dst': 'work_dirs/gid_predictions',
        'config': 'configs/gid/sfrnet_swinl_320k_gid.py',
        'distances': (1, 3, 14),
        'num_classes': 6,
    },
    'fbps': {
        'src': '/mnt/dataset/zhongchuyu/GID/Image_test',
        'dst': 'work_dirs/fbps_predictions',
        'config': 'configs/fbps/sfrnet_swinl_320k_fbps.py',
        'distances': (1, 3, 14),
        'num_classes': 25,
    },
    'inria_aerial': {
        'src': '/mnt/dataset/zhongchuyu/inria_aerial/images/test',
        'dst': 'work_dirs/inria_aerial_predictions',
        'config': (
            'configs/inria_aerial/sfrnet_swinl_320k_inria_aerial.py'),
        'distances': (1, 3, 10),
        'num_classes': 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scale-Frustum Representation inference for UWA images')
    parser.add_argument(
        '--dataset', default='gid', choices=tuple(DATASETS),
        help='dataset preset')
    parser.add_argument(
        '--src',
        help='folder containing ultra-wide area images')
    parser.add_argument(
        '--dst',
        help='folder to save prediction masks')
    parser.add_argument(
        '--config',
        help='SFR-Net config path')
    parser.add_argument('--ckpt', required=True, help='checkpoint path')
    parser.add_argument('--stride', default=128, type=int, help='sliding stride')
    parser.add_argument(
        '--load-type', default='random', choices=['random', 'local', 'global'],
        help='scale-frustum construction mode')
    parser.add_argument(
        '--reverse', action='store_true',
        help='process input files in reverse lexical order')
    return parser.parse_args()


def build_scale_frustum(img,
                        prp_h,
                        prp_w,
                        local_crop_size,
                        distances,
                        use_global=True,
                        load_type='random'):

    height, width, _ = img.shape
    crop_h, crop_w = local_crop_size
    sfr_windows = []

    if load_type == 'random':
        for idx, distance in enumerate(distances):
            if idx < len(distances) - 1:
                window_h = int(crop_h * distance)
                window_w = int(crop_w * distance)
                start_h = int(prp_h * (distances[-1] - distance) /
                              (distances[-1] - 1))
                start_w = int(prp_w * (distances[-1] - distance) /
                              (distances[-1] - 1))
                window = img[start_h:start_h + window_h,
                             start_w:start_w + window_w, :].copy()
                sfr_windows.append(cv2.resize(window, dsize=local_crop_size))
            elif use_global:
                sfr_windows.append(cv2.resize(img.copy(), dsize=local_crop_size))
            else:
                window = img[:min(crop_h * distances[-1], height),
                             :min(crop_w * distances[-1], width), :].copy()
                sfr_windows.append(cv2.resize(window, dsize=local_crop_size))

    elif load_type == 'local':
        sfr_windows.append(img[prp_h:prp_h + crop_h,
                               prp_w:prp_w + crop_w, :].copy())
    else:
        sfr_windows.append(cv2.resize(img.copy(), dsize=local_crop_size))

    return np.concatenate(sfr_windows, axis=2).squeeze()


def sfr_infer_single(img,
                     model,
                     stride,
                     local_crop_size,
                     distances,
                     num_classes,
                     load_type='random'):
    assert load_type in ['random', 'local', 'global']
    start_time = time.time()

    if load_type in ['random', 'local']:
        logits_sum = np.zeros(
            (img.shape[0], img.shape[1], num_classes), dtype=np.float32)

        for h in range(0, img.shape[0], stride):
            for w in range(0, img.shape[1], stride):
                prp_h = min(h, img.shape[0] - local_crop_size[0])
                prp_w = min(w, img.shape[1] - local_crop_size[1])

                model.cfg.test_pipeline = [
                    dict(type='LoadImageFromNDArray', to_float32=False),
                    dict(type='Resize', keep_ratio=True, scale=local_crop_size),
                    dict(type='PackSegInputs'),
                ]

                img_sfr = build_scale_frustum(
                    img, prp_h, prp_w, local_crop_size, distances,
                    use_global=True, load_type=load_type)
                result = inference_model(model, img_sfr)
                pred_logits = result.seg_logits.data.cpu().numpy().astype(
                    np.float32)
                logits_sum[prp_h:prp_h + local_crop_size[0],
                           prp_w:prp_w + local_crop_size[1], :] += (
                               pred_logits.transpose(1, 2, 0))

        pred = np.argmax(logits_sum, axis=2)
    else:
        img_sfr = cv2.resize(img.copy(), dsize=local_crop_size)
        result = inference_model(model, img_sfr)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(
            pred_mask, dsize=(img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    return pred, time.time() - start_time


def sfr_infer(src,
              dst,
              config_path,
              checkpoint_path,
              distances,
              stride,
              num_classes,
              load_type='random',
              reverse=False):
    os.makedirs(dst, exist_ok=True)
    model = init_model(config_path, checkpoint_path)
    file_list = sorted(os.listdir(src), reverse=reverse)
    total_infer_time = 0.0

    for file_name in file_list:
        out_name = file_name.replace('.tif', '.png').replace('.jpg', '.png')
        out_path = os.path.join(dst, out_name)
        if os.path.exists(out_path):
            print('skip', file_name)
            continue

        print('start processing', file_name)
        img = mx.image.imread(os.path.join(src, file_name), flag=1).asnumpy()
        src_h, src_w = img.shape[:2]
        target_h = ((src_h + 511) // 512) * 512
        target_w = ((src_w + 511) // 512) * 512
        if (target_h, target_w) != (src_h, src_w):
            padded = np.zeros((target_h, target_w, img.shape[2]),
                              dtype=img.dtype)
            padded[:src_h, :src_w] = img
            img = padded

        pred, infer_time = sfr_infer_single(
            img, model, stride, (512, 512), distances, num_classes, load_type)
        total_infer_time += infer_time
        cv2.imwrite(out_path, pred[:src_h, :src_w])

    print('Total inference time:', total_infer_time)
    if total_infer_time > 0:
        print('Average FPS:', len(file_list) / total_infer_time)
    print('Done!')


def main():
    args = parse_args()
    preset = DATASETS[args.dataset]
    src = args.src or preset['src']
    dst = args.dst or preset['dst']
    config_path = args.config or preset['config']
    sfr_infer(
        src, dst, config_path, args.ckpt, preset['distances'], args.stride,
        preset['num_classes'], args.load_type, args.reverse)


if __name__ == '__main__':
    main()
