import argparse
from pathlib import Path

import cv2
import numpy as np


PALETTES = {
    'gid': [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 255],
        [255, 255, 0],
        [0, 0, 255],
    ],
    'fbps': [
        [0, 0, 0],
        [200, 0, 0],
        [0, 200, 0],
        [150, 250, 0],
        [150, 200, 150],
        [200, 0, 200],
        [150, 0, 250],
        [150, 150, 250],
        [200, 150, 200],
        [250, 200, 0],
        [200, 200, 0],
        [0, 0, 200],
        [250, 0, 150],
        [0, 150, 200],
        [0, 200, 250],
        [150, 200, 250],
        [250, 250, 250],
        [200, 200, 200],
        [200, 150, 150],
        [250, 200, 150],
        [150, 150, 0],
        [250, 150, 150],
        [250, 150, 0],
        [250, 200, 250],
        [200, 150, 0],
    ],
    'inria_aerial': [
        [0, 0, 0],
        [255, 255, 255],
    ],
}


def get_palette(dataset):
    if dataset not in PALETTES:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return PALETTES[dataset]


def CLS2RGB(label, dataset):
    if label.ndim != 2:
        raise ValueError('label must be a two-dimensional class-index mask')
    colored = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(get_palette(dataset)):
        colored[label == class_id] = color
    return colored


def parse_args():
    parser = argparse.ArgumentParser(description='colorize prediction masks')
    parser.add_argument('--src', default='work_dirs/gid_predictions')
    parser.add_argument('--dst')
    parser.add_argument(
        '--dataset',
        default='gid',
        choices=tuple(PALETTES),
        help='dataset palette')
    return parser.parse_args()


def visualizer(src, dst, dataset):
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    completed = {path.name for path in dst_path.iterdir() if path.is_file()}

    for label_path in sorted(src_path.iterdir()):
        if not label_path.is_file():
            continue
        if label_path.name in completed:
            print('skip', label_path.name)
            continue
        print('start processing', label_path.name)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f'Unable to read label mask: {label_path}')
        colored = cv2.cvtColor(CLS2RGB(label, dataset), cv2.COLOR_RGB2BGR)
        output_path = dst_path / label_path.name
        if not cv2.imwrite(str(output_path), colored):
            raise OSError(f'Unable to save colored mask: {output_path}')

    print('Done!')


def main():
    args = parse_args()
    dst = args.dst if args.dst is not None else f'{args.src.replace("result","colored_result")}'
    visualizer(args.src, dst, args.dataset)


if __name__ == '__main__':
    main()
