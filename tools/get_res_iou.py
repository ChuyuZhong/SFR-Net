import argparse
import os
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    'gid': ('/mnt/dataset/zhongchuyu/GID/annos_test_5l', 6),
    'fbps': ('/mnt/dataset/zhongchuyu/GID/annos_test_24l', 25),
    'inria_aerial': ('/mnt/dataset/zhongchuyu/inria_aerial/Label/test', 2),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='use a segmentor to inference folder')
    parser.add_argument(
        '--dataset', default='gid', choices=tuple(DATASETS), help='dataset name')
    parser.add_argument(
        '--pred',
        default=str(REPO_ROOT / 'work_dirs' / 'gid_predictions'),
        help='save path')
    return parser.parse_args()


def confusion_matrix(y_true, y_pred, labels):
    labels = np.asarray(labels)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sort_order = np.argsort(labels)
    sorted_labels = labels[sort_order]
    valid = np.isin(y_true, labels) & np.isin(y_pred, labels)
    true_indices = sort_order[np.searchsorted(sorted_labels, y_true[valid])]
    pred_indices = sort_order[np.searchsorted(sorted_labels, y_pred[valid])]
    size = len(labels)
    return np.bincount(
        size * true_indices + pred_indices,
        minlength=size * size).reshape(size, size)


def get_acc(confusion, dataset):
    if dataset == 'fbps':
        confusion = confusion[1:, 1:]
    acc_per_class = confusion.diagonal() / confusion.sum(axis=1)
    acc = np.sum(confusion.diagonal()) / np.sum(confusion)
    return acc_per_class, acc


def get_p(confusion):
    return confusion.diagonal() / np.sum(confusion, axis=1)


def get_r(confusion):
    return confusion.diagonal() / np.sum(confusion, axis=0)


def get_f1(precision, recall):
    return 2 * np.multiply(precision, recall) / (precision + recall)


def getIoU(conf_matrix, dataset=None):
    if conf_matrix.sum() == 0:
        return 0
    if dataset == 'fbps':
        conf_matrix = conf_matrix[1:, 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        union = np.maximum(
            1.0,
            conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0)
            - np.diag(conf_matrix))
        intersect = np.diag(conf_matrix)
        iou = np.nan_to_num(intersect / union)
    return iou


def get_mean_iou(conf_mat, dataset):
    iou = getIoU(conf_mat, dataset)
    if dataset in {'gid', 'fbps', 'inria_aerial'}:
        return np.nanmean(iou)
    raise ValueError(f'Not implementation for dataset {dataset}')


def get_overall_iou(conf_mat, dataset):
    if dataset not in {'gid', 'fbps', 'inria_aerial'}:
        raise ValueError(f'Not implementation for dataset {dataset}')
    return get_mean_iou(conf_mat, dataset)


def get_res_iou(pred_dir, gt_dir, num_classes, dataset, labels, decimal=5):
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred_files = set(os.listdir(pred_dir))
    for file_name in sorted(os.listdir(gt_dir)):
        pred_name = file_name.replace('.jpg', '.png').replace(
            '.tif', '.png').replace('_mask', '_sat')
        if pred_name not in pred_files:
            print('file not found:', file_name)
            continue
        print('calculating', file_name)
        gt = cv2.imread(
            os.path.join(gt_dir, file_name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(
            os.path.join(pred_dir, pred_name), cv2.IMREAD_GRAYSCALE)
        tmp_mat = confusion_matrix(gt.flatten(), pred.flatten(), labels)
        conf_mat += tmp_mat
        print('tmp IoU:', get_overall_iou(tmp_mat, dataset))
        print('mean IoU:', get_overall_iou(conf_mat, dataset))
        print('')

    print('-' * 30)
    acc_per_class, acc = get_acc(conf_mat, dataset)
    precision = get_p(conf_mat)
    recall = get_r(conf_mat)
    f1 = get_f1(precision, recall)
    mean_f1 = np.mean(f1[1:])
    iou = getIoU(conf_mat, dataset)
    mean_iou = get_overall_iou(conf_mat, dataset)

    print(conf_mat)
    print('Accuracy:', np.round(100 * acc, decimal))
    print('Accuracy of each classes', np.round(100 * acc_per_class, decimal))
    print('Precision of each classes', np.round(100 * precision, decimal))
    print('Recall of each classes', np.round(100 * recall, decimal))
    print('f1 score of each classes', np.round(100 * f1, decimal))
    print('mean f1:', np.round(100 * mean_f1, decimal))
    print('IoU of each classes', np.round(100 * iou, decimal))
    print('mean IoU:', np.round(100 * mean_iou, decimal))


def main():
    args = parse_args()
    assert args.dataset in DATASETS, 'dataset not implemented yet'
    gt, num_classes = DATASETS[args.dataset]
    labels = list(range(num_classes))
    get_res_iou(args.pred, gt, num_classes, args.dataset, labels)


if __name__ == '__main__':
    main()
