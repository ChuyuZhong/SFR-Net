import numpy as np


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    """Compute confusion matrix

    Args:
        x (np.array): 1 x h x w
            prediction array
        y (np.array): 1 x h x w
            groundtruth array
        n (int): number of classes
        ignore_label (int, optional): index of ignored label. Defaults to None.
        mask (np.array, optional): mask of regions that is needed to compute. Defaults to None.

    Returns:
        np.array: n x n
            confusion matrix
    """
    if mask is None:
        mask = np.ones_like(x)
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n)

def get_acc(confusion, dataset):
    """
    计算准确率

    :param confusion: 混淆矩阵
    :return: Accuracy
    """
    if dataset in ['fbps']:
        confusion = confusion[1:,1:]
    acc_per_class = confusion.diagonal() / confusion.sum(axis=1)
    return acc_per_class, np.sum(confusion.diagonal())/np.sum(confusion)


def get_p(confusion):
    """
    计算每类精确率

    :param confusion: 混淆矩阵
    :return: p
    """
    return confusion.diagonal()/np.sum(confusion,axis=1)

def get_r(confusion):
    """
    计算每类召回率

    :param confusion: 混淆矩阵
    :return: Rc
    """
    return confusion.diagonal()/np.sum(confusion,axis=0)

def get_f1(PC,RC):
    """
    计算F1 score

    :param PC: 精准率
    :param RC: 召回率
    :return: F1 score
    """
    return 2*np.multiply(PC,RC)/(PC+RC)


def getIoU(conf_matrix,dataset=None):
    """Compute IoU

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n,)
            IoU of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    if dataset in ['fbps']:
        conf_matrix = conf_matrix[1:,1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        union = np.maximum(1.0, conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        intersect = np.diag(conf_matrix)
        IU = np.nan_to_num(intersect / union)
    return IU


def getFreq(conf_matrix):
    """Compute frequentice of each class

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n, )
            frequentices of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    return freq


def get_mean_iou(conf_mat, dataset):
    """Get mean IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: mean IoU
    """
    IoU = getIoU(conf_mat,dataset)
    if dataset == "deepglobe":
        return np.nanmean(IoU[1:])
    elif dataset == "cityscapes":
        return np.nanmean(IoU)
    elif dataset == "inria_aerial":
        return np.nanmean(IoU)
    elif dataset == "blu":
        return np.nanmean(IoU[1:])
    elif dataset == "urur":
        return np.nanmean(IoU[:])
    elif dataset == "gid":
        return np.nanmean(IoU[:])
    elif dataset == "fbps":
        return np.nanmean(IoU[:])
    elif dataset == "loveda":
        return np.nanmean(IoU[:])
    elif dataset == "potsdam":
        return np.nanmean(IoU[1:])
    elif dataset == "glh":
        return np.nanmean(IoU)
    else:
        raise "Not implementation for dataset {}".format(dataset)


def get_freq_iou(conf_mat, dataset):
    """Get frequent IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: frequent IoU
    """
    IoU = getIoU(conf_mat)
    freq = getFreq(conf_mat)
    if dataset == "deepglobe":
        return (IoU[1:] * freq[1:]).sum() / freq[1:].sum()
    elif dataset == "cityscapes":
        return (IoU * freq).sum()


def get_overall_iou(conf_mat, dataset):
    """Get overall IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: overall iou
    """
    if dataset in ["deepglobe", "cityscapes" ,"inria_aerial", "blu", "urur","gid","fbps", "loveda", "potsdam","glh"]:
        return get_mean_iou(conf_mat, dataset)
    else:
        raise ValueError("Not implementation for dataset {}".format(dataset))
