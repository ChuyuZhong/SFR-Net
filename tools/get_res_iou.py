import os 
import numpy as np
import cv2
import time
import argparse
from metrics import get_overall_iou , getIoU , get_mean_iou , get_p , get_r , get_f1, get_acc
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='use a segmentor to inference folder')
    parser.add_argument('--dataset', default='gid', choices=['gid', 'fbps'],
                        help='dataset name')
    parser.add_argument('--pred', required=True, help='prediction folder')
    parser.add_argument('--gt', default=None, help='ground-truth folder')
    
    args = parser.parse_args()
    return args

def get_res_iou(pred_dir,gt_dir,num_classes,dataset,labels,decimal=1):
    conf_mat = np.zeros((num_classes,num_classes), dtype=np.int64)
    count = 0
    for file in sorted(os.listdir(gt_dir)):
        if file.replace('.jpg','.png').replace('.tif','.png').replace('_mask','_sat') not in os.listdir(pred_dir):
            print('file not found:',file)
            continue
        print('calculating',file)
        count += 1
        gt = cv2.imread(os.path.join(gt_dir,file),cv2.IMREAD_GRAYSCALE)
        h,w = gt.shape
        # gt.reshape((1,h,w))
        pred = cv2.imread(os.path.join(pred_dir,file.replace('.jpg','.png').replace('.tif','.png').replace('_mask','_sat')),cv2.IMREAD_GRAYSCALE)
        # pred.reshape((1,h,w))
        tmp_mat = confusion_matrix(gt.flatten(), pred.flatten(),labels=labels)
        # print(tmp_mat)
        conf_mat += tmp_mat
        print("tmp IoU:",get_overall_iou(tmp_mat,dataset))
        print("mean IoU:",get_overall_iou(conf_mat,dataset))
        print('')

    print("-"*30)
    acc_per_class, acc = get_acc(conf_mat, dataset)
    p = get_p(conf_mat)
    r = get_r(conf_mat)
    f1 = get_f1(get_p(conf_mat),get_r(conf_mat))
    mean_f1 = np.mean(f1[1:])
    iou = getIoU(conf_mat,dataset)
    mean_iou = get_overall_iou(conf_mat,dataset)

    print(conf_mat)
    print("Accuracy:",np.round(100*acc,decimal))
    print("Accuracy of each classes",np.round(100*acc_per_class,decimal))
    print("Precision of each classes",np.round(100*p,decimal))
    print("Recall of each classes",np.round(100*r,decimal))
    print("f1 score of each classes",np.round(100*f1,decimal))
    print("mean f1:",np.round(100*mean_f1,decimal))
    print("IoU of each classes",np.round(100*iou,decimal))
    print("mean IoU:",np.round(100*mean_iou,decimal))



def main():
    args = parse_args()
    if args.dataset == 'gid':
        gt = args.gt or 'data/GID/annos_test_5l'
        labels = [0,1,2,3,4,5]
        num_classes = 6
    elif args.dataset == 'fbps':
        gt = args.gt or 'data/GID/annos_test_24l'
        labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        num_classes = 25
    get_res_iou(args.pred,gt,num_classes,args.dataset,labels)


if __name__ == '__main__':
    main()
