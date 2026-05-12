import os
import cv2
import numpy as np
import argparse

def get_palette(dataset):
    if dataset == 'deepglobe':
        palette=[[0, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    elif dataset == 'inria_aerial':
        palette=[[0, 0, 0], [255, 255, 255]]
    elif dataset == 'blu':
        palette=[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125]]
    elif dataset == 'urur':
        palette=[[255, 255, 255], [230, 230, 230], [100, 100, 100], [200, 230, 160], [95, 163, 7], [255, 255, 100], [150, 200, 250], [240, 100, 80]]
    elif dataset == 'gid':
        palette=[[0,0,0],[255,0,0],[0,255,0],[0,255,255],[255,255,0],[0,0,255]]
    elif dataset == 'fbps':
        palette=[[0,0,0], [200,0,0],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],[200,150,200],[250,200,0],[200,200,0],[0,0,200],[250,0,150],[0,150,200],[0,200,250],[150,200,250],[250,250,250],[200,200,200],[200,150,150],[250,200,150],[150,150,0],[250,150,150],[250,150,0],[250,200,250],[200,150,0]]
    elif dataset == 'loveda':
        palette=[[0,0,0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
    elif dataset == 'potsdam':
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    elif dataset == 'fbps_hard':
        palette=[
            [0,0,0],         # 'unlabeled'
            [150,200,150],   # 'dry cropland' (原色)
            [200,0,200],     # 'garden land' (原色)
            [150,150,250],   # 'shrub forest' (原色)
            [0,200,250],     # 'pond' (原色)
            [150,150,0]      # 'square' (原色)
        ]
    elif dataset == 'glh':
        palette=[[0, 0, 0], [255, 255, 255]]
    
    return palette

def CLS2RGB(label,dataset):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.uint8)
    palette = get_palette(dataset)
    for i in range(len(palette)):
        indices = np.where(label == i)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = palette[i]
    return colmap

def parse_args():
    parser = argparse.ArgumentParser(description='visualize a dataset')
    parser.add_argument('--src',default='./uhr_res/label/pspnet_r50_4xb16_inria_aerial_s128',help='save path')
    parser.add_argument('--dst',default=None, help='colored label save files')
    parser.add_argument('--dataset',default='gid', help='dataset name')

    args = parser.parse_args()
    return args

def visualizer(src,dst,dataset):
    os.makedirs(dst,exist_ok=True)
    for file in sorted(os.listdir(src)):
        if file in os.listdir(dst):
            print("skip",file)
            continue
        print("start processing",file)
        label = cv2.imread(os.path.join(src,file),cv2.IMREAD_GRAYSCALE)
        colored_label = cv2.cvtColor(CLS2RGB(label,dataset),cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dst,file),colored_label)
        
    print('Done!')


def single_CLS2RGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.uint8)
    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]]
    for i in range(len(palette)):
        indices = np.where(label == i)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = palette[i]
    return colmap

def single_visualizer():
    label_file_name = "data/example_label.png"
    label = cv2.imread(label_file_name, cv2.IMREAD_GRAYSCALE)
    colored_label = single_CLS2RGB(label)
    cv2.imwrite("./single_visualize.png", colored_label)

def main():
    args = parse_args()
    if args.dst == None:
        args.dst = args.src.replace("result","colored_result")
    os.makedirs(args.dst,exist_ok=True)
    visualizer(args.src,args.dst,args.dataset)
    # single_visualizer()


if __name__ =='__main__':
    main()
