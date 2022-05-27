import argparse

import cv2
import numpy as np
import torch

from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
from tqdm import tqdm


@torch.no_grad()

def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        #img = img.replace('*','_')
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    
    net.eval()
    feat = net(img).numpy()
    return(feat)

def cal_cossim(net,img_a, img_b):
    a = inference(net,  os.path.join(args.img_dir,'images',img_a))
    b = inference(net,  os.path.join(args.img_dir,'images',img_b))
    cossim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    return cossim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='models/model.pt')
    parser.add_argument('--img_dir', type=str, default='data/val')
    parser.add_argument('--out_name', type=str, default='valid.csv')
    parser.add_argument('--threshold', type=float, default=0.15)
    args = parser.parse_args()

    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()
    res = []
    
    with open('./data/val/valid_data.csv') as f, open(os.path.join(args.img_dir, args.out_name),'w', encoding='utf-8', newline='') as wf:
        val_csv = csv.reader(f)
        tsv_w = csv.writer(wf)
        tsv_w.writerow(['imageA', 'imageB', 'prediction'])
        n = 0
        for line in tqdm(val_csv):
            if 'imageA' not in line:
                pred = cal_cossim(net,line[0],line[1])
                line.append(pred[0][0])
                tsv_w.writerow(line)
                n+=1
                if n > 4:break
        f.close()
        wf.close()

