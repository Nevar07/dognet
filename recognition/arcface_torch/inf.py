import argparse

import cv2
import numpy as np
import torch

from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import csv
import os
from tqdm import tqdm
import torch.nn as nn
import models
from torchvision import  transforms
from PIL import Image

@torch.no_grad()
def inference(net, cls,img):

    # img = img.replace('*','_')
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    # img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    # img = Image.open(img)
    # trans = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # img = trans(img)

    img = img.cuda()

    net.eval()
    feat = net(img).cpu().numpy()
    # feat = cls(net(img)).cpu().numpy()
    return (feat)


def cal_cossim(net, cls,img_a, img_b):
    a = inference(net,cls, os.path.join(args.img_dir, 'images', img_a))
    b = inference(net, cls,os.path.join(args.img_dir, 'images', img_b))
    cossim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    # ed = euclidean_distances(a.reshape(1, -1), b.reshape(1, -1))
    return (cossim+1)/2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='vit_b', help='backbone network')
    parser.add_argument('--weight', type=str, default='models/vitb40/model.pt')
    parser.add_argument('--cls_weight', type=str, default='best_cls.pt45')
    parser.add_argument('--img_dir', type=str, default='data/val')
    parser.add_argument('--out_name', type=str, default='valid_vitb.csv')
    parser.add_argument('--threshold', type=float, default=0.15)
    args = parser.parse_args()

    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))

    cls = nn.Linear(512, 6000)
    # cls_weight = args.weight[:-13] + 'cls/'+args.cls_weight
    # cls.load_state_dict(torch.load(cls_weight))


    net.eval()
    cls.eval()
    net,cls = net.cuda(),cls.cuda()
    res = []

    with open('./data/val/valid_data.csv') as f, open(os.path.join(args.img_dir, args.out_name), 'w', encoding='utf-8',
                                                      newline='') as wf:
        val_csv = csv.reader(f)
        tsv_w = csv.writer(wf)
        tsv_w.writerow(['imageA', 'imageB', 'prediction'])
        for line in tqdm(val_csv):
            if 'imageA' not in line:
                pred = cal_cossim(net,cls, line[0], line[1])
                line.append(pred[0][0])
                tsv_w.writerow(line)
        f.close()
        wf.close()



