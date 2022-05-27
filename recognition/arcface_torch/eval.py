import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
from tqdm import tqdm


@torch.no_grad()

def inference(net, img):
    
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    
    net.eval()
    feat = net(img).numpy()
    return(feat)

def cal_cossim(net,path,img_a, img_b):
    a = inference(net,  os.path.join(path,'images',img_a))
    b = inference(net,  os.path.join(path,'images',img_b))
    cossim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    return cossim

def eval(net, path):
    path = path.split('train')+'eval'
    net.eval()
    gt = []
    out = []
    with open(os.path.join(path,'eval.csv')) as f:
        val_csv = csv.reader(f)
        for line in val_csv:
            out.append(cal_cossim(net,path,line[0], line[1]))
            gt.append(line(2))
    criterion= nn.CrossEntropyLoss()

    loss = criterion(torch.Tensor(out).cuda(),torch.Tensor(gt).cuda())
    return loss.item()
