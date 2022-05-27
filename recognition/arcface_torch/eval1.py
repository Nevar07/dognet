import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


@torch.no_grad()
def inference(net, img,eval_data):
    net.eval()
    img = eval_data[img]
    with torch.no_grad():
        out = net(img).cpu().detach()
        feat = out.numpy()
    return(feat)

def cal_cossim(net,img_a, img_b,eval_data):
    a = inference(net, img_a,eval_data)
    b = inference(net, img_b,eval_data)
    cossim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    torch.cuda.empty_cache()
    return (cossim+1)/2


def arc_eval(network, path, eval_data):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load('models/model.pt'))

    path = path.strip('train')
    net.eval()
    net = net.cuda()
    gt = []
    out = []
    with torch.no_grad():

        with open(os.path.join(path,'eval.csv')) as f:
            val_csv = csv.reader(f)
            n=0
            for line in val_csv:
                if 'imageA' not in line:
                    out.append(cal_cossim(net,line[0], line[1],eval_data)[0][0])
                    gt.append(int(line[2]))

            out, gt = np.asarray(out), np.asarray(gt)

            loss = mean_squared_error(gt, out)
            auc = roc_auc_score(gt ,out)

    return loss, auc
