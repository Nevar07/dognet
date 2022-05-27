from re import T
import torch
import math
import random
import numpy as np
from collections import defaultdict


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise        

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class Contrastive(torch.nn.Module):
    def __init__(self,m):
        super(Contrastive, self).__init__()
        self.m = m

    def ed(self,a,b):
        m,n = a.size(0),b.size(0)
        sq_a = torch.pow(a,2).sum(1, keepdim=True).expand(m,n)
        sq_b = torch.pow(b,2).sum(1, keepdim=True).expand(n,m).t()
        dist = sq_a + sq_b
        dist.addmm_(1,-2,a,b.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # sq_a = a**2
        # sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        # sq_b = b**2
        # sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        # bt = b.t()


        # a = torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
        # a = torch.sqrt((sum_sq_a+sum_sq_b-2*a.mm(bt))/l)
        dist = dist.squeeze(0)
        return dist.squeeze(0)

    def pair(self, labels):
        labels_np = labels.cpu().numpy()
        total_num = labels_np.shape[0]
        if total_num % 2 != 0: total_num -= 1
        #pair_idx [[idx1,idx2,i(=0 when diff/ =1 when same)],...]
        pair_idx = []
        idx = []
        for i in range(total_num):
            idx.append(i)

        labels_list = labels_np.tolist()
        dct = defaultdict(list)
        for key, value in [(v, i) for i, v in enumerate(labels_list)]:
            dct[key].append(value)
        for label in dct:
            if len(dct[label]) > 1:
                a = random.sample(dct[label], 2)
                if a[0] in idx and a[1] in idx:
                    for i in a:
                        idx.remove(i)
                    a.append(1)
                    pair_idx.append(a)

        while idx != []:
            a = random.sample(idx, 2)
            for i in a:
                idx.remove(i)
            a.append(0)
            pair_idx.append(a)
            
        return pair_idx
        
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        pair_idx = self.pair(labels)
        num = len(pair_idx)
        total_loss = 0
        for line in pair_idx:
            em1,em2,i= embeddings[line[0]],embeddings[line[1]],line[2]
            d =self.ed(em1.reshape(1, -1),em2.reshape(1, -1))
            loss_it = (1-i) * (max(0,self.m-d))**2 + i*d**2
            total_loss += loss_it

        loss = total_loss / (num+1e-10)
        # print(loss.item())
        
        return loss
