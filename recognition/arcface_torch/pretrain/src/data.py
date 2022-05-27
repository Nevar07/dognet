import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader,RandomSampler, BatchSampler
from PIL import Image
import os

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]

class Valset(Dataset):
    def __init__(self,src_set,src_dict,dict):
        self.set = src_set
        self.dict = dict
        self.sd = src_dict

    def __getitem__(self,idx):
        (x,y) = self.set[idx]
        y = list(self.dict.keys())[list(self.dict.values()).index(y)]
        y = self.sd[y]
        return x,y

    def __len__(self):
        return len(self.set)

def load_data(args,input_size = 112,batch_size = 36):
    data_dir = args.data_dir
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.Grayscale(),
            transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
            transforms.RandomPerspective(distortion_scale=0.6,p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        
        ]) 
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    # 1-Large-Scale/train 2-Medium-Scale/train sorted
    
    train_set = datasets.ImageFolder(os.path.join(data_dir,'train'), data_transforms['train'])
    set = datasets.ImageFolder(os.path.join(data_dir,'val'), data_transforms['test'])
    val_set = Valset(set,train_set.class_to_idx,set.class_to_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
