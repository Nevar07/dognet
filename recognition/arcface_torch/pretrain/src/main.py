from cv2 import CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
import data
import models
import os
import argparse
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
from torch.autograd import Variable

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def get_cos_lr_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, num_cycles= 0.5, last_epoch= -1):
    def lr_lambda(current_epoch):
        # Warmup
        if current_epoch < num_warmup_epochs:
          return float(current_epoch) / float(max(1, num_warmup_epochs))
        # cos
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max( 0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def mixup_data(x, y, alpha, device):
    
    lam = np.random.beta(alpha, alpha) #beta分布采样lambda
    index = torch.randperm(x.size()[0]).to(device) #生成乱序index用于mixup
    
    mixed_x = lam * x + (1- lam) * x[index, :] #生成x
    y_a , y_b = y , y[index] #生成y
    
    return mixed_x , y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam): 
    # mixup后的loss function
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) 

def train_model(model, train_loader, valid_loader, criterion, optimizer, args):
    save_dir = args.save_dir
    num_epochs = args.num_epochs
    loss_dict = {'train':[],'val':[]}
    acc_dict = {'train':[],'val':[]}
    lr_record = []
    early_stop = 0
    
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            #mixup
            inputs, label_a, label_b , lam = mixup_data(inputs, labels, alpha = 0.5, device = 'cuda')
            inputs, label_a, label_b = map(Variable, (inputs, label_a, label_b))
            outputs = model(inputs)
            loss_fn = mixup_criterion(label_a, label_b, lam)
            loss = loss_fn(criterion, outputs)

            # outputs = model(inputs)
            # loss = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()
        

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        loss_dict['val'].append(epoch_loss)
        acc_dict['val'].append(epoch_acc.item())
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch+1, num_epochs))
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        loss_dict['train'].append(train_loss)
        acc_dict['train'].append(train_acc)
        print("training: Loss:{:.4f}, Acc:{:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))

        lr_record.append(optimizer.param_groups[0]['lr'])
        torch.save(loss_dict, os.path.join(save_dir, 'loss.dict'))
        torch.save(acc_dict, os.path.join(save_dir, 'acc.dict'))
        torch.save(lr_record, os.path.join(save_dir, 'lr.rec'))

        #writer.add_scalars
        # writer.add_scalar(tags[0],train_loss,epoch)
        # writer.add_scalar(tags[1],train_acc,epoch)
        # writer.add_scalar(tags[2],valid_loss,epoch)
        # writer.add_scalar(tags[3],valid_acc,epoch)
        # writer.add_scalar(tags[4],lr_record[-1],epoch) 


        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            if torch.cuda.device_count() > 1:
                torch.save(best_model.module.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            else:
                torch.save(best_model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f'best model saved at epoch {epoch+1}')
            early_stop = 0
        else:
            early_stop +=1

        if early_stop > 30: break
        print('*' * 100)

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='hw2')
    parser.add_argument('--data_dir', type=str,default='../', help='dataset path')
    parser.add_argument('--save_dir', type=str,default='../output/ct_1', help='model save path')
    parser.add_argument('--model', type=str,default='ct', help='type of model')
    parser.add_argument('--bs', type=int,default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, 
                        default=200, help='number of epochs')
    
    args = parser.parse_args()

    ## about model
    num_classes = 6000
    if not os.path.exists(args.save_dir):os.mkdir(args.save_dir)

    ## about data
    input_size = 224
    batch_size = args.bs

    ## data preparation
    train_loader, valid_loader = data.load_data(args, input_size=input_size, batch_size=batch_size)

    ## about training
    total_step = args.num_epochs * len(train_loader)
    warmup_step = int(total_step*0.01)
    lr = 0.05


    ## model initialization
    if args.model == 'ct':model = models.model_ct(num_classes=num_classes)
    if args.model == 'cb':model = models.model_cb(num_classes=num_classes)
    if args.model == 'vitb16':model = models.model_vitb16(num_classes=num_classes)
    if args.model == 'swtb': model = models.model_swtb(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    

    ## tensorboard init
    # writer = SummaryWriter(log_dir = os.path.join(args.save_dir, 'runs/vis'))
    # # init_img = torch.zeros((3,224,224),device = device)
    # # writer.add_graph(model, init_img)
    # tags = ['train_loss','train_acc','test_loss','test_acc','learning_rate']

    
   
   
    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = get_cos_lr_with_warmup(optimizer, warmup_step, total_step)

    
    
    
    ## loss function
    criterion= nn.CrossEntropyLoss()

    
    train_model(model, 
                train_loader, 
                valid_loader, 
                criterion,
                optimizer, 
                args, 
                )









