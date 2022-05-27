from torchvision import models
import torch.nn as nn
import torch
from swt import build_model


def model_A(num_classes):
    model_resnet = models.convnext_tiny(pretrained=True)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_ct(num_classes):
    model = models.convnext_tiny(pretrained=True)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    return model

def model_cb(num_classes):
    model = models.convnext_base(pretrained=True)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    return model

def model_vitb16(num_classes):
    model = models.vit_b_16(pretrained=True)
    num_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(num_features, num_classes)
    return model

def model_vitb32(num_classes):
    model = models.vit_b_32(pretrained=True)
    num_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(num_features, num_classes)
    return model

def model_swtb(num_classes):
    model_type = 'swin'
    model = build_model(model_type)
    model.load_state_dict(torch.load('./weights/swt_base.pth'))
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    return model


if __name__ == '__main__':
    model = model_swtb(6000)
    print(model)
    
    from torchsummary import summary
    
    summary(model, (3, 224, 224), device="cpu")
