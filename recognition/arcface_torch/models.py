from torchvision import models
import torch.nn as nn
import torch



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

def model_vit(num_classes):
    model = models.vit_b_32(pretrained=True)
    num_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(num_features, num_classes)
    return model

class Block(nn.Module):
    
    def __init__(self,i_c,h_c,o_c):
        super(Block, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(i_c, h_c, 1),
            nn.BatchNorm2d(h_c),
            nn.ReLU(),
            nn.Conv2d(h_c, h_c, 3, 1, 1, groups=h_c),
            nn.BatchNorm2d(h_c),
            nn.ReLU(),
            nn.Conv2d(h_c, o_c, 1),
            nn.BatchNorm2d(o_c),
            )
        
        self.down = nn.Sequential(
            nn.Conv2d(i_c, i_c, 3, 1, 1, groups=i_c),
            nn.BatchNorm2d(i_c),
            nn.ReLU(),
            nn.Conv2d(i_c, o_c, 1),
            nn.BatchNorm2d(o_c),
            )
        self.relu = nn.ReLU()
           
    def forward(self,x):
        x_layer = self.layer(x)
        x_base = self.down(x)
        x = x_layer + x_base
        x = self.relu(x)
        
        return x
    
    
class Block2(nn.Module):
    
    def __init__(self,i_c,h_c):
        super(Block2, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(i_c, h_c, 1),
            nn.BatchNorm2d(h_c),
            nn.ReLU(),
            nn.Conv2d(h_c, h_c, 3, 1, 1, groups=h_c),
            nn.BatchNorm2d(h_c),
            nn.ReLU(),
            nn.Conv2d(h_c, i_c, 1),
            nn.BatchNorm2d(i_c),
            )

        self.relu = nn.ReLU()
        
    def forward(self,x):
            x_layer = self.layer(x)
            x = x_layer + x
            x = self.relu(x)
            
            return x



def model_B(num_classes):
         

    class Mymodel_B(nn.Module):
        
        def __init__(self):
            super(Mymodel_B, self).__init__()
                
            self.res_1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                )
            
            self.down1 = nn.Sequential(
                nn.Conv2d(3, 3, 3, 2, 1, groups=3),
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
                nn.Conv2d(3, 64, 1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2,2)
                )
            
            self.res2 = nn.Sequential(
                        Block(64,256,128),
                        nn.MaxPool2d(2,2)
                        )
            
            self.res3 = nn.Sequential(
                        Block(128,512,256),
                        nn.MaxPool2d(2,2)
                        )

            self.res4 = nn.Sequential(
                        Block(256,1024,512),
                        nn.MaxPool2d(2,2)
                        )
 
            self.res5 = Block(512,1024,1024)
            
            self.relu = nn.ReLU()    
                
            self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
            
            self.fc_layers = nn.Linear(1024, num_classes)
            
        def forward(self,x):

            x_res = self.res_1(x)
            x_base = self.down1(x)
            x = x_res + x_base
            x = self.relu(x)
            
            x = self.res2(x)
            
            x = self.res3(x)
            
            x = self.res4(x)
            
            x = self.res5(x)

            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc_layers(x)
            return x
    
    model = Mymodel_B()
    return model


def model_C(num_classes):
    
    class PSBatchNorm2d(nn.BatchNorm2d):
    
        def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
            super().__init__(num_features, eps, momentum, affine, track_running_stats)
            self.alpha = alpha
    
        def forward(self, x):
            return super().forward(x) + self.alpha
    
    
    class BasicBlockPreAct(nn.Module):
        def __init__(
                self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False
            ):
            super(BasicBlockPreAct, self).__init__()
            self.bn1 = PSBatchNorm2d(in_chan, momentum=0.001)
            self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.conv1 = nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.bn2 = PSBatchNorm2d(out_chan, momentum=0.001)
            self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
            self.conv2 = nn.Conv2d(
                out_chan,
                out_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.downsample = None
            if in_chan != out_chan or stride != 1:
                self.downsample = nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=stride, bias=False
                )
            self.pre_res_act = pre_res_act
            self.init_weight()
    
        def forward(self, x):
            bn1 = self.bn1(x)
            act1 = self.relu1(bn1)
            residual = self.conv1(act1)
            residual = self.bn2(residual)
            residual = self.relu2(residual)
            if not self.dropout is None:
                residual = self.dropout(residual)
            residual = self.conv2(residual)
    
            shortcut = act1 if self.pre_res_act else x
            if self.downsample is not None:
                shortcut = self.downsample(shortcut)
    
            out = shortcut + residual
            return out
    
        def init_weight(self):
            for _, md in self.named_modules():
                if isinstance(md, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if not md.bias is None: nn.init.constant_(md.bias, 0)
    
    
    
    class WideResnetBackbone(nn.Module):
        def __init__(self, k=1, n=28, drop_rate=0):
            super(WideResnetBackbone, self).__init__()
            self.k, self.n = k, n
            assert (self.n - 4) % 6 == 0
            n_blocks = (self.n - 4) // 6
            n_layers = [16,] + [self.k*16*(2**i) for i in range(3)]
    
            self.conv1 = nn.Conv2d(
                3,
                n_layers[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.layer1 = self.create_layer(
                n_layers[0],
                n_layers[1],
                bnum=n_blocks,
                stride=1,
                drop_rate=drop_rate,
                pre_res_act=True,
            )
            self.layer2 = self.create_layer(
                n_layers[1],
                n_layers[2],
                bnum=n_blocks,
                stride=2,
                drop_rate=drop_rate,
                pre_res_act=False,
            )
            self.layer3 = self.create_layer(
                n_layers[2],
                n_layers[3],
                bnum=n_blocks,
                stride=2,
                drop_rate=drop_rate,
                pre_res_act=False,
            )
            self.bn_last = PSBatchNorm2d(n_layers[3], momentum=0.001)
            self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.init_weight()
    
        def create_layer(
                self,
                in_chan,
                out_chan,
                bnum,
                stride=1,
                drop_rate=0,
                pre_res_act=False,
            ):
            layers = [
                BasicBlockPreAct(
                    in_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=stride,
                    pre_res_act=pre_res_act),]
            for _ in range(bnum-1):
                layers.append(
                    BasicBlockPreAct(
                        out_chan,
                        out_chan,
                        drop_rate=drop_rate,
                        stride=1,
                        pre_res_act=False,))
            return nn.Sequential(*layers)
    
        def forward(self, x):
            feat = self.conv1(x)
    
            feat = self.layer1(feat)
            feat2 = self.layer2(feat) # 1/2
            feat4 = self.layer3(feat2) # 1/4
    
            feat4 = self.bn_last(feat4)
            feat4 = self.relu_last(feat4)
            return feat2, feat4
    
        def init_weight(self):
            for _, child in self.named_children():
                if isinstance(child, nn.Conv2d):
                    n = child.kernel_size[0] * child.kernel_size[0] * child.out_channels
                    nn.init.normal_(child.weight, 0, 1. / ((0.5 * n) ** 0.5))
                    if not child.bias is None: nn.init.constant_(child.bias, 0)
    
    
    class WideResnet(nn.Module):
        
        def __init__(self, num_classes, k=2, n=28):
            super(WideResnet, self).__init__()
            self.n_layers, self.k = n, k
            self.backbone = WideResnetBackbone(k=k, n=n)
            self.classifier = nn.Linear(64 * self.k, num_classes, bias=True)
    
        def forward(self, x):
            feat = self.backbone(x)[-1]
            feat = torch.mean(feat, dim=(2, 3))
            feat = self.classifier(feat)
            return feat
    
        def init_weight(self):
            nn.init.xavier_normal_(self.classifier.weight)
            if not self.classifier.bias is None:
                nn.init.constant_(self.classifier.bias, 0)
    
    
    model = WideResnet(num_classes)
    
    return model


def model_D(num_classes):
    class Mymodel_D(nn.Module):
        
        def __init__(self):
            super(Mymodel_D, self).__init__()
                
            self.res_1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                )
            
            self.down1 = nn.Sequential(
                nn.Conv2d(3, 3, 3, 2, 1, groups=3),
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
                nn.Conv2d(3, 64, 1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2,2)
                )
            self.res1_2 = nn.Sequential(
                        Block2(64,128),
                        nn.MaxPool2d(2,2)
                        )
            
            self.res2 = Block(64,256,128)
            self.res2_2 = nn.Sequential(
                        Block2(128,256),
                        nn.MaxPool2d(2,2)
                        )     
            
            self.res3 = Block(128,512,256)
            self.res3_2 = nn.Sequential(
                        Block2(256,512),
                        nn.MaxPool2d(2,2)
                        )

            self.res4 = Block(256,1024,512)
            self.res4_2 = nn.Sequential(
                        Block2(512,1024),
                        nn.MaxPool2d(2,2)
                        )
 
            self.res5 = Block(512,1024,1024)
            
            self.relu = nn.ReLU()    
                
            self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
            
            self.fc_layers = nn.Linear(1024, num_classes)
            
        def forward(self,x):

            x_res = self.res_1(x)
            x_base = self.down1(x)
            x = x_res + x_base
            x = self.relu(x)
            x = self.res1_2(x)
            
            x = self.res2(x)
            x = self.res2_2(x)
            
            x = self.res3(x)
            x = self.res3_2(x)
            
            x = self.res4(x)
            x = self.res4_2(x)
            
            x = self.res5(x)

            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc_layers(x)
            return x
    
    model = Mymodel_D()
    return model

if __name__ == '__main__':
    model = model_A(6000)
    print(model)
    
    from torchsummary import summary
    
    summary(model, (3, 224, 224), device="cpu")
