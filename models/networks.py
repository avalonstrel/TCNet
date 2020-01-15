
import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import densenet169
from .model_utils import *
import pretrainedmodels

class Resnext100(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Resnext100, self).__init__()

        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(2048, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat    

class DenseNet(torch.nn.Module):
    def __init__(self, f_dim=512, pretrained=True, **kwargs):
        super(DenseNet, self).__init__()
        
        self.model = densenet169(pretrained=pretrained, num_classes=f_dim)
        self.norm = kwargs.get('norm', True)
        self.param_range = 'all'
        
    def forward(self, x, mode_flag=None):

        if mode_flag == None:
            feat = self.model(x, mode='pho')
        elif mode_flag == 'connect':
            raise NotImplementedError
            n = x.size(0) // 3
            f_s = self.model(x[:n], mode='skt')
            f_p = self.model

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat

    def parameters(self):
        all_blocks = [self.model.features_pho, self.model.features_input, self.model.features.denseblock1,
                      self.model.features.denseblock2, self.model.features.denseblock3, self.model.features.denseblock4,
                      self.model.classifier]
        if self.param_range == 'all':
            pass
        elif self.param_range == 'mfc':
            all_blocks = all_blocks[-1:]
        elif self.param_range == 'm4':
            all_blocks = all_blocks[-2:]
        elif self.param_range == 'm3':
            all_blocks = all_blocks[-3:]

        params = []
        for block in all_blocks:
            params += list(block.parameters())

        return params

class SketchANet(nn.Module):
    def __init__(self, args={}):
        super(SketchANet, self).__init__()

        self.args = args

        self.conv1 = nn.Conv2d(1, 64, kernel_size=15, stride=3, padding=0)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=2)
        curr_dim = 256 * 7 * 7

        att_mode = args.get('dssa', False)
        if att_mode:
            self.attention = AttentionNet('softmax')

        self.fc6 = nn.Linear(curr_dim, 512)
        self.act6 = nn.ReLU()

        self.fc7 = nn.Linear(512, 256)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if args.get('pretrain', True):
            print('loading pretrained model...')
            self.load_state_dict(torch.load('logs/model/sketchnet_init.cpkt'), strict=False)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool3(self.act5(self.conv5(x)))

        if hasattr(self, 'attention'):
            x = self.attention(x)
            h = F.avg_pool2d(x, kernel_size=x.size(2))
            h = h.squeeze(3).squeeze(2)
            h = F.normalize(h, dim=1, p=2)

        x = x.view(x.size(0), -1)
        x = self.fc7(self.act6(self.fc6(x)))
        x = F.normalize(x, dim=1, p=2)

        if hasattr(self, 'attention'):
            x = torch.cat([x, h], dim=1)

        return x


    def parameters(self):
        update_params = []
        update_params += list(self.fc6.parameters())
        update_params += list(self.fc7.parameters())
        if hasattr(self, 'attention'):
            update_params += list(self.conv5.parameters())
            update_params += list(self.attention.parameters())
        return update_params
