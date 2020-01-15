import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *

class TripletLoss(nn.Module):
    def __init__(self, delta=0.3, dist='sq'):
        super(TripletLoss, self).__init__()

        self.delta = delta
        if dist == 'sqr':
            dist = lambda z1, z2: torch.norm(z1-z2, p=2, dim=1)
        elif dist == 'sq':
            dist = lambda z1, z2: torch.norm(z1-z2, p=2, dim=1) ** 2
        self.dist = dist

    def forward(self, s, pp, pn):

        dp = self.dist(s, pp)
        dn = self.dist(s, pn)
        dist = self.delta+dp-dn
        dist = torch.clamp(dist, min=0.0)

        return torch.mean(dist)


class HofelLoss(nn.Module):
    def __init__(self, feat_dim, delta=0.3, lamb=0.0005):
        super(HofelLoss, self).__init__()

        self.eye = torch.diag(torch.ones(feat_dim)).cuda()
        self.w = Parameter(self.eye.clone() + torch.randn(feat_dim, feat_dim).cuda() * 0.05)
        self.delta = delta
        self.lamb = lamb

    def distance(self, vec1, vec2, train=False):
        assert vec1.shape == vec2.shape

        if train:
            w = self.w
        else:
            w = self.w.data.cpu()

        d = len(vec1.shape)
        f = vec1.shape[-1]
        vec1 = vec1.unsqueeze(d).repeat(*[1]*d, f)
        vec2 = vec2.unsqueeze(d).repeat(*[1]*d, f).transpose(d-1, d)

        diff = (vec1 - vec2)
        dist = (diff * diff * w).sum(dim=d).sum(dim=d-1)

        return dist

    def forward(self, skt, imgp, imgn):
        dp = self.distance(skt, imgp, True)
        dn = self.distance(skt, imgn, True)
        dist = self.delta+dp-dn
        dist = torch.clamp(dist, min=0.0).mean()

        regu = (self.w - self.eye).abs().sum() + (self.w - self.eye).pow(2).sum().sqrt()

        loss = dist + self.lamb * regu
        return loss






class SphereLoss(nn.Module):
    def __init__(self, gamma=0, config=None):
        super(SphereLoss, self).__init__()

        self.fc = AngleLinear(config.feat_dim, config.c_dim).cuda()

        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = self.fc(input)
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.detach() * 0.0 #size=(B,Classnum)
        index.scatter_(1,target,1)
        index = index.byte()

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()
        
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss      


class CentreLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CentreLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # one-hot
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class AttributeLoss(nn.Module):
    def __init__(self, config=None):
        super(AttributeLoss, self).__init__()

        self.fc = nn.Linear(config.feat_dim, config.y_dim).cuda()

    def forward(self, input, target):
        logits = self.fc(input)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        return loss  

class ClassificationLoss(nn.Module):
    def __init__(self, feat_dim, c_dim):
        super(ClassificationLoss, self).__init__()

        self.fc = nn.Linear(feat_dim, c_dim).cuda()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logits = self.fc(input)
        loss = self.loss(logits, target)

        return loss