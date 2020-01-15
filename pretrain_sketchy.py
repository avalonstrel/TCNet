import torch
from data_loader import SketchyData
from models import *
from torch.backends import cudnn
import os
import argparse

device = torch.device('cuda:0')
cudnn.benchmark = True

def str2bool(s):
    return s.lower() == 'true'

# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--epoch_sep', type=int, default=10)
    parser.add_argument('--feat_dim', type=int, default=512)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--sketchy_root', type=str, required=True)

    config = parser.parse_args()
    return config


def main():
    # config
    config = get_parameter()

    # model
    model = DenseNet(f_dim=config.feat_dim, norm=False)
    model.to(device)
    # model = torch.nn.DataParallel(model)

    # dataset
    data = SketchyData(config.sketchy_root)
    data.set_mode('pretrain')
    config.y_dim = len(data.trainset_cate)

    # load
    if os.path.exists(os.path.join('logs', 'model', 'pretrained_densenet_%d.cpkt'%config.feat_dim)):
        model.load_state_dict(torch.load(os.path.join('logs', 'model', 'pretrained_densenet_%d.cpkt'%config.feat_dim)))

    # main
    train(model, data, config)

    # save
    torch.save(model.state_dict(), os.path.join('logs', 'model', 'pretrained_densenet_%d.cpkt'%config.feat_dim))


def train(model, data, config):
    loss_fn = ClassificationLoss(config.feat_dim, config.y_dim)
    opt = torch.optim.Adam([*model.parameters(), *loss_fn.parameters()], lr=config.lr, weight_decay=config.weight_decay)
    loader = data.loader(batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size//4)
    iTer = iter(loader)


    for e in range(config.epochs):
        try:
            xs, ys = next(iTer)
        except Exception:
            iTer = iter(loader)
            xs, ys = next(iTer)

        xs, ys = xs.to(device), ys.to(device)

        feat = model(xs)
        loss = loss_fn(feat, ys[:,0])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (e+1) % config.epoch_sep == 0:
            info = 'epoch:{}, loss:{:.4f}'.format((e+1), loss.item())
            print('\r%s'%info, end='')

        if (e+1) % (config.epochs//10) == 0:
            lr = config.lr * (1-(e+1)/config.epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            torch.save(model.state_dict(),
                       os.path.join('logs', 'model', 'pretrained_densenet_%d.cpkt' % config.feat_dim))

    print('\ncomplete pretraining')

if __name__ == '__main__':
    main()
