import os
import torch
import argparse
from models import *
from data_loader import *
from torch.backends import cudnn
from test import test

device = torch.device('cuda:0')
cudnn.benchmark = True

def str2bool(s):
    return s.lower() == 'true'

# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='shoes')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_sep', type=int, default=20)
    parser.add_argument('--flag', type=str, default='sbir')
    parser.add_argument('--data_root', type=str, default='data/QUML_v2')

    parser.add_argument('--phase', type=str, default='train_from_scratch')
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--loss_type', type=str, default='triplet')
    parser.add_argument('--loss_ratio', type=str, default='1.0')
    parser.add_argument('--model_type', type=str, default='deep_sbir')
    parser.add_argument('--fix_bn', type=str2bool, default=False)

    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--hard_ratio', type=float, default=0.75)
    parser.add_argument('--test_complex', type=str2bool, default=True)
    parser.add_argument('--test_verbose', type=str2bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    config = parser.parse_args()

    config.model_path = os.path.join('logs', 'model', '%s_%s.cpkt'%(config.obj, config.flag))
    config.log_path = os.path.join('logs', 'log', '%s_%s.txt'%(config.obj, config.flag))
    config.loss_type = config.loss_type.split(',')
    config.loss_ratio = [eval(r) for r in config.loss_ratio.split(',')]
    config.device = device
    config.distance = 'sq'
    config.model_type = config.model_type.lower()
    assert len(config.loss_type) == len(config.loss_ratio)

    return config

def main():    

    # config
    config = get_parameter()

    # model
    edge = True
    if config.model_type == 'densenet':
        model = DenseNet(f_dim=config.feat_dim, norm=False)
        edge = False
    elif config.model_type == 'resnext':
        model = Resnext100(f_dim=config.feat_dim, norm=False)
        edge = False        
    elif config.model_type == 'densenet_norm':
        model = DenseNet(f_dim=config.feat_dim, norm=True)
        edge = False
    elif config.model_type == 'densenet_nopretrain':
        model = DenseNet(f_dim=config.feat_dim, norm=False, pretrained=False)
        edge = False
    elif config.model_type.startswith('deep_sbir'):
        model = SketchANet({'pretrain':not config.model_type.endswith('nopretrain')})
        config.feat_dim = 256
    elif config.model_type.startswith('dssa'):
        model = SketchANet({'dssa':True, 'pretrain':not config.model_type.endswith('nopretrain')})
        config.feat_dim = 512
    model.to(device)
    #model = torch.nn.DataParallel(model)
    
    if config.phase == 'train_from_scratch':
        #model.load_state_dict(torch.load(config.model_path))
        pass
    elif config.phase.startswith('train_from_sketchy'):
        path = os.path.join('logs', 'model', 'pretrained_densenet_%d.cpkt'%config.feat_dim)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        model.param_range = config.phase[19:]
        print('load pretrained densenet + '+model.param_range)

    # dataset

    if config.obj.endswith('v2'):
        data = SbirData_v2(config.data_root, edge=edge)
    elif config.obj == 'sketchy':
        data = SketchyData(config.data_root, edge=edge)
        config.y_dim = len(data.trainset_cate)
    elif config.obj.startswith('hairstyle'):
        if config.obj.endswith('complex'):
            data = HairData(config.data_root, 'complex', edge=edge)
        else:
            data = HairData(config.data_root, edge=edge)
        config.y_dim = len(data.trainset_cate)
    else:
        data = SbirData(config.obj, hard_ratio=config.hard_ratio, edge=edge)
        if 'rd' in config.obj:
            data.set_mode(config.obj.split('_',1)[1])
        if config.obj == 'shoes':
            config.y_dim = 21
        if config.obj == 'chairs':
            config.y_dim = 15

    config.c_dim = len(data)


    # main
    if config.phase.startswith('train'):
        if config.phase.endswith('continue'):
            print('loading ...')
            model.load_state_dict(torch.load(config.model_path))
        train(model, data, config)
    elif config.phase.startswith('test'):
        print('testing ...')

        accu, accu_complex= test(model, data, config, verbose=True)
        info = 'TESTING'
        if accu:
            for key, value in accu.items():
                info = info + '\nsimple - ' + key + ':' + '{:.4f}'.format(value)
        if accu_complex:
            for key, value in accu_complex.items():
                info = info + '\nmulti-view - ' + key + ':' + '{:.4f}'.format(value)
        print(info)



def train(model, data, config=None):

    losses = dict()
    # creterion
    if 'triplet' in config.loss_type:
        t_loss = TripletLoss(config.margin)
        losses['triplet'] = [config.loss_ratio[config.loss_type.index('triplet')], 0]
    else:
        t_loss = None

    if 'sphere' in config.loss_type:
        s_loss = SphereLoss(config=config)
        losses['sphere'] = [config.loss_ratio[config.loss_type.index('sphere')], 0]
    else:
        s_loss = None

    if 'attribute' in config.loss_type:
        if config.obj in ['sketchy', 'hairstyle']:
            a_loss = ClassificationLoss(config.feat_dim, config.y_dim)
        else:
            a_loss = AttributeLoss(config=config)
        losses['attribute'] = [config.loss_ratio[config.loss_type.index('attribute')], 0]
    else:
        a_loss = None

    if 'centre' in config.loss_type:
        c_loss = CentreLoss(num_classes=len(data), feat_dim=config.feat_dim)
        losses['centre'] = [config.loss_ratio[config.loss_type.index('centre')], 0]
    else:
        c_loss = None

    if 'softmax' in config.loss_type:
        sf_loss = ClassificationLoss(config.feat_dim, config.c_dim)
        losses['softmax'] = [config.loss_ratio[config.loss_type.index('softmax')], 0]
    else:
        sf_loss = None

    if 'hofel' in config.loss_type:
        h_loss = HofelLoss(config.feat_dim, config.margin)
        losses['hofel'] = [config.loss_ratio[config.loss_type.index('hofel')], 0]
        config.distance = h_loss.distance
    else:
        h_loss = None


    print(losses)

    # data loader
    loader = data.loader(batch_size=config.batch_size, num_workers=config.batch_size//4, shuffle=True)

    # optimizor
    lr, decay = config.lr, config.weight_decay
    opts = dict()
    if config.model_type.startswith('deep_sbir') or config.model_type.startswith('dssa'):
        opts['net'] = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=decay)
    else:
        opts['net'] = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    if 'sphere' in losses.keys():
        opts['sphere'] = torch.optim.Adam(s_loss.parameters(), lr=lr, weight_decay=decay)
    if 'softmax' in losses.keys():
        opts['softmax'] = torch.optim.Adam(sf_loss.parameters(), lr=lr, weight_decay=decay)
    if 'attribute' in losses.keys():
        opts['attribute'] = torch.optim.Adam(a_loss.parameters(), lr=lr, weight_decay=decay)
    if 'centre' in losses.keys():
        opts['centre'] = torch.optim.SGD(c_loss.parameters(), lr=0.5)
    if 'hofel' in losses.keys():
        #opts['hofel'] = torch.optim.SGD(h_loss.parameters(), lr=0.05)
        opts['hofel'] = torch.optim.Adam(h_loss.parameters(), lr=lr, betas=(0.5,0.999))
    print(opts)


    # log
    f = open(config.log_path, 'a+')
    best_accu = {}

    # training
    for epoch in range(2000):
        for i, [skts, img1s, img2s, idxs, attrs] in enumerate(loader):
            print(skts.shape, img1s.shape, img2s.shape, skts.min(), skts.max(), img1s.min(), img1s.max())
            exit(0)

            skts, img1s, img2s, idxs, attrs = skts.to(device), img1s.to(device), img2s.to(device), idxs.to(device), attrs.to(device)

            feat_skts, feat_img1s, feat_img2s = model(torch.cat([skts, img1s, img2s])).split(skts.size(0))

            #feat_skts = model(skts)
            #feat_img1s = model(img1s)
            #feat_img2s = model(img2s)
           
            triplet_loss = t_loss(feat_skts, feat_img1s, feat_img2s) if t_loss else 0
            hofel_loss = h_loss(feat_skts, feat_img1s, feat_img2s) if h_loss else 0
            sphere_loss = s_loss(feat_skts, idxs[:,0])+s_loss(feat_img1s, idxs[:,1])+s_loss(feat_img2s, idxs[:,2]) if s_loss else 0
            softmax_loss = sf_loss(feat_skts, idxs[:,0])+sf_loss(feat_img1s, idxs[:,1])+sf_loss(feat_img2s, idxs[:,2]) if sf_loss else 0
            if config.obj in ('sketchy', 'hairstyle'):
                attribute_loss = a_loss(feat_skts, attrs[:, 0]) + a_loss(feat_img1s, attrs[:, 1]) + a_loss(
                    feat_img2s, attrs[:, 2]) if a_loss else 0
            else:
                attribute_loss = a_loss(feat_skts, attrs[:,0,:])+a_loss(feat_img1s, attrs[:,1,:])+a_loss(feat_img2s, attrs[:,2,:]) if a_loss else 0
            centre_loss = c_loss(feat_skts, idxs[:,0])+c_loss(feat_img1s, idxs[:,1])+c_loss(feat_img2s, idxs[:,2]) if c_loss else 0
            loss = 0
            for key in losses.keys():
                l = eval(key+'_loss')
                losses[key][1] = l.detach().item()
                loss += l * losses[key][0]

            for opt in opts.values():
                opt.zero_grad()
            loss.backward()
            for key, opt in opts.items():
                if key == 'centre':
                    for param in opt.param_groups:
                        # I don't know what is wrong with centre loss
                        continue
                        for p in param['params']:
                        	p.grad.data *= (1.0/losses[key][0])
                opt.step()
            
        info = 'e:{}'.format(epoch+1)
        for key in losses.keys():
            info += ', '
            info += key[0]+'l: '
            info += '{:.4f}'.format(losses[key][1])

        print('\r'+info, end='')

        if (epoch+1) % config.epoch_sep == 0:

            accu, accu_complex = test(model, data, config, config.test_verbose)
            #best_accu = {}
            info = config.flag

            if accu:
                info = info + '\nsimple - '
                for key, value in accu.items():
                    info = info + key + ':' + '{:.4f}'.format(value) + '\t'
                    best_accu['simple - '+key] = max(best_accu.get('simple - '+key, 0), value)

            if accu_complex:
                info = info + '\nmulti-view - '
                for key, value in accu_complex.items():
                    info = info + key + ':' + '{:.4f}'.format(value) + '\t'
                    best_accu['complex - '+key] = max(best_accu.get('complex - '+key, 0), value)

            info = info + '\nbest:'
            for key, value in best_accu.items():
                info = info + '\t{:.4f}'.format(value)
                
            print('\n' + info)
            f.write(info+'\n')
            f.flush()

        torch.save(model.state_dict(), config.model_path)




if __name__ == "__main__":
    main()


