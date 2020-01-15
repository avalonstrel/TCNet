import torch
import os
from .utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


class SbirData_v2(Dataset):
    def __init__(self, root_path, obj='shoes', crop_size=225, edge=False):

        self.crop_size = crop_size
        self.edge = edge

        if not edge:
            self.photo_path = os.path.join(root_path, 'ShoeV2_photo')
        else:
            self.photo_path =os.path.join(root_path, 'ShoeV2_edge_eb')
        self.sketch_path = os.path.join(root_path, 'ShoeV2_sketch')

        # load train/test set
        with open(os.path.join(root_path, 'photo_train.txt'), 'r') as f:
            train_photos = f.read().split('\n')
            train_photos = list(filter(lambda x: x.strip(), train_photos))

        with open(os.path.join(root_path, 'sketch_train.txt'), 'r') as f:
            train_sketch = f.read().split('\n')
            train_sketch = list(filter(lambda x: x.strip(), train_sketch))
            train_sketch = list(map(lambda x: x.split('.')[0] + '.png', train_sketch))

        self.train_set = []
        for pho in train_photos:
            fn = pho.split('.')[0]
            sns = list(filter(lambda x: x.startswith(fn), train_sketch))
            self.train_set.append([pho, sns])

        with open(os.path.join(root_path, 'photo_test.txt'), 'r') as f:
            test_photos = f.read().split('\n')
            test_photos = list(filter(lambda x: x.strip(), test_photos))

        with open(os.path.join(root_path, 'sketch_test.txt'), 'r') as f:
            test_sketch = f.read().split('\n')
            test_sketch = list(filter(lambda x: x.strip(), test_sketch))
            test_sketch = list(map(lambda x: x.split('.')[0] + '.png', test_sketch))

        self.test_set = []
        for pho in test_photos:
            fn = pho.split('.')[0]
            sns = list(filter(lambda x: x.startswith(fn), test_sketch))
            self.test_set.append([pho, sns])

        self.all_neg = [[j for j in range(len(self)) if j != i] for i in range(len(self))]

        self.resize = transforms.Resize(256)
        self.to_tensor = transforms.ToTensor()

        self.crop = crop_gen(self.crop_size)

        self.mode = 'triplet'

        self.prepare_test()


    def __len__(self):
        return len(self.train_set)

    def set_mode(self, mode):
        assert mode in ['triplet', 'pair', 'triplet_rand']
        self.mode = mode

    def __getitem__(self, index):
        if self.mode.startswith('triplet'):
            skt, imgs, idx, attr = self.get_triplet(index)
        elif self.mode.startswith('pair'):
            skt, imgs, idx, attr = self.get_pair(index)

        imgs = [randomflip(self.crop(img)) for img in imgs]
        skt = randomflip(self.crop(skt))

        if self.edge:
            skt = skt[0:1] * 255 - 250.42
            imgs = list(map(lambda img: img *255 - 250.42, imgs))

        return (skt, *imgs, idx, attr)

    def get_pair(self, index):
        img, skts = self.train_set[index]
        skt = random.choice(skts)

        skt = np.array(self.resize(Image.open(os.path.join(self.sketch_path, skt))))[:, :, 3]
        skt = torch.FloatTensor(np.array(skt < 50, dtype='float')).unsqueeze(0).repeat(3, 1, 1)
        img = self.to_tensor(self.resize(Image.open(os.path.join(self.photo_path, img))))

        idxs = torch.LongTensor([index, index])
        attrs = 0

        return skt, [img], idxs, attrs

    def get_triplet(self, index):

        img_idx1 = index
        img_idx2 = random.choice(self.all_neg[index])

        img1, skts = self.train_set[index]
        skt = random.choice(skts)
        img2 = self.train_set[img_idx2][0]

        skt = np.array(self.resize(Image.open(os.path.join(self.sketch_path, skt))))[:, :, 3]
        skt = torch.FloatTensor(np.array(skt < 50, dtype='float')).unsqueeze(0).repeat(3, 1, 1)
        img1 = self.to_tensor(self.resize(Image.open(os.path.join(self.photo_path, img1))))
        img2 = self.to_tensor(self.resize(Image.open(os.path.join(self.photo_path, img2))))

        idxs = torch.LongTensor([index, img_idx1, img_idx2])
        attrs = 0
        return skt, [img1, img2], idxs, attrs

    def get_test(self, complex=False):

        if not complex:
            return (self.crop(self.test_skts, 'center'), self.crop(self.test_imgs, 'center'), self.test_idxs)
        elif complex and hasattr(self, 'test_data_complex'):
            return self.test_data_complex

        skts = []
        imgs = []
        for mode in ['center', 'upleft', 'upright', 'downleft', 'downright']:
            skts.append(self.crop(self.test_skts, mode))
            imgs.append(self.crop(self.test_imgs, mode))

        skts = torch.cat(skts)
        imgs = torch.cat(imgs)
        skts = torch.cat([skts, randomflip(skts, p=1)], dim=0)
        imgs = torch.cat([imgs, randomflip(imgs, p=1)], dim=0)

        self.test_data_complex = (skts, imgs, self.test_idxs)
        return self.test_data_complex

    def prepare_test(self):

        f_skts, f_phos, idxs = [], [], []
        for i, (pho, skts) in enumerate(self.test_set):
            f_phos.append(os.path.join(self.photo_path, pho))
            for skt in skts:
                f_skts.append(os.path.join(self.sketch_path, skt))
                idxs.append(i)

        self.test_imgs = torch.stack(list(map(lambda x: self.to_tensor(self.resize(Image.open(x))), f_phos))) * 255 - 250.42
        f_skts = list(map(lambda x: np.array(self.resize(Image.open(x)))[:, :, 3], f_skts))
        self.test_skts = torch.stack(list(
            map(lambda x: torch.FloatTensor(np.array(x < 50, dtype='float')).unsqueeze(0).repeat(3, 1, 1), f_skts)))
        if self.edge:
        	self.test_skts = self.test_skts[:,0:1,:,:] * 255 - 250.42

        self.test_idxs = torch.LongTensor(idxs)

    def loader(self, **args):
        return DataLoader(dataset=self, **args)



