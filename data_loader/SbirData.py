import torch
import os
from .utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from scipy.io import loadmat
import random

class SbirData(Dataset):
    def __init__(self, obj='shoes', crop_size=225, hard_ratio=1, edge=False):
 
        obj = obj.split('_')[0]

        root_path = os.path.join('data', 'QUML_v1', obj)

        self.hard_ratio = hard_ratio
        self.crop_size = crop_size

        # get trainset/testset/triplets
        with open(os.path.join(root_path, 'annotation', '%s_annotation.json'%obj), 'r') as f:
            self.annotation = json.load(f)


        if not edge:
            self.train_imgs = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_image_db_train.mat'%obj))['data'].transpose(0,3,1,2)).float()/255
            self.test_imgs = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_image_db_test.mat' % obj))['data'].transpose(0, 3, 1,2)).float() / 255
            self.train_skts = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_sketch_db_train.mat'%obj))['data']).unsqueeze(1).float().repeat(1,3,1,1)/255
            self.test_skts = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_sketch_db_test.mat'%obj))['data']).unsqueeze(1).float().repeat(1,3,1,1)/255
        else:
            self.train_imgs = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_edge_db_train.mat' % obj))['data']).unsqueeze(1).float() - 250.42
            self.train_skts = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_sketch_db_train.mat' % obj))['data']).unsqueeze(1).float() - 250.42
            self.test_imgs = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_edge_db_test.mat' % obj))['data']).unsqueeze(1).float() - 250.42
            self.test_skts = torch.from_numpy(loadmat(os.path.join(root_path, 'dbs', '%s_sketch_db_test.mat' % obj))['data']).unsqueeze(1).float() - 250.42

        # get label
        if obj in ('shoes', 'chairs'):
            label_key = 'label' if obj == 'shoes' else 'labels'
            self.img_labels = torch.FloatTensor(loadmat(os.path.join(root_path, 'annotation', 'image_label.mat'))[label_key])
            self.skt_labels = torch.FloatTensor(loadmat(os.path.join(root_path, 'annotation', 'sketch_label.mat'))[label_key])
        else:
            self.img_labels = self.skt_labels = torch.zeros(len(self.train_imgs))


        self.train_triplets = self.annotation['train']['triplets']
        self.num_triplets = len(self.train_triplets[0])
        pos_list = [merge_list(triplets) for triplets in self.train_triplets]
        total = set(range(len(self)))
        self.easy_neg = [list(total.difference(pos_list[j])) for j in range(len(self))]
        self.all_neg = [[j for j in range(len(self)) if j != i] for i in range(len(self))]

        self.mode = 'anno'

        self.test_idxs = torch.arange(len(self.test_imgs))
        self.crop = crop_gen(self.crop_size)

    def __len__(self):
        return len(self.train_imgs)

    def set_mode(self, mode):
        assert mode == 'anno' or mode.startswith('rd')
        self.mode = mode
        if self.mode.startswith('rd'):
            num = eval(self.mode.split('_')[1])
            self.rd_neg = [random.sample(negs, num) for negs in self.all_neg]

    def __getitem__(self, index):
        #if self.mode[:7] == 'triplet':
        if True:
            skt, imgs, idx, attr = self.get_triplet(index)
        else:
            skt, imgs, idx, attr = self.get_pair(index)

        imgs = [randomflip(self.crop(img)) for img in imgs]
        skt = randomflip(self.crop(skt))

        return (skt, *imgs, idx, attr)


    def get_triplet(self, index):

        if self.mode == 'anno':
            if self.hard_ratio == 0:
                img_idx1 = index
                img_idx2 = random.choice(self.all_neg[index])
            else:
                if random.random() < self.hard_ratio:
                    idx = random.randint(0, self.num_triplets-1)
                    img_idx1, img_idx2 = self.train_triplets[index][idx]
                else:
                    img_idx1 = index
                    img_idx2 = random.choice(self.easy_neg[index])
        elif self.mode.startswith('rd'):
            #print(len(self.rd_neg), len(self.rd_neg[0]))
            img_idx1 = index
            img_idx2 = random.choice(self.rd_neg[index])


        skt = self.train_skts[index]
        img1 = self.train_imgs[img_idx1]
        img2 = self.train_imgs[img_idx2]

        idxs = torch.LongTensor([index, img_idx1, img_idx2])
        attrs = torch.cat([self.skt_labels[index:index+1], self.img_labels[[img_idx1, img_idx2]]])
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

    def loader(self, **args):
        return DataLoader(dataset=self, **args)


if __name__ == '__main__':
    data = SbirData(obj='handbags', hard_ratio=0.75)
