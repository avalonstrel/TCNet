import torch
import os
from .utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
try:
    import cv2
except:
    pass



class HairData(Dataset):

    def __init__(self, root_path, sketch='simple', crop_size=225, hard_ratio=1, edge=False):

        self.hard_ratio = hard_ratio
        self.edge = edge


        if not edge:
            self.photo_root = os.path.join(root_path,'photos')
        else:
            self.photo_root = os.path.join(root_path, 'edges')
        self.sketch_root = os.path.join(root_path, 'sketches')
        self.crop_size = crop_size

        # load train/test set
        info_root = root_path
        with open(os.path.join(info_root, 'train_files.txt'), 'r') as f:
            self.trainset = set(filter(lambda x:x, f.read().split('\n')))
        with open(os.path.join(info_root, 'test_files.txt'), 'r') as f:
            self.testset = set(filter(lambda x:x, f.read().split('\n')))

        # load train set
        all_cates = list(filter(lambda x:x[0] != '.', os.listdir(self.photo_root)))
        all_cates.sort()
        self.cate2label = dict(zip(all_cates, range(len(all_cates))))

        self.trainset_cate = all_cates[:]
        self.testdata = []
        for key, value in self.cate2label.items():
            if key[0] == '.':
                continue

            self.trainset_cate[value] = []
            for f in os.listdir(os.path.join(self.photo_root, key)):
                if f[0] == '.':
                    continue

                fn = f.split('.')[0]

                skts = [fn+'s.png', fn+'c.png']
                if sketch == 'simple':
                    skts = skts[0:1]
                elif sketch == 'complex':
                    skts = skts[1:]

                if key+'/'+f in self.testset:
                    self.testdata.append((key, f, skts))
                else:
                    self.trainset_cate[value].append((key, f, skts))

        self.cate_instances = [len(value) for value in self.trainset_cate]
        self.idx2cate = []
        for i in range(len(self.cate_instances)):
            self.idx2cate += [i] * self.cate_instances[i]


        # merge
        self.train_photos = []
        self.train_sketches = []
        for item in self.trainset_cate:
            for cate, pho, skts in item:
                self.train_photos.append((self.cate2label[cate], os.path.join(self.photo_root, cate, pho)))
                for skt in skts:
                    self.train_sketches.append((self.cate2label[cate],os.path.join(self.sketch_root, cate, skt)))


        # transformations
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(crop_size)
        self.resize256 = transforms.Resize(256)
        # for training photo and sketch
        self.resizedcrop_pretrain = transforms.RandomResizedCrop(size=crop_size, scale=(0.5, 1))
        self.resizedcrop_train = transforms.RandomResizedCrop(size=crop_size, scale=(0.8, 1))

        self.crop = crop_gen(self.crop_size)

        self.mode = 'triplet'
        self.prepare_test()


    def prepare_test(self):
        skts, imgs, idxs = [], [], []

        for i, (cate, img, skt) in enumerate(self.testdata):
            img = Image.open(os.path.join(self.photo_root, cate, img)).resize((256, 256))
            imgs.append(self.to_tensor(img)[:3,:,:])
            for j, s in enumerate(skt):

                s = np.array(Image.open(os.path.join(self.sketch_root, cate, s)))
                s = np.array(s>0.1, np.uint8)[:,:,3]
                x,y,w,h = cv2.boundingRect(s)
                s = s[y:y+h,x:x+w]   
                s = cv2.resize(s, (256,256))
                s = torch.FloatTensor(1-s).repeat(3,1,1)

                skts.append(s)
                idxs.append(i)

        self.test_skts = torch.stack(skts)
        self.test_imgs = torch.stack(imgs)
        self.test_idxs = torch.LongTensor(idxs)

        if self.edge:
            self.test_skts = self.test_skts[:,0:1,:,:] * 255 - 250.42
            self.test_imgs = self.test_imgs * 255 - 250.42


    def __len__(self):
        return len(self.train_photos)

    def set_mode(self, mode):
        assert mode in ['triplet', 'pair', 'pretrain']
        self.mode = mode

    def __getitem__(self, index):
        if self.mode[:7] == 'triplet':
            skt, imgs, idx, attr = self.get_triplet(index)
            imgs = [randomflip(self.crop(img)) for img in imgs]
            skt = randomflip(self.crop(skt))

            if self.edge:
                skt = skt[0:1] * 255 - 250.42
                imgs = [img * 255 - 250.42 for img in imgs]

            return (skt, *imgs, idx, attr)

        elif self.mode == 'pretrain':
            img, label = self.get_random(0)
            return img, label


    def get_triplet(self, index):

        cate_id = self.idx2cate[index]
        pos_idx = index
        for c in range(cate_id):
            pos_idx -= self.cate_instances[c]

        img_skt = self.trainset_cate[cate_id][pos_idx]
        cate, img1, skt = img_skt[0], img_skt[1], random.choice(img_skt[2])

        if random.random() < self.hard_ratio:
            neg_idx = random.randint(0, len(self.trainset_cate[cate_id])-1)
            while neg_idx == pos_idx:
                neg_idx = random.randint(0, len(self.trainset_cate[cate_id]) - 1)
            img2 = self.trainset_cate[cate_id][neg_idx][1]
            idxs = torch.LongTensor([index, index, index + neg_idx - pos_idx])
            attrs = torch.LongTensor([cate_id, cate_id, cate_id])

        else:
            neg_idx = random.randint(0, len(self)-1)
            while neg_idx == index:
                neg_idx = random.randint(0, len(self) - 1)
            idxs = torch.LongTensor([index, index, neg_idx])

            cate_id_neg = self.idx2cate[neg_idx]
            for c in range(cate_id_neg):
                neg_idx -= self.cate_instances[c]
            img_skt = self.trainset_cate[cate_id_neg][pos_idx]
            img2 = img_skt[1]
            attrs = torch.LongTensor([cate_id, cate_id, cate_id])


        # read image and process
        skt = np.array(Image.open(os.path.join(self.sketch_root, cate, skt)))
        skt = np.array(skt>0.1, np.uint8)[:,:,3]
        assert len(skt.shape) == 2, '%s, %s'%(cate, img1)
        x,y,w,h = cv2.boundingRect(skt)
        skt = skt[y:y+h,x:x+w]   
        skt = cv2.resize(skt, (226,226))
        skt_tmp = np.zeros(shape=(256,256), dtype=skt.dtype)
        skt_tmp[15:241, 15:241] = skt
        skt = torch.FloatTensor(1-skt_tmp).repeat(3,1,1)

        img1 = self.to_tensor(Image.open(os.path.join(self.photo_root, cate, img1)).resize((256,256)))[:3,:,:]
        img2 = self.to_tensor(Image.open(os.path.join(self.photo_root, cate, img2)).resize((256,256)))[:3,:,:]

        return skt, [img1, img2], idxs, attrs

    def get_random(self, index):

        if random.random() > 0.5:
            label, fname = random.choice(self.train_photos)
        else:
            label, fname = random.choice(self.train_sketches)

        img = Image.open(fname)
        img = randomflip(self.to_tensor(self.resizedcrop_pretrain(img)))

        label = torch.LongTensor([label])

        return img, label



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

