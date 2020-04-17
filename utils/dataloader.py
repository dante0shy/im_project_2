import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
import cv2,gif2numpy
import random
class BasicDataset(Dataset):
    train_val_dict= {
        'train':[35,36,37,38,39,40],
        'val': [24],
        'test':[2]
    }
    filp_dict = [-1,0,1,2]

    def __init__(self, imgs_dir, scale=1, set_mode ='train'):
        assert set_mode in self.train_val_dict.keys()
        self.set_mode = set_mode
        self.imgs_dir = imgs_dir
        if set_mode != 'test':
            self.data_path_format = '{:0>2d}_training.tif'#os.path.join(imgs_dir, )
            self.mask_path_format = '{:0>2d}_manual1.gif'#os.path.join(imgs_dir, )
        else:
            self.data_path_format = '{:0>2d}_test.tif'#os.path.join(imgs_dir,)
            self.mask_path_format = ''#os.path.join(imgs_dir)

        # data_path_format = glob.glob(os.path.join(imgs_dir, '*'))
        self.data = []
        for d in self.train_val_dict[set_mode]:
            data_path= os.path.join(imgs_dir, self.data_path_format.format(d))
            if set_mode == 'test':
                self.data.append([data_path,''])
            else:
                mask_path = os.path.join(imgs_dir, self.mask_path_format.format(d))
                self.data.append([data_path,mask_path])
        self.scale = scale

    def __len__(self):
        return len(self.data)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    def get_all(self):
        data = []
        mm = []
        for d in self.data:
            img = cv2.imread(d[0])
            if data[1]:
                mask = gif2numpy.convert(d[1])
                mask = mask[0][0]
                mask = mask[:, :, 0] == 255
                mm.append(mask)
            data.append(img)
        data = np.array(data)
        mm = np.array(mm)
        return data,mm

    def process_data(self,img,random_seed = 2):
        if random_seed!=2:
            h_flip = cv2.flip(img, random_seed)
        else:
            h_flip = img
        return h_flip

    def __getitem__(self, i):
        random_seed = random.randint(0,3)
        random_seed = self.filp_dict[random_seed]
        data = self.data[i]
        img = cv2.imread(data[0])
        mask = np.zeros_like(img[:,:,0])
        if data[1]:
            mask = gif2numpy.convert(data[1])
            mask = mask[0][0]
            # mask_out = mask[:, :, 0] == 255
            mask_out = np.zeros_like(mask[:, :, 0])
            mask_out[mask[:, :, 0] == 255] = 1
            mask = mask_out
            # if self.set_mode=='train':
            #     mask = self.process_data(mask,random_seed)
            # mask = mask[224:448,224:448]
        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        # img= img[224:448,224:448,:]
        # img_out = img
        # img = self.process_data(img,random_seed)
        img_out = cv2.resize(img, (448,448))
        # cv2.imshow('a',img_out)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img_out = img_out / 255
        # mean = (img_out).mean((0,1))
        # img_out = img_out - mean
        return {'image': torch.from_numpy(img_out).permute((2, 0, 1)), 'mask': torch.from_numpy(np.array(mask)),'o_s': img.shape[:2]}