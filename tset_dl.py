import argparse
import logging
import os
import sys

import numpy as np
import torch,cv2
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import UNet,ResNetUNet
from utils.dataloader import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extras')
if not  os.path.exists(out_img):
    os.mkdir(out_img)
dir_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ckpt')
if not  os.path.exists(dir_checkpoint):
    os.mkdir(dir_checkpoint)

# def test_net(net, loader):
#     net.eval()
#     n_val = len(loader)
#     with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
#         for batch in loader:
#             imgs = batch['image']
#             imgs = imgs.to(dtype=torch.float32)
#             with torch.no_grad():
#                 mask_pred = net(imgs)
#             per = torch.argmax(mask_pred,dim=1)
#
#     return per
#     # return mid[0,0] / (mid[1,1]+mid[0,0])

def test_net(net,ckpt,
              batch_size=1,
              img_scale=0.5):

    test_data = BasicDataset(dir_img, img_scale,set_mode='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    n_val = len(test_loader)
    with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
        idx = 0
        for batch in test_loader:
            imgs = batch['image']
            imgs = imgs.to(dtype=torch.float32)
            with torch.no_grad():
                mask_pred = net(imgs)
            per = torch.nn.functional.upsample(mask_pred, size=(batch['o_s'][0][0], batch['o_s'][1][0]),
                                               mode='bilinear')
            if net.n_classes > 1:
                per = torch.softmax(per, dim=1)
                per = torch.argmax(per, dim=1)
            else:
                per = torch.sigmoid(per) >= 0.5
            # per = torch.argmax(mask_pred, dim=1)
            per = per.numpy()
            per[per ==1 ] =255
            for im in per:
                cv2.imwrite(os.path.join(out_img,'res-{}.jpg'.format(idx)),im)



if __name__ == '__main__':

    net = ResNetUNet(2)
    ckpt_dir = os.path.join(dir_checkpoint, 'CP_epoch61.pth')
    test_net(net,ckpt = ckpt_dir,batch_size = 1 ,img_scale=1)
