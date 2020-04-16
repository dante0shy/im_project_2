import argparse
import logging
import os
import sys

import numpy as np,cv2
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import ResNetUNet
from utils.dataloader import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extras')
if not  os.path.exists(out_img):
    os.mkdir(out_img)
dir_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ckpt')
if not  os.path.exists(dir_checkpoint):
    os.mkdir(dir_checkpoint)

def eval_net(net, loader):
    net.eval()
    mask_type = torch.long
    n_val = len(loader)
    acc = 0
    mid = torch.FloatTensor([[0,0],[0,0]])
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(dtype=torch.float32)
            true_masks = true_masks.to(dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            per = torch.nn.functional.upsample(mask_pred, size=(batch['o_s'][0][0], batch['o_s'][1][0]), mode='bilinear')
            if net.n_classes >1:
                per = torch.softmax(per,dim=1)
                per = torch.argmax(per,dim=1)
            else:
                per = torch.sigmoid(per)>=0.5
            t = (true_masks == per).sum()
            f = (true_masks != per).sum()
            mid[0,0] += t
            mid[1,1] += f
            pbar.update()

            per = per.numpy()
            per[per == 1] = 255
            for im in per:
                cv2.imwrite(os.path.join(out_img, 'res-val-{}.jpg'.format(1)), im)

    return mid[0,0] / (mid[1,1]+mid[0,0])
    # return mid[0,0] / (mid[1,1]+mid[0,0])

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              img_scale=0.5):

    train_data = BasicDataset(dir_img, img_scale)
    eval_data = BasicDataset(dir_img, img_scale,set_mode='val')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    global_step = 0

    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay=1e-8)
    if net.n_classes > 1:
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = nn.CrossEntropyLoss(weight = torch.Tensor(np.array([1.0,10.0])))#
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        # with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to( dtype=mask_type)
            masks_pred = net(imgs)
            masks_pred = torch.nn.functional.upsample(masks_pred, size=(batch['o_s'][0][0],batch['o_s'][1][0]), mode='bilinear')
            if net.n_classes ==1:
                masks_pred = masks_pred.view((masks_pred.size(0),masks_pred.size(2),masks_pred.size(3),))
            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()
                # pbar.set_postfix(**{'loss (batch)': loss.item()})
                #
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            # pbar.update(imgs.shape[0])
            global_step += 1
        logging.warning('\n{} train loss: {}'.format(epoch, epoch_loss/len(train_loader)))
        if epoch%3 ==0:
            val_score = eval_net(net, val_loader)

            if net.n_classes > 1:
                logging.warning('\nValidation acc: {}'.format(val_score))
            else:
                logging.warning('\nValidation acc: {}'.format(val_score))
            torch.save(net.state_dict(),
                           os.path.join(dir_checkpoint , f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')


if __name__ == '__main__':

    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net = ResNetUNet(2)

    train_net(net=net,
                  epochs=400,
                  batch_size=2,
                  lr=0.001,
                  # device=device,
                  img_scale=1,
                  val_percent=10 / 100)
