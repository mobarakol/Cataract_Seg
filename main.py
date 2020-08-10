#System
import numpy as np
import sys
import math
import os
from PIL import Image
import random
import cv2
#import sinkhorn_pointcloud as spc
#Torch
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

#custom
from instrument_dataset import SurgicalDataset, SurgicalDataset_seg
from model import ST_MTL_SEG

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,0"

args = {
    'num_class': 8,
    'num_gpus':3,
    'lr': 0.0001,
    'batch_size': 6,
    'max_epoch': 150,
    'lr_decay': 0.9,
    'weight_decay': 1e-4,
    'opt': 'adam',
    'log_interval': 50,
    'ckpt_dir': 'ckpt/ours/'}

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])

def dice_2d(pred, label):
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)))
    return dice_val

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def good_worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 2 ** 32 - 1))


def seed_everything(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(train_loader, model, criterion, optimizer, epoch, epoch_iters):
    model.train()
    for batch_idx, (inputs, labels_seg, _) in enumerate(train_loader):
        inputs, labels_seg = Variable(inputs).cuda(), Variable(labels_seg).cuda()
        optimizer.zero_grad()
        pred_seg = model(inputs)
        loss = criterion(pred_seg, labels_seg)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args['log_interval'] == 0:
            print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.6f]' % (
                epoch, batch_idx + 1, epoch_iters, loss.item(),
                optimizer.param_groups[0]['lr']))

def validate(valid_loader, model):
    w, h = 0, args['num_class']
    dice_valid = [[0 for x in range(w)] for y in range(h)]
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels_seg, _) in enumerate(valid_loader):
            inputs, labels_seg = Variable(inputs).cuda(), np.array(labels_seg)
            pred_seg = model(inputs)
            pred_seg = pred_seg.data.max(1)[1].squeeze_(1).cpu().numpy()

            for b_idx in range(labels_seg.shape[0]):
                lab_cls = np.unique(labels_seg[b_idx])
                for cls_idx in range(1, len(lab_cls)):
                    labels_temp = np.zeros(labels_seg.shape[1:])
                    img_pred_temp = np.zeros(labels_seg.shape[1:])
                    labels_temp[labels_seg[b_idx] == lab_cls[cls_idx]] = 1
                    img_pred_temp[pred_seg[b_idx] == lab_cls[cls_idx]] = 1
                    if (np.max(labels_temp) == 0):
                        continue
                    dice_valid[lab_cls[cls_idx]].append(dice_2d(img_pred_temp, labels_temp))            

    return dice_valid

if __name__ == '__main__':
    seed_everything()
    dataset_train = SurgicalDataset_seg(data_seq = [1,2,3,5,6,8], isTrain=True)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args['batch_size'], shuffle=True, num_workers=2, worker_init_fn=good_worker_init_fn)   
    dataset_valid = SurgicalDataset_seg(data_seq = [4,7], isTrain=False)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=args['batch_size'], shuffle=False, num_workers=2, worker_init_fn=good_worker_init_fn)
    model = ST_MTL_SEG(num_classes=args['num_class']).cuda()
    model = torch.nn.parallel.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    print('Length of dataset- train:', dataset_train.__len__(), ' valid:', dataset_valid.__len__())
    epoch_iters = dataset_train.__len__() / args['batch_size']
    best_dice = 0
    best_epoch = 0
    for epoch in range( args['max_epoch']):
        train(train_loader, model, criterion, optimizer, epoch, epoch_iters)
        torch.save(model.state_dict(), os.path.join(args['ckpt_dir'], str(epoch) + '.pth.tar'))
        dice_valid = validate(valid_loader, model)
        avg_dice = []
        each = []
        for idx_eval in range(1, args['num_class']):
            if math.isnan(float(np.mean(dice_valid[idx_eval]))):
                dice_valid[idx_eval] = 0
                continue
            AD = np.mean(dice_valid[idx_eval])
            avg_dice.append(AD)
            each.append('cls_%d:%.4f' % (idx_eval, AD))

        if np.mean(avg_dice) > best_dice:
            best_dice = np.mean(avg_dice)
            best_epoch = epoch

        print('Epoch:%d ' % epoch, 'Avg Dice:%.4f %s ' % (np.mean(avg_dice), str(each)),
              'Best Avg=%d : %.4f ' % (best_epoch, best_dice))

