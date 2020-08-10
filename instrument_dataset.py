from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop)

def train_transform(p=1):
    return Compose([
        VerticalFlip(p=0.0),
        HorizontalFlip(p=0.0),
        Normalize(p=1)
    ], p=p)

seed = 12
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

IMG_MEAN = np.array((0.4789, 0.3020, 0.3410), dtype=np.float32)

class SurgicalDataset(Dataset):
    def __init__(self, training_set, batch_size, clip_length, seq_len, transform=None, is_train=None):
        self.is_train = is_train
        self.dir_root_pred = './Instrument_17/instrument_dataset_'
        self.dir_root_gt = './Instrument_17/instrument_dataset_'
        self.transform = transform
        self.clip_len = clip_length
        self.list = training_set
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sal_list = {}

        for i in self.list:
            dir_sal = self.dir_root_gt + str(i) + '/salmap/'
            sal_files = os.listdir(dir_sal)
            sal_files.sort()
            iter_num = len(sal_files) // self.clip_len
            sal_files = sal_files[:iter_num * self.clip_len]
            self.sal_list[dir_sal] = sal_files

    def __len__(self):
        return int(len(self.sal_list) / self.batch_size)

    def __clipincreament__(self, currentclip_seq):
        self.currentclip = currentclip_seq

    def __getitem__(self, index):
        tot_sal_files = []
        for i in range(self.batch_size):
            dir_sals = self.dir_root_gt + str(self.list[self.batch_size * index + i]) + '/salmap/'
            sal_files = self.sal_list[dir_sals]
            sal_files = sal_files[self.currentclip * self.seq_len:self.currentclip * self.seq_len + self.seq_len]
            for j in range(len(sal_files)):
                tot_sal_files.append([dir_sals, sal_files[j]])

        tot_sal_files = np.array(tot_sal_files)
        sort_sal_ind = np.argsort(tot_sal_files[:, -1], kind='mergesort')
        sort_sal = tot_sal_files[sort_sal_ind]
        sort_sal = sort_sal.tolist()

        data = []
        gt_seg = []
        gt_sal = []
        packed = []
        isAugment = random.random() < 0.5
        isHflip = random.random() < 0.5
        for i, sal_path in enumerate(sort_sal):
            dir_sal = sal_path[0] + sal_path[1]
            dir_img = sal_path[0][:-7] + 'images/' + sal_path[1][:-4] + '.jpg'
            dir_mask = sal_path[0][:-7] + 'instruments_masks/' + sal_path[1][:-4] + '.png'
            # print(dir_mask)

            _img = Image.open(dir_img).convert('RGB')
            _target = Image.open(dir_mask)
            _target_sal = Image.open(dir_sal).convert('L')
            if self.is_train:
                if isAugment:
                    if isHflip:
                        _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                        _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
                        _target_sal = _target_sal.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        _img = _img.transpose(Image.FLIP_TOP_BOTTOM)
                        _target = _target.transpose(Image.FLIP_TOP_BOTTOM)
                        _target_sal = _target_sal.transpose(Image.FLIP_TOP_BOTTOM)

            transform_data = {"image": np.array(_img), "mask": np.array(_target)}
            augmented = self.transform(**transform_data)
            _img, _target = augmented["image"], augmented["mask"]
            _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
            _target = torch.from_numpy(np.array(_target)).long()
            _target_sal = np.array(_target_sal)
            _target_sal = _target_sal * 1.0 / 255
            _target_sal = torch.from_numpy(np.array(_target_sal)).float()

            data.append(_img.unsqueeze(0))
            gt_seg.append(_target.unsqueeze(0))
            gt_sal.append(_target_sal.unsqueeze(0))
            if (i + 1) % self.batch_size == 0:
                data_tensor = torch.cat(data, 0)  # bug was actually here
                gt_seg_tensor = torch.cat(gt_seg, 0)
                gt_sal_tensor = torch.cat(gt_sal, 0)
                packed.append((data_tensor, gt_seg_tensor, gt_sal_tensor))
                data = []
                gt_seg = []
                gt_sal = []
        return packed



class SurgicalDataset_seg(Dataset):
    def __init__(self, data_seq,transform=train_transform(p=1), isTrain=None):
        self.is_train = isTrain
        self.dir_root_pred = './Instrument_17/instrument_dataset_'
        self.dir_root_gt = './Instrument_17/instrument_dataset_'
        self.transform = transform
        self.sal_list = {}
        self.sal_files = []
        self.list = data_seq

        for i in self.list:
            dir_sal = self.dir_root_gt + str(i) + '/salmap/'
            self.sal_files = self.sal_files + glob(dir_sal + '/*.png')
            random.shuffle(self.sal_files)

    def __len__(self):
        return len(self.sal_files)

    def __getitem__(self, index):
        _target_sal = Image.open(self.sal_files[index]).convert('L')
        _img = Image.open(
            os.path.dirname(os.path.dirname(self.sal_files[index])) + '/images/' + os.path.basename(
                self.sal_files[index][:-4]) + '.jpg').convert('RGB')
        _target = Image.open(os.path.dirname(os.path.dirname(self.sal_files[index])) + '/instruments_masks/'
                             + os.path.basename(self.sal_files[index][:-4]) + '.png')
        if self.is_train:
            isAugment = random.random() < 0.5
            if isAugment:
                isHflip = random.random() < 0.5
                if isHflip:
                    _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
                    _target_sal = _target_sal.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    _img = _img.transpose(Image.FLIP_TOP_BOTTOM)
                    _target = _target.transpose(Image.FLIP_TOP_BOTTOM)
                    _target_sal = _target_sal.transpose(Image.FLIP_TOP_BOTTOM)

        transform_data = {"image": np.array(_img), "mask": np.array(_target)}
        augmented = self.transform(**transform_data)
        _img, _target = augmented["image"], augmented["mask"]


        #_img = np.asarray(_img, np.float32) * 1.0 / 255
        #_img -= IMG_MEAN
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        _target = torch.from_numpy(np.array(_target)).long()
        _target_sal = np.array(_target_sal)
        _target_sal = _target_sal * 1.0 / 255
        _target_sal = torch.from_numpy(np.array(_target_sal)).float()
        return _img, _target, _target_sal