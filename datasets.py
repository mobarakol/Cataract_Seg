from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SurgicalDataset(Dataset):
    def __init__(self,data_root=None, data_seq=None, isTrain=None):
        self.isTrain = isTrain
        self.img_dir_list = []
        for seq_idx in data_seq:
            self.img_dir_list = self.img_dir_list + glob(data_root+'{0:02d}'.format(seq_idx)+'/Images/*.png')
            random.shuffle(self.img_dir_list)

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, index):
        _img = Image.open(self.img_dir_list[index]).convert('RGB')
        _tar = Image.open(self.img_dir_list[index].replace('Images','Labels')).convert('L')

        if self.isTrain:
            if random.choice([True, False]):
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                _tar = _tar.transpose(Image.FLIP_LEFT_RIGHT)
        
        _img = transforms.ToTensor()(_img)
        _tar = torch.from_numpy(np.array(_tar)).long()

        return _img, _tar
            

