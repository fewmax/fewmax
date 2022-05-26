import os
from re import T
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pdb import set_trace as bp
import glob
import itertools
import random
from collections import Counter

class VisDA17(Dataset):

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor(), label_one_hot=False, portion=1.0):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
  
        """

        self.lines = open('data/file_fewshot.txt', 'r').readlines() 
        self.root_dir = root_dir
        self.transform = transform
        self.label_one_hot = label_one_hot
        self.portion = portion
        self.number_classes = 12

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = str.split(self.lines[idx])
        path_img = os.path.join(self.root_dir, line[0])
        image = Image.open(path_img)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.label_one_hot:
            label = np.zeros(12, np.float32)
            label[np.asarray(line[1], dtype=np.int)] = 1
        else:
            label = line[1]
            label = np.asarray(line[1], dtype=np.int)
        label = torch.from_numpy(label)
        return image, label

        

if __name__ == "__main__":
    a = VisDA17('/csiNAS/visda/train/image_list.txt', '/csiNAS/visda/train', portion=0.05)
    list_ = []
    for i in range(a.__len__()):
        list_.append(a.__getitem__(i))
    print(Counter(list_).keys())
    print(Counter(list_).values())
    exit()
