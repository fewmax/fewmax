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


class VisDA17(Dataset):

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor(), label_one_hot=False, portion=1.0):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        self.lines = open('fileo.txt', 'r').readlines()
        for i in range(len(self.lines)):
            line = self.lines[i]
            print(line[1])
            exit()
            print(line[0])
        exit()
        exit()
                self.lines = open('file_pth.txt', 'r').readlines()
        for i in range(len(self.lines)):
            line = str.split(self.lines[i])
        exit()
        exit()
                list_ = []
        with open("file_fewshot.txt", 'w') as output:
            for ind in range(len(self.lines)):
                line = str.split(self.lines[ind])
                list_.append(line[1])
                Counter(list_).keys()
                Counter(list_).values()
                if 

                output.write(line[0] + '\t'+ line[1] + '\n')
            self.lines = open('file_fewshot.txt', 'r').readlines()
        list_= []
        for i in range(len(self.lines)):
            line = str.split(self.lines[i])
            list_.append(line[])
        """
        self.lines = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.label_one_hot = label_one_hot
        self.portion = portion
        self.number_classes = 12
        assert portion != 0
        if self.portion > 0:
            self.lines = self.lines[:round(self.portion * len(self.lines))]
        else:
            self.lines = self.lines[round(self.portion * len(self.lines)):]

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
            label = np.asarray(line[1], dtype=np.int)
        label = torch.from_numpy(label)
        return image, label


if __name__ == "__main__":
    a = VisDA17('/csiNAS/visda/test/image_list.txt', '/csiNAS/visda/test')
    print(a.__getitem__(1))
    exit()
