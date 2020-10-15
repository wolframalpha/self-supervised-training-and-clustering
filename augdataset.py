import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms 
from glob import glob
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset


class AugmentImageDataset(Dataset):
    def __init__(self, file_dirs, tfms, label_vector=None, weight=1.0, return_aug=False, return_fp=True):
        file_paths = [fp for file_dir in file_dirs for fp in glob.glob(file_dir, recursive=True)]
        print('Number of Images: ', len(file_paths))
        file_paths = [x for x in file_paths if (x.lower().endswith("bmp") or x.lower().endswith('png'))]

        self.filepaths = file_paths 
        print(tfms)
        self.label_vector = label_vector
        
        if self.label_vector is None:
            self.classes = np.array(list(set([os.path.split(os.path.dirname(x))[-1] for x in file_paths])))
        self.weight = weight
        self.tfms_1 = tfms
        self.tfms_2 = tfms
        self.return_aug = return_aug
        self.return_fp = return_fp
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with open(self.filepaths[idx], 'rb') as f:
            X_org = Image.open(f).convert('RGB')

        X = self.tfms_1(X_org)
        if self.return_aug:

            X_aug = self.tfms_2(X_org)
            
        if self.label_vector is None:
            y = os.path.split(os.path.dirname(self.filepaths[idx]))[-1]
            y = np.where(self.classes == y, 1.0, 0.0).astype(float).argmax()
            y = torch.tensor(y)
        else: 
            y = self.label_vector[idx]
        if self.return_aug:
            return X, X_aug
        if self.return_fp:
            return X, self.filepaths[idx]
        return X

# SimCLR transformations
train_tfms = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomResizedCrop(size=150, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            # aadded 
            transforms.RandomVerticalFlip(),    
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
test_tfms =  transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])




def get_train_dl(dirs, return_aug=True):
    ds = AugmentImageDataset(file_dirs=dirs, tfms=train_tfms, return_aug=return_aug)
    dl = DataLoader(ds,batch_size=256, shuffle=True)
    return dl

def get_test_dl(dirs, return_aug=False, return_fp=True):
    ds = AugmentImageDataset(file_dirs=dirs, tfms=test_tfms, return_aug=return_aug, return_fp=return_fp)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    return dl
