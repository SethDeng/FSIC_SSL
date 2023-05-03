import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import torchvision.transforms.functional as TF
import numpy as np

class MiniImageNet(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split')

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))
        self.setname = setname

        if setname == 'val':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

            # rotation transform
            self.transform_rot = transforms.Compose([
                transforms.RandomResizedCrop(image_size)
            ])
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])

            # contrastive transform
            self.transform_con = transforms.Compose([
            transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # BYOL
                ], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([GaussianBlur()],
            #     p = 0.2),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])),
                                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.setname == 'val':
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
            return image, label
        
        elif self.setname == 'train':
            path, label = self.data[i], self.label[i]
            original_image = self.transform(Image.open(path).convert('RGB'))
            
            image = self.transform_rot(Image.open(path).convert('RGB'))
            image_0 = self.to_tensor(image)
            image_90 = self.to_tensor(TF.rotate(image, 90))
            image_180 = self.to_tensor(TF.rotate(image, 180))
            image_270 = self.to_tensor(TF.rotate(image, 270))
            rot_images = torch.stack([image_0, image_90, image_180, image_270], 0) # <4, 3, size, size>

            return original_image, rot_images, label


if __name__ == '__main__':
    pass