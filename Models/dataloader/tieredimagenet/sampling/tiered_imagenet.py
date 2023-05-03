import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torch
import os
import torchvision.transforms.functional as TF

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size = 3):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class tieredImageNet(Dataset):

    def __init__(self, setname, args):
        TRAIN_PATH = osp.join(args.data_dir, 'tiered_imagenet/train')
        VAL_PATH = osp.join(args.data_dir, 'tiered_imagenet/val')
        TEST_PATH = osp.join(args.data_dir, 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Unkown setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if 'num_patch' not in vars(args).keys():
            self.num_patch = 9
            print('no num_patch parameter, set as default:',self.num_patch)
        else:
            self.num_patch = args.num_patch
        
        image_size = 84
        self.num_patch=args.num_patch

        # original transform
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
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
        path, label = self.data[i], self.label[i]
        patch_list=[]

        con_sample_1 = self.transform_con(Image.open(path).convert('RGB'))
        con_sample_2 = self.transform_con(Image.open(path).convert('RGB'))

        patch_list = []
        for j in range(self.num_patch):
            patch_list.append(self.transform(Image.open(path).convert('RGB')))
        patch_list=torch.stack(patch_list,dim=0)

        return patch_list, con_sample_1, con_sample_2, label


if __name__ == '__main__':
    pass
