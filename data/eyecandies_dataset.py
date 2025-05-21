import os
import cv2
import glob
import time
import torch
import warnings
import tifffile
import numpy as np
from PIL import Image
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

ALL_CATEGORY_eyecandies = [
    'CandyCane',
    'ChocolateCookie',
    'ChocolatePraline',
    'Confetto',
    'GummyBear',
    'HazelnutTruffle',
    'LicoriceSandwich',
    'Lollipop',
    'Marshmallow',
    'PeppermintCandy',  
]
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def resize_organized_pc(organized_pc, target_height=224, target_width=224):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(
        torch_organized_pc,
        size=(target_height, target_width),
        mode='nearest'
    )

    return torch_resized_organized_pc.squeeze(dim=0).contiguous()


class EyeCandDataset(Dataset):
    def __init__(
            self,
            is_train,
            eyecand_dir,
            resize_shape,
            pc_type,
            n_fills,
            bg_thresh,
            k_shot=None,
            indices_file=None,
            sampling='order',
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.eyecand_dir = eyecand_dir
        self.pc_type = pc_type
        self.n_fills = n_fills
        self.bg_thresh = bg_thresh
        self.k_shot = k_shot
        if is_train:
            self.train_rgb_dir = os.path.join(eyecand_dir, "rgb")
            self.train_pc_dir = os.path.join(eyecand_dir, "xyz")
            self.rgb_paths = sorted(glob.glob(self.train_rgb_dir + "/*.png"))
            self.pc_paths = sorted(glob.glob(self.train_pc_dir + "/*.tiff"))

            if k_shot == -1 or k_shot >= len(self.rgb_paths):
                flag = 'full'
                print("Selected samples for training: Full-shot")
                pass

            elif k_shot >= 0:
                flag = 'kshot'
                if indices_file is not None and sampling == 'file':
                    with open(indices_file, 'r') as f:
                        indices = [int(line.strip()) for line in f.readlines()]
                    if k_shot > 0:
                        indices = indices[:k_shot]
                    self.rgb_paths = [self.rgb_paths[i] for i in indices]
                    self.pc_paths = [self.pc_paths[i] for i in indices]
                    print("Selected samples for training from indices file:")
                    for path in self.rgb_paths:
                        print(os.path.basename(path))
                
                elif sampling == 'order':
                    self.rgb_paths = self.rgb_paths[:k_shot]
                    self.pc_paths = self.pc_paths[:k_shot]
                    print(f"Selected first {k_shot} samples for training.")
                
                elif sampling == 'random':
                    indices = np.random.choice(len(self.rgb_paths), k_shot, replace=False)
                    self.rgb_paths = [self.rgb_paths[i] for i in indices]
                    self.pc_paths = [self.pc_paths[i] for i in indices]
                    print(f"Randomly selected {k_shot} samples for training.")

            else:
                print('Wrong k_shot settings!')
            
            if flag == 'kshot':
                sample_names = [os.path.basename(path) for path in self.rgb_paths]
                print(f"Selected {k_shot} samples for training: " + ", ".join(sample_names))
                    
        else:
            self.rgb_paths = sorted(glob.glob(eyecand_dir + "/*/rgb/*.png"))
            self.pc_paths = sorted(glob.glob(eyecand_dir + "/*/xyz/*.tiff"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.resize_shape, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        rgb_image = Image.open(self.rgb_paths[index]).convert("RGB")
        rgb = rgb_image.copy()
        rgb = self.rgb_transform(rgb)
        depth_image = tifffile.imread(self.pc_paths[index])
        depth_copy = depth_image.copy()
        depth_copy = depth_copy[:, :, 2]

        if self.pc_type == 'pc':
            resized_organized_pc = resize_organized_pc(depth_image, target_height=self.resize_shape[0], target_width=self.resize_shape[1])
            pc = resized_organized_pc.clone().detach().float()

        elif self.pc_type == 'depth':
            depth = cv2.resize(depth_copy, self.resize_shape, interpolation=cv2.INTER_LINEAR)
            depth_3channel = np.repeat(np.expand_dims(depth, axis=0), 3, axis=0)
            pc = torch.tensor(depth_3channel)

        # # Extract foreground
        fore_mask = depth_copy

        if self.is_train:
            return {"rgb_image": rgb, "point_cloud": pc, "foreground": fore_mask}
        else:
            dir_path, file_name = os.path.split(self.rgb_paths[index])
            base_dir = os.path.basename(os.path.dirname(dir_path))
            if base_dir == "good":
                mask = torch.zeros_like(rgb[:1])
            else:
                mask_path = os.path.join(self.eyecand_dir, base_dir)
                mask_path = os.path.join(mask_path, "gt")
                mask_file_name = file_name
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path).convert("L")
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask > 0, torch.ones_like(mask), torch.zeros_like(mask)
                )
            return {"rgb_image": rgb, "point_cloud": pc, "foreground": fore_mask, "mask": mask}
