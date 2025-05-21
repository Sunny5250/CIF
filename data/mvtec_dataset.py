import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

ALL_CATEGORY_mvtec = [
    "bottle",
    "cable",
    "carpet",
    "capsule",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(
            self,
            is_train,
            mvtec_dir,
            resize_shape,
            k_shot=None,
            indices_file=None,
            sampling='order',
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.k_shot = k_shot
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))

            if k_shot == -1 or k_shot >= len(self.mvtec_paths):
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
                    self.mvtec_paths = [self.rgbmvtec_paths_paths[i] for i in indices]
                    print("Selected samples for training from indices file:")
                    for path in self.mvtec_paths:
                        print(os.path.basename(path))
                
                elif sampling == 'order':
                    self.mvtec_paths = self.mvtec_paths[:k_shot]
                    print(f"Selected first {k_shot} samples for training.")
                
                elif sampling == 'random':
                    indices = np.random.choice(len(self.mvtec_paths), k_shot, replace=False)
                    self.mvtec_paths = [self.mvtec_paths[i] for i in indices]
                    print(f"Randomly selected {k_shot} samples for training.")

            else:
                print('Wrong k_shot settings!')
            
            if flag == 'kshot':
                sample_names = [os.path.basename(path) for path in self.mvtec_paths]
                print(f"Selected {k_shot} samples for training: " + ", ".join(sample_names))

        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
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
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = self.rgb_transform(image)
        _, h, w = image.shape
        fore_mask = torch.ones((h, w))

        if self.is_train:
            return {"rgb_image": image, "foreground": fore_mask}
        else:
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"rgb_image": image, "foreground": fore_mask, "mask": mask}

