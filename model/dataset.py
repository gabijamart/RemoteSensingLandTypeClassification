import os
from PIL import Image
from torchvision import transforms
import torch
import random
import glob

CLASSES = (
    "Unknown",
    "Water (Permanent)",
    "Artificial Bare Ground",
    "Natural Bare Ground",
    "Snow/Ice (Permanent)",
    "Woody",
    "Non-Woody Cultivated",
    "Non-Woody (Semi) Natural"
)

IMAGE_CLASSES = (
    "TrueColor",
    "FalseColor",
    "SWIR",
    "NDVI"
)

class LandCoverNetDataset(torch.utils.data.Dataset):
    base_path: str
    is_training: bool

    image_path_list: list[str]
    mask_dictionary: dict[str, torch.Tensor]

    def __init__(self, base_path, image_class, is_training = False, image_split = None):
        self.base_path = base_path
        self.is_training = is_training

        self.image_path_list = []
        self.mask_dictionary = {}

        for tile in glob.glob("*", root_dir = base_path):
            tile_path = os.path.join(base_path, tile)
            for chip in glob.glob("*", root_dir = tile_path):
                chip_path = os.path.join(tile_path, chip)

                mask = Image.open(os.path.join(chip_path, f"{tile}_{chip}_2018_MASK.tif"))
                self.mask_dictionary[tile + chip] = transforms.functional.to_tensor(mask).long()

                image_path = os.path.join(chip_path, image_class)
                for image in glob.glob("*", root_dir = image_path):
                    self.image_path_list.append(os.path.join(image_path, image))
        
        if image_split != None:
            self.image_path_list = self.image_path_list[image_split[0]:image_split[1]]
    
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx: int):
        image_path = self.image_path_list[idx]
        image = Image.open(image_path)
        image = transforms.functional.to_tensor(image)

        image_path_split = image_path.split(os.sep)
        mask = self.mask_dictionary[image_path_split[1] + image_path_split[2]]
        
        # Apply transforms
        rotation_angles = (0, 90, 180, 270)

        if self.is_training:
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            angle = random.choice(rotation_angles)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = torch.clamp(image, min=-1.0, max=1.0)

        return image, mask.squeeze(dim=0)
