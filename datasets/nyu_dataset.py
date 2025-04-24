import os
import numpy as np
from PIL import Image
import zipfile

from torch.utils.data import Dataset

from loguru import logger


class NYUDataset(Dataset):
    """
    Loads images and applies model-specific transformations.
    """

    def __init__(self, dataset_path: str, val: bool, transform, split=None, *args, **kwargs):
        super(NYUDataset, self).__init__()
        self.val = val
        self.transform = transform
        self.split = split
        self.filenames = self._load_images(dataset_path)
        # logger.debug(f"filenames: {self.filenames}")

    def _load_images(self, room_path):
        """Recursively search for matching RGB and depth images inside rooms or a zip archive."""
        images = []
        split = getattr(self, 'split', None)
        print(split)
        if room_path.endswith('.zip'):
            with zipfile.ZipFile(room_path, 'r') as zf:
                all_files = zf.namelist()
                # Find all folders under 'data/' (e.g., data/train, data/val, data/test)
                splits = set([f.split('/')[1] for f in all_files if f.startswith('data/') and len(f.split('/')) > 2])
                splits_to_use = [split] if split else splits
                for split_name in splits_to_use:
                    rgb_images = sorted([f for f in all_files if f.startswith(f'data/{split_name}/') and f.endswith('.jpg')])
                    gt_depths = [f.replace('.jpg', '.png') for f in rgb_images]
                    for idx, (rgb_image, gt_depth) in enumerate(zip(rgb_images, gt_depths)):
                        if idx == 0:
                            continue
                        if gt_depth in all_files:
                            images.append((rgb_image, gt_depth))
            self.zip_file = zipfile.ZipFile(room_path, 'r')
            self.is_zip = True
        elif os.path.isdir(room_path):
            rgb_images = sorted(
                [f for f in os.listdir(room_path) if f.endswith(".jpg")])
            gt_depths = [
                filename.replace(".jpg", ".png") for filename in rgb_images
            ]
            for idx, (rgb_image, gt_depth)in enumerate(zip(rgb_images, gt_depths)):
                if idx == 0:
                    continue
                rgb_path = os.path.join(room_path, rgb_image)
                gt_path = os.path.join(room_path, gt_depth)
                paths = (rgb_path, gt_path)
                images.append(paths)
            self.is_zip = False
            self.zip_file = None
        else:
            raise ValueError(f"NYUDataset: Provided path {room_path} is not a directory or zip file.")
        return images

    def __len__(self):
        return len(self.filenames)

    def get_image(self, image_path):
        if hasattr(self, 'is_zip') and self.is_zip and self.zip_file is not None:
            from io import BytesIO
            with self.zip_file.open(image_path, 'r') as f:
                return Image.open(BytesIO(f.read())).convert('RGB' if image_path.endswith('.jpg') else 'I')
        return Image.open(image_path)

    def __getitem__(self, index):
        rgb_path, depth_path = self.filenames[index]
        image = self.get_image(rgb_path)
        depth = self.get_image(depth_path)

        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)

        return sample
