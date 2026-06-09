import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str or Path): Path to the dataset directory (e.g., 'raw_data/FruitinAmazon/train')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 1. Find all class folders (ignoring hidden files)
        self.classes = sorted([f.name for f in self.root_dir.iterdir() if f.is_dir()])
        
        # 2. Map class names to numeric IDs (e.g., {'apple': 0, 'banana': 1})
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        # 3. Gather all valid image file paths and their matching labels
        self.samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetches one image and its corresponding label at the given index."""
        img_path, label = self.samples[idx]
        
        # Open image and convert to RGB (handles grayscale or CMYK anomalies)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformation pipeline if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label


class CustomImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Find all class folders
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        # Collect image paths and labels
        for cls in self.classes:
            class_path = self.root_dir / cls

            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append(
                        (img_path, self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label