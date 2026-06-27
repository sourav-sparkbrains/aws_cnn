import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class XRAYDataset(Dataset):
    def __init__(self, data_path: str, transform):
        directories = Path(data_path)
        self.transform = transform
        self.classes = []
        self.labels = {}
        self.samples = []

        for dir in directories.iterdir():
            if dir.is_dir():
                self.classes.append(dir.name)

        for i, cls in enumerate(self.classes):
            self.labels[cls] = i

        for cls in self.classes:
            image_paths = Path(data_path) / cls
            for image_path in image_paths.iterdir():
                self.samples.append((image_path, self.labels[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.transform(Image.open(path))
        return image, torch.tensor(label,dtype=torch.long)