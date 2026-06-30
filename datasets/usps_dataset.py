import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms


class USPSDataset(Dataset):
    def __init__(self,h5_file,imageSize, mode ):
        self.h5_file = h5_file
        self.imageSize = imageSize
        self.mode  = mode
        self.transform = transforms.Compose([transforms.Resize(self.imageSize)]) 
        with h5py.File(self.h5_file, "r") as f:
            self.images = f[f"{self.mode}/data"][:]
            self.labels = f[f"{self.mode}/target"][:]
    def __len__(self):
        return len(self.images)
    def __getitem__ (self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
         # reshape 256 -> 16x16
        image = image.reshape(16, 16)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = self.transform(image)
        return image, label