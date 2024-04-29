from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os

class RoadSeg(Dataset):
    def __init__(self, root_images, root_masks, train=True):
        self.root_img_path = root_images
        if train:
            self.images = sorted([f"{root_images}/train/{img}" for img in os.listdir(f"{root_images}/train")])
            self.masks = sorted([f"{root_masks}/train/{mask}" for mask in os.listdir(f"{root_masks}/train")])
        else:
            self.images = sorted([f"{root_images}/test/{img}" for img in os.listdir(f"{root_images}/test")])
            self.masks = sorted([f"{root_masks}/test/{mask}" for mask in os.listdir(f"{root_masks}/test")])

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return len(self.images)