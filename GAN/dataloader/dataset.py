from PIL import Image
import torch
from torch.utils.data import Dataset
import os


class KITTIDataset(Dataset):
    def __init__(self, data_dir, transform, data_type='training'):#testing or training
        path2data = os.path.join(data_dir, data_type, 'image_2')
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image_resize = image.resize((150, 50))
        image_resize = self.transform(image_resize)
        
        return image_resize
    

#사이즈가 달라서 [3, 375, 1242]로 맞췄음!!! -> [3, 50, 150]
