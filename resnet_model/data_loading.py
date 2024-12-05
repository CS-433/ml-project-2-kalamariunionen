import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class ImageLabelDataset(Dataset): 
    
    def __init__(self, images_dir,df, transform=None, split='train'):
        self.images_dir = images_dir
        self.transforms = transform

        train_ratio = 0.8
        val_ratio = 0.1

        # Calculate split indices based on the size of the dataset
        total_size = len(df)
        train_end = int(total_size * train_ratio)
        val_end = train_end + int(total_size * val_ratio)

        SPLITS = {
            'train': list(range(0, train_end)),
            'val':   list(range(train_end, val_end)),
            'test':  list(range(val_end, total_size))
        }

        self.data = []
        for imgIndex in SPLITS[split]:
            row = df.iloc[imgIndex]
            imgName = os.path.join(images_dir, row['original_file'])

            if not os.path.exists(imgName):
                print(f"File not found: {imgName}")
                continue
            if os.path.getsize(imgName) == 0:
                print(f"File is empty: {imgName}")
                continue
            try:
                with Image.open(imgName) as img:  # Open to verify it's a valid image
                    img.verify()
            
            except (IOError, UnidentifiedImageError):
                print(f"Invalid image file: {imgName}")
                continue

            rgb_tensor = torch.tensor((row['r_thorax'], row['g_thorax'], row['b_thorax']), dtype=torch.float32)

            self.data.append((
                imgName,
                rgb_tensor
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        imgName, label = self.data[index]

        if not os.path.exists(imgName):
            print(f"File not found: {imgName}")
        elif os.path.getsize(imgName) == 0:
            print(f"File is empty: {imgName}")
        else:
            img = Image.open(imgName)#.convert('HSV')
            if self.transforms is not None:
                img = self.transforms(img)
        

        return img, label