import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class ImageLabelDataset(Dataset): 
    
    def __init__(self, images_dir,df,train_ratio,val_ratio,transform=None, split='train'):
        self.images_dir = images_dir
        self.transforms = transform

        # Calculate split indices based on the size of the dataset
        total_size = len(df)
        train_end = int(total_size * train_ratio)
        val_end = train_end + int(total_size * val_ratio)

        SPLITS = {
            'train': list(range(0, 100)),
           'val':   list(range(100, 120)),
            'test':  list(range(120, 140))
        }

        #SPLITS = {
        #    'train': list(range(0, train_end)),
        #   'val':   list(range(train_end, val_end)),
        #    'test':  list(range(val_end, total_size))
        #}

        self.data = []
        for imgIndex in SPLITS[split]:
            row = df.iloc[imgIndex]
            imgName = os.path.join(images_dir, row['original_file'])

            #Checking if image exsists and is valid
            if not os.path.exists(imgName):
                print(f"File not found: {imgName}")
                continue
            if os.path.getsize(imgName) == 0:
                print(f"File is empty: {imgName}")
                continue
            try:
                with Image.open(imgName) as img:
                    img.verify()
            
            except (IOError, UnidentifiedImageError):
                print(f"Invalid image file: {imgName}")
                continue
            
            #Extracting target color
            rgb_tensor = torch.tensor((row['r'], row['g'],row['b']), dtype=torch.float32)

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
            img = Image.open(imgName)
            if self.transforms is not None:
                img = self.transforms(img)
        
        return img, label
