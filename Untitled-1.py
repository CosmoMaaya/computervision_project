# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from platform import python_version
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torch.utils.data as data_utils
import matplotlib.animation as animation
from IPython.display import HTML

# %%
import PIL
from tqdm import tqdm
from torch.utils.data import  DataLoader, Dataset

# %%

origin_file_path = "origin"
test_folder, train_folder = "test_data", "train_data"

# %%
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mask_size=(400, 300), train=True):
        self.transform = T.Compose(transforms_)
        self.mask_size = mask_size
        self.train = train
        self.files = sorted([os.path.join(root, x) for x in os.listdir(root) if os.path.isfile(os.path.join(root, x))])
        
    def apply_random_mask(self, img):
        """Randomly masks image"""
        channel, y, x = img.size()
        y1, x1 = np.random.randint(0, y - self.mask_size[1]), np.random.randint(0, x - self.mask_size[0])
        y2, x2 = y1 + self.mask_size[1], x1 + self.mask_size[0]
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        channel, y, x = img.size()
        y1, x1 = (y - self.mask_size[1]) // 2, (x - self.mask_size[0]) // 2
        y2, x2 = y1 + mask_size[1], x1 + mask_size[0]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, i

    def __getitem__(self, index):
        img = PIL.Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.train:
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

# %%
batch_size = 16
transforms_ = [
    # T.Resize((img_size, img_size), Image.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(train_folder, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_dataloader = DataLoader(
    ImageDataset(test_folder, transforms_=transforms_, train=False),
    batch_size=12,
    shuffle=False,
    num_workers=0,
)



# %%
gen_adv_losses, gen_pixel_losses, disc_losses, counter = [], [], [], []
# number of epochs of training
n_epochs = 4
# size of the batches
it = iter(dataloader)
print(next(it))

