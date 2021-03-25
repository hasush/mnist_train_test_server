import os

import idx2numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mnist_classifier.config import Config

class MnistDataset(Dataset):

    def __init__(self, data_mode):

        # Set configuration.
        self.config=Config()

        # MNIST parameters.
        self.num_classes=10

        # Load images and data for training, validation, and testing.
        if data_mode=="train" or data_mode=="val":
            data = idx2numpy.convert_from_file(os.path.join(self.config.images_dir, 'train-images-idx3-ubyte'))
            labels = idx2numpy.convert_from_file(os.path.join(self.config.labels_dir, 'train-labels-idx1-ubyte'))

            # Create train/validation split.
            all_indices=set([i for i in range(len(data))])
            val_indices=set(np.random.choice(len(data), size=int(self.config.train_val_split[1]*len(data)), replace=False))
            train_indices=all_indices-val_indices
            train_indices,val_indices=list(train_indices),list(val_indices)
            assert len(train_indices)+len(val_indices)==len(all_indices)
            if data_mode=="train":
                self.data=data[train_indices]
                self.labels=labels[train_indices]
            else:
                self.data=data[val_indices]
                self.labels=labels[val_indices]

        elif data_mode=="test":
            self.data = idx2numpy.convert_from_file(os.path.join(self.config.images_dir, 't10k-images-idx3-ubyte'))
            self.labels = idx2numpy.convert_from_file(os.path.join(self.config.labels_dir, 't10k-labels-idx1-ubyte'))
        else:
            raise ValueError("Data mode must be either 'train','val', or 'test'")
        
        # Normalize images.
        # ToTensor()->[0,1]
        # Normalize(0.5,0.5)->[-1,1]
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

        # Set length needed for inherited Dataset.
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Convert index to tensor index.
        if torch.is_tensor(idx):
            idx=idx.tolist()

        # Get image and label. 
        # Note: no need to one hot encode as NLL_Loss expects (batch, 1) tensor for targets
        # and (batch, num_classes) tensor for output.
        image = self.transform(self.data[idx])
        label = torch.tensor(self.labels[idx],dtype=torch.long)
        return image, label