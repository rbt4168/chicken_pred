
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os

class ChickenDataset(Dataset):
    def __init__(self, path, tfm):
        super(ChickenDataset).__init__()
        self.transform = tfm
        files_csv = pd.read_csv(path)
        self.files = files_csv["images"].tolist()
        self.labels = files_csv["label"].tolist()
        self.label_map = np.unique(self.labels)
        self.label_map = dict(zip(self.label_map, range(len(self.label_map))))
        self.labels = [self.label_map[l] for l in self.labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(os.path.join("./Train", fname)).convert("RGB")
        im = self.transform(im)

        return im, self.labels[idx]
    
    def get_label_map(self):
        return self.label_map