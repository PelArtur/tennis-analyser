import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from typing import List


class TrackNetDataset(Dataset):
    def __init__(self, path: str, width: int, height: int, k: int = 3) -> None:
        self.path = path
        self.images_path = os.path.join(self.path, 'images')
        self.labels_path = os.path.join(self.path, 'ground_truth')
        self.data = []
        self.labels = []
        self.width = width
        self.height = height
        
        files = sorted(os.listdir(self.images_path))
        labels = sorted(os.listdir(self.labels_path))
        self.__split_into_samples(k, files, labels)
        super().__init__()
        

    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> tuple[np.ndarray]:
        images = []
        for image_id in self.data[index]:
            image = cv.imread(os.path.join(self.images_path, image_id), cv.COLOR_BGR2RGB)
            image = cv.resize(image, (self.width, self.height))
            images.append(image.astype(np.float32) / 255.0) 
            
        data = np.concatenate(images, axis=2)
        data = np.transpose(data, (2, 0, 1)) #RGB, width, height
        label = cv.imread(os.path.join(self.labels_path, self.labels[index]), cv.IMREAD_GRAYSCALE)
        label = cv.resize(label, (self.width, self.height)).astype(np.long)
        return data, label
    
    
    def __split_into_samples(self, k: int, files: List[str], labels: List[str]) -> None:
        curr_id = files[0][:3]
        i = k - 1
        while i < len(files):
            if i + k - 1 < len(files) and files[i][:3] != curr_id:
                i += k - 1
                curr_id = files[i][:3]
                continue
                
            prevs = []
            for j in range(k):
                prevs.append(files[i - k + j + 1])
            self.data.append(tuple(prevs))
            self.labels.append(labels[i])
            i += 1
