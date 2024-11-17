import os
import csv
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from typing import List


class TrackNetDataset(Dataset):
    def __init__(self, path: str, width: int, height: int, k: int = 3) -> None:
        self.path = path
        self.images_path = os.path.join(self.path, 'images')
        self.labels_path = os.path.join(self.path, 'ground_truth')
        self.coords_path = os.path.join(self.path, "coords.csv")
        self.data = []
        self.labels = []
        self.coords = dict()
        self.width = width
        self.height = height
        
        files = sorted(os.listdir(self.images_path))
        labels = sorted(os.listdir(self.labels_path))
        self.__split_into_samples(k, files, labels)
        self.__read_coord_gt(self.coords_path)
        super().__init__()
        

    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> tuple[np.ndarray]:
        images = []
        x_scale = 1
        y_scale = 1
        for image_id in self.data[index]:
            image = cv.imread(os.path.join(self.images_path, image_id), cv.COLOR_BGR2RGB)
            x_scale = image.shape[1] / self.width
            y_scale = image.shape[0] / self.height
            image = cv.resize(image, (self.width, self.height))
            images.append(image.astype(np.float32) / 255.0) 
            
        data = np.concatenate(images, axis=2)
        data = np.transpose(data, (2, 0, 1)) #RGB, width, height
        label = cv.imread(os.path.join(self.labels_path, self.labels[index]), cv.IMREAD_GRAYSCALE)
        label = cv.resize(label, (self.width, self.height)).astype(np.long)
        x, y = self.coords[self.labels[index][:-4]]
        return data, label, x / x_scale, y / y_scale
    

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

    
    def __read_coord_gt(self, path: str) -> None:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                data = row[0].split(",")
                if len(data) == 3:
                    self.coords[data[0][:-4]] = (int(data[1]), int(data[2])) 
