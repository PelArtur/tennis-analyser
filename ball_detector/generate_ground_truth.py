import os
import csv
import cv2 as cv
import numpy as np
from typing import List


def extract_ground_truth(clip_id: int, images: List[int], input_path: str, labels: dict, out_im: str, out_lbl: str) -> None:
    clip_id_str = '0' * (3 - len(str(clip_id))) + str(clip_id)
    image_output_path = os.path.join(out_im,  clip_id_str) + '_'
    label_output_path = os.path.join(out_lbl, clip_id_str) + '_'
    
    for image in images:
        image_data = cv.imread(os.path.join(input_path, image))
        shape = (image_data.shape[0], image_data.shape[1], 1)
        label = np.zeros(shape)
        label = cv.circle(label, labels[image], radius=5, color=255, thickness = -1)
        label = cv.GaussianBlur(label, (7, 7), 4)
        cv.imwrite(image_output_path + image, image_data)
        cv.imwrite(label_output_path + image[:-4] + ".png", label)


def read_csv(path: str) -> dict:
    labels_data = dict()

    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            data = row[0].split(",")
            if len(data) == 5:
                labels_data[data[0]] = (int(data[2]), int(data[3])) if int(data[1]) != 0 else (-1, -1)
                
    return labels_data


def create_dataset(input_path: str, output_path: str) -> None:
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    if not os.path.exists(os.path.join(output_path, "ground_truth")):
        os.makedirs(os.path.join(output_path, "ground_truth"))
    
    clip_id = 0
    for game in os.listdir(input_path):
        if game[:4] == "game":
            for clip in os.listdir(os.path.join(input_path, game)):
                if clip[:4] == "Clip":
                    dir_path = os.path.join(input_path, game, clip)
                    files = os.listdir(os.path.join(input_path, game, clip))
                    images = files[:-1]
                    label = files[-1]
                    
                    label_data = read_csv(os.path.join(dir_path, label))
                    extract_ground_truth(clip_id, images, dir_path, label_data, os.path.join(output_path, "images"), os.path.join(output_path, "ground_truth"))
                    clip_id += 1
