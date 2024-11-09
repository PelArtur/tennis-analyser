import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from math import inf
from typing import List
from math import dist


def extract_ball_center(image: np.ndarray) -> tuple[int]:
    heatmap = image * 255
    heatmap = heatmap.astype(np.uint8)
    _, heatmap = cv.threshold(heatmap, 127, 255, cv.THRESH_BINARY)
    hough_circles = cv.HoughCircles(image=heatmap, method=cv.HOUGH_GRADIENT, dp=0.5, minDist=1, param1=10, param2=0.2, minRadius=2, maxRadius=6)
    if hough_circles is not None and len(hough_circles) == 1:
        return hough_circles[0][0][0], hough_circles[0][0][1]
    return -1, -1


def train(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, num_epochs: int, device: str = "cuda") -> List[List[float]]:
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    model.to(device)
    max_f1_score = 0
    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader)}")
        train_loss_arr.append(train_loss / len(train_loader))

        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            with torch.no_grad():
                for images, masks, x_gt, y_gt in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images).squeeze(1)

                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    outputs = outputs.argmax(dim=1).detach().cpu().numpy()
                    for i in range(len(outputs)):
                        x, y = extract_ball_center(outputs[i])
                        if x_gt[i] != -1 and y_gt[i] != -1:
                            out_dist = dist((x, y), (x_gt[i], y_gt[i]))
                            if x == -1 and y == -1:
                                fn += 1
                            elif out_dist < 5:
                                tp += 1
                            else:
                                fp += 1
                        else:
                            if x == -1 and y == -1:
                                tn += 1
                            else:
                                fp += 1

            print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val_loss}")
            val_loss_arr.append(avg_val_loss)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            print(f"Precision: {(precision * 100):.2f}%")
            print(f"Recall: {(recall * 100):.2f}%")
            print(f"Accuracy: {(accuracy * 100):.2f}%")
            print(f"f1 score: {(f1_score * 100):.2f}%")

            torch.save(model.state_dict(), "model.pt")
            if max_f1_score < f1_score:
                max_f1_score = f1_score
                torch.save(model.state_dict(), "model_best.pt")

    return train_loss_arr, val_loss_arr