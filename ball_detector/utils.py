import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from math import inf
from typing import List


def train(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, num_epochs: int, device: str = "cuda") -> List[List[float]]:
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    model.to(device)
    min_val_loss = inf
    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
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

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images).squeeze(1)

                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val_loss}")
            val_loss_arr.append(avg_val_loss)

            torch.save(model.state_dict(), "model.pt")
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), "model_best.pt")

    return train_loss_arr, val_loss_arr