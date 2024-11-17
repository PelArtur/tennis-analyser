import os
import torch
import torch.version
from dataset import TrackNetDataset
from model import TrackNet
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from utils import train, evaluate_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    params = {
        "device": "cuda",
        "path": "./TrackNetDataset",
        "width": 640,
        "height": 360,
        "k": 3,
        "epochs": 30,
        "batch_size": 2,
        "test_split": 0.15,
        "val_split": 0.15,
        "random_seed": 42,
        "dataset_partition": 300
    }
    
    dataset = TrackNetDataset(
        path=params["path"],
        width=params["width"],
        height=params["height"],
        k=params["k"],
    )
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    test_size = int(params["test_split"] * dataset_size)
    val_size = int(params["val_split"] * dataset_size)
    
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=params["random_seed"])
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=params["random_seed"])
    
    train_sampler = SubsetRandomSampler(train_indices[:len(train_indices) // params["dataset_partition"]])
    val_sampler = SubsetRandomSampler(val_indices[:len(val_indices) // params["dataset_partition"]])
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=test_sampler)

    torch.cuda.empty_cache() 
    model = TrackNet(in_channels=params["k"] * 3, out_channels=256, training=True)

    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None

    train_loss, val_loss = train(model, train_loader, val_loader, params["epochs"], params["device"], checkpoint)
    test_loss, precision, recall, accuracy, f1_score = evaluate_model(model, test_loader, params["device"])

    print(f"Precision: {(precision * 100):.2f}%")
    print(f"Recall: {(recall * 100):.2f}%")
    print(f"Accuracy: {(accuracy * 100):.2f}%")
    print(f"f1 score: {(f1_score * 100):.2f}%")
    
    with open("train_loss.txt", "w", encoding="utf-8") as file:
        file.write(str(train_loss))
    with open("val_loss.txt", "w", encoding="utf-8") as file:
        file.write(str(val_loss))
