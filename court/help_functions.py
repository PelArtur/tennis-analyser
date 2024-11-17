from torchvision import models, transforms
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
def draw_keypoints(frame, keypoints):
    for i in range(0, len(keypoints), 2):
        x, y = int(keypoints[i]), int(keypoints[i+1])
        cv.circle(frame, (x, y), 5, (0, 255, 0), -1)  
    return frame

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_keypoints_model(model_path):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 14*2)  
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def predict_keypoints(model, device, image):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    keypoints = outputs.squeeze().cpu().numpy()
    original_h, original_w = image.shape[:2]
    
    keypoints[::2] *= original_w / 224.0
    keypoints[1::2] *= original_h / 224.0
    
    return keypoints