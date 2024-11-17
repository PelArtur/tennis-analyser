import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from ball_detector.model import TrackNet
from ball_detector.utils import extract_ball_center
from court.help_functions import predict_keypoints, load_keypoints_model, draw_keypoints
from torchvision import models, transforms

def extract_frames(video_path: str) -> tuple[List[np.ndarray], int]:
    video = cv.VideoCapture(video_path)
    fps = int(video.get(cv.CAP_PROP_FPS))
 
    frames = []
    success, frame = video.read()
     
    while success:
        frames.append(frame)
        success, frame = video.read()

    video.release()
    return frames, fps


def process_frames(model_ball: nn.Module, model_keypoints: nn.Module, device: str, frames: List[np.ndarray], height: int = 360, width: int = 640) -> tuple[List[tuple[float]], List[tuple[float]]]:
    ball_coords: List[tuple[float]] = [(-1.0, -1.0), (-1.0, -1.0)]
    keypoints_list: List[List[float]] = []
    
    torch.cuda.empty_cache() 
    
    model_ball.eval()
    model_keypoints.eval()
    
    with torch.no_grad():
        for i in tqdm(range(2, len(frames))):
            input = np.concatenate([
                cv.cvtColor(cv.resize(frames[i - 2], (width, height)), cv.COLOR_BGR2RGB),
                cv.cvtColor(cv.resize(frames[i - 1], (width, height)), cv.COLOR_BGR2RGB),
                cv.cvtColor(cv.resize(frames[i], (width, height)), cv.COLOR_BGR2RGB)
            ], axis=2)
            input = np.transpose(input, (2, 0, 1))
            input = input.astype(np.float32) / 255.0
            
            output = model_ball(torch.tensor(np.array([input])).to(device))
            x, y = extract_ball_center(output.argmax(dim=1).to('cpu').numpy()[0])
            ball_coords.append((x, y))
            
            keypoints = predict_keypoints(model_keypoints, device, frames[i])  
            keypoints_list.append(keypoints)

    return ball_coords, keypoints_list


def save_video(video_path: str, fps: int, frames: List[np.ndarray], ball_coords: List[tuple[float]], keypoints_list: List[List[float]], height: int = 360, width: int = 640) -> None:
    out_height, out_width = frames[0].shape[0], frames[0].shape[1] 
    y_scaling, x_scaling = out_height / height, out_width / width
    
    output = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (out_width, out_height))
    
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        
        for j in range(10):
            if i - j >= 0 and ball_coords[i - j][0] != -1 and ball_coords[i - j][1] != -1:
                overlay = frame.copy()
                overlay = cv.circle(frame, (int(ball_coords[i - j][0] * x_scaling), int(ball_coords[i - j][1] * y_scaling)), int(5 - 0.5 * j), (0, 0, 255), -1)
                alpha = 1 - 0.1 * j
                frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        if i < len(keypoints_list):
            keypoints = keypoints_list[i]
            frame_with_keypoints = draw_keypoints(frame, keypoints)
        
        output.write(frame_with_keypoints)
    output.release()


if __name__ == "__main__":
    video_path = './input_video.mp4'
    all_frames, fps = extract_frames(video_path)

    ball_model = TrackNet()
    ball_model_data = torch.load("ball_detector.pt", weights_only=True, map_location="cpu")
    ball_model.load_state_dict(ball_model_data['model'])
    ball_model.to("cuda")

    keypoints_model, device = load_keypoints_model("model_epoch_18.pth")

    ball_coords, keypoints_list = process_frames(ball_model, keypoints_model, device, all_frames)

    save_video('./output_with_ball_and_keypoints.mp4', fps, all_frames, ball_coords, keypoints_list)
    print("Output video saved at: ./output_with_ball_and_keypoints.mp4")
