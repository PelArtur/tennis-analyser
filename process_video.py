import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import List
from ball_detector.model import TrackNet
from ball_detector.utils import extract_ball_center
 
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


def process_frames(model: nn.Module, frames: List[np.ndarray], height: int = 360, width: int = 640, device: str = "cuda") -> List[tuple[float]]:
    ball_coords: List[tuple[float]] = [(-1.0, -1.0), (-1.0, -1.0)]
    torch.cuda.empty_cache() 
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(2, len(frames))):
            input = np.concatenate([
                cv.cvtColor(cv.resize(frames[i - 2], (width, height)), cv.COLOR_BGR2RGB),
                cv.cvtColor(cv.resize(frames[i - 1], (width, height)), cv.COLOR_BGR2RGB),
                cv.cvtColor(cv.resize(frames[i],     (width, height)), cv.COLOR_BGR2RGB)
            ], axis=2)
            input = np.transpose(input, (2, 0, 1))
            input = input.astype(np.float32) / 255.0

            output = model(torch.tensor(np.array([input])).to(device))
            x, y = extract_ball_center(output.argmax(dim=1).to('cpu').numpy()[0])
            ball_coords.append((x, y))
    return ball_coords
 

def save_video(video_path: str, fps: int, frames: List[np.ndarray], coords: List[tuple[float]], height: int = 360, width: int = 640) -> None:
    out_height, out_width = frames[0].shape[0], frames[0].shape[1] 
    y_scaling, x_scaling = out_height / height, out_width / width
    
    output = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (out_width, out_height))
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        for j in range(10):
            if i - j >= 0 and coords[i - j][0] != -1 and coords[i - j][1] != -1:
                overlay = frame.copy()
                overlay = cv.circle(frame, (int(coords[i - j][0] * x_scaling), int(coords[i - j][1] * y_scaling)), int(5 - 0.5 * j), (0, 0, 255), -1)
                alpha = 1 - 0.1 * j
                frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        output.write(frame)
    output.release()


if __name__ == "__main__":
    video_path = './input_video.mp4'
    all_frames, fps = extract_frames(video_path)

    model = TrackNet()
    model_data = torch.load("model_final.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(model_data['model'])
    model.to("cuda")

    ball_coord = process_frames(model, all_frames)
    save_video('./output.mp4', fps, all_frames, ball_coord)
