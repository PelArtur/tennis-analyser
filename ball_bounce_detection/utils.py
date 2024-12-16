import csv
import numpy as np
from aeon.classification.interval_based import TimeSeriesForestClassifier
from typing import List


def calculate_speed(x_curr: int, y_curr: int, x_prev: int, y_prev: int) -> float:
    return np.sqrt(np.power(x_curr - x_prev, 2) + np.power(y_curr - y_prev, 2))


def traj_diff(delta_x, delta_y):
    return np.arctan2(delta_y, delta_x) * 180 / np.pi


def add_features(points: List[tuple[int]]) -> List[List[int | float]]:
    delta_x: float = 0.0
    delta_y: float = 0.0
    points_with_features: List[List[int | float]] = [[points[0][0], points[0][1], delta_x, delta_y, traj_diff(delta_x, delta_y), 0.0, 0.0]]
    for i in range(1, len(points)):
        if points[i][1] == -1 or points[i - 1][1] == -1:
            delta_x = 0.0
            delta_y = 0.0
            points_with_features.append([points[i][0], points[i][1], delta_x, delta_y, traj_diff(delta_x, delta_y), 0.0, 0.0])
        else:
            delta_x = points[i][0] - points[i - 1][0]
            delta_y = points[i][1] - points[i - 1][1]
            curr_speed: float = calculate_speed(points[i][0], points[i][1], points_with_features[i - 1][0], points_with_features[i - 1][1])
            points_with_features.append([points[i][0], points[i][1], delta_x, delta_y, traj_diff(delta_x, delta_y), float(curr_speed), float(curr_speed - points_with_features[i - 1][2])])
    return points_with_features


def read_dataset_data(path: str) -> List[List[int | float]]:
    dataset_data: List[List[int | float]] = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i: int = 0
        for row in spamreader:
            if i > 0:
                data = row[0].split(",")
                #game_id: int, x: int, y: int, v: float, a: float, bounce: int
                #v - velocity, a - acceleration  
                dataset_data.append([int(data[0]), int(data[1]), int(data[2]), float(data[3]), float(data[4]), int(data[5])])
            i += 1
    return dataset_data


def fill_window(dataset_data: List[List[int | float]], game_id: int, idx: int, window_size: int) -> tuple[List[List[int | float]]]:
    window: List[List[int | float]] = []
    n: int = len(dataset_data)

    if game_id != -1:
        while idx < n and dataset_data[idx][0] == game_id and len(window) < window_size:
            if dataset_data[idx][1] == -1:
                window = []
                while idx < n and dataset_data[idx][0] == game_id and dataset_data[idx][1] == -1:
                    idx += 1
            if idx >= n or dataset_data[idx][0] != game_id:
                break
            window.append(dataset_data[idx])
            idx += 1
    else:
        while idx < n and len(window) < window_size:
            if dataset_data[idx][1] == -1:
                window = []
                while idx < n and dataset_data[idx][1] == -1:
                    idx += 1
            if idx < n:
                window.append(dataset_data[idx])
                idx += 1
    return (window, idx) if len(window) == window_size else ([], idx)


def split_data(dataset_data: List[List[int | float]], total_games: int = 51, window_size: int = 30, features_range: tuple[int] = (1, 5)) -> List[List[List[int | float]]]:
    x_data: List[List[List[int | float]]] = []
    y_data: List[List[int]] = []
    y_window_data: List[int] = []
    idx: int = 0
    n: int = len(dataset_data)

    curr_window: List[List[int | float]]
    for game_id in range(total_games):
        curr_window, idx = fill_window(dataset_data, game_id, idx, window_size=window_size)
        if len(curr_window) == window_size:
            x_data.append([[curr_window[i][j] for j in range(features_range[0], features_range[1])] for i in range(window_size)])
            y_data.append([curr_window[i][features_range[1]] for i in range(window_size)])
            y_window_data.append(1 if sum(y_data[-1]) > 0 else 0)
        while idx < n and dataset_data[idx][0] == game_id:
            if dataset_data[idx][1] == -1:
                curr_window, idx = fill_window(dataset_data, game_id, idx, window_size=window_size)
            else:
                curr_window.pop(0)
                curr_window.append(dataset_data[idx])
            if len(curr_window) == window_size:
                x_data.append([[curr_window[i][j] for j in range(features_range[0], features_range[1])] for i in range(window_size)])
                y_data.append([curr_window[i][features_range[1]] for i in range(window_size)])
                y_window_data.append(1 if sum(y_data[-1]) > 0 else 0)
            idx += 1
    return x_data, y_data, y_window_data


def split_data_with_ind(dataset_data: List[List[int | float]], window_size: int = 30, features_range: tuple[int] = (1, 5)) -> List[List[List[int | float]]]:
    windows: List[List[List[int | float]]] = []
    indices: List[int] = []
    n: int = len(dataset_data)

    curr_window, idx = fill_window(dataset_data, game_id=-1, idx=0, window_size=window_size)
    if len(curr_window) == window_size:
        windows.append([[curr_window[i][j] for j in range(features_range[0], features_range[1])] for i in range(window_size)])
        indices.append(idx - window_size)
    while idx < n:
        if dataset_data[idx][1] == -1:
            curr_window, idx = fill_window(dataset_data, -1, idx, window_size=window_size)
        else:
            curr_window.pop(0)
            curr_window.append(dataset_data[idx])
        if len(curr_window) == window_size:
            windows.append([[curr_window[i][j] for j in range(features_range[0], features_range[1])] for i in range(window_size)])
            indices.append(idx - window_size)
        idx += 1
    return windows, indices


def reshape_data(dataset_data: List[List[int | float]], is_window: bool = True, window_size: int = 30) -> List[List[int | float]]:
    reshaped_data: List[List[int | float]] = []
    if is_window:
        for i in range(len(dataset_data)):
            reshaped_data.append([
                [dataset_data[i][j][0] for j in range(window_size)],  #x
                [dataset_data[i][j][1] for j in range(window_size)],  #y
                [dataset_data[i][j][2] for j in range(window_size)],  #delta x
                [dataset_data[i][j][3] for j in range(window_size)],  #delta y
                [dataset_data[i][j][4] for j in range(window_size)],  #traj
                [dataset_data[i][j][5] for j in range(window_size)],  #b
                [dataset_data[i][j][6] for j in range(window_size)]   #a
            ])
    else:
        for i in range(len(dataset_data)):
            reshaped_data.append([
                [dataset_data[i][0]],   #x
                [dataset_data[i][1]],   #y
                [dataset_data[i][2]],   #v
                [dataset_data[i][3]],   #a
            ])
    return reshaped_data


def delete_outofbound_points(dataset_data: List[List[int | float]]) -> List[List[List[int | float]]]:
    x_data: List[List[List[int | float]]] = []
    y_data: List[List[List[int | float]]] = []

    for data in dataset_data:
        if data[1] != -1 and data[2] != -1:
            x_data.append([data[i] for i in range(1, 5)])
            y_data.append(data[5])
    return x_data, y_data


# def process_bounce_points(dataset_data: List[List[int | float]], tsfc5: TimeSeriesForestClassifier, tsfc10: TimeSeriesForestClassifier) -> List[int]:
#     print(f"Input size: {len(dataset_data)}")
#     windows, indices = split_data_with_ind(dataset_data, 10, features_range=(0, 4))
#     windows = reshape_data(windows, window_size=10)
#     is_bounce: List[bool] = [False for _ in range(len(dataset_data))]

#     predicts = tsfc10.predict(np.array(windows, dtype=np.float32))
#     for i in range(len(windows) - 1):
#         if indices[i + 1] - indices[i] == 1 and predicts[i + 1] == 0 and predicts[i] == 1:
#             is_bounce[i] = True

#     windows, indices = split_data_with_ind(dataset_data, 5, features_range=(0, 4))
#     windows = reshape_data(windows, window_size=5)  

#     predicts = tsfc5.predict(np.array(windows, dtype=np.float32))
#     for i in range(len(windows) - 10, len(windows) - 1):
#         if indices[i + 1] - indices[i] == 1 and predicts[i + 1] == 0 and predicts[i] == 1:
#             is_bounce[i] = True

#     bounce_frames: List[int] = []
#     for i in range(len(is_bounce)):
#         if is_bounce[i]:
#             bounce_frames.append(i)
#     return bounce_frames


def process_bounce_points(dataset_data: List[List[int | float]], tsfc15: TimeSeriesForestClassifier) -> List[int]:
    windows, indices = split_data_with_ind(dataset_data, 15, features_range=(0, 7))
    windows = reshape_data(windows, window_size=15)
    is_bounce: List[bool] = [False for _ in range(len(dataset_data))]

    predicts = tsfc15.predict(np.array(windows, dtype=np.float32))
    for i in range(len(windows) - 1):
        if indices[i + 1] - indices[i] == 1 and predicts[i + 1] == 0 and predicts[i] == 1:
            is_bounce[i] = True

    bounce_frames: List[int] = []
    for i in range(len(is_bounce)):
        if is_bounce[i]:
            bounce_frames.append(i)
    return bounce_frames
