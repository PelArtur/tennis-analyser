import csv
from typing import List
from copy import deepcopy


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

    while idx < n and dataset_data[idx][0] == game_id and len(window) < window_size:
        if dataset_data[idx][1] == -1:
            window = []
            while idx < n and dataset_data[idx][0] == game_id and dataset_data[idx][1] == -1:
                idx += 1
        if idx >= n or dataset_data[idx][0] != game_id:
            break
        window.append(dataset_data[idx])
        idx += 1
    return (window, idx) if len(window) == window_size else ([], idx)


def split_data(dataset_data: List[List[int | float]], total_games: int = 51, window_size: int = 30) -> List[List[List[int | float]]]:
    x_data: List[List[List[int | float]]] = []
    y_data: List[List[int]] = []
    y_window_data: List[int] = []
    idx: int = 0
    n: int = len(dataset_data)

    curr_window: List[List[int | float]]
    for game_id in range(total_games):
        curr_window, idx = fill_window(dataset_data, game_id, idx, window_size=window_size)
        if len(curr_window) == window_size:
            x_data.append([[curr_window[i][j] for j in range(1, 5)] for i in range(window_size)])
            y_data.append([curr_window[i][5] for i in range(window_size)])
            y_window_data.append(1 if sum(y_data[-1]) > 0 else 0)
        while idx < n and dataset_data[idx][0] == game_id:
            if dataset_data[idx][1] == -1:
                curr_window, idx = fill_window(dataset_data, game_id, idx, window_size=window_size)
            else:
                curr_window.pop(0)
                curr_window.append(dataset_data[idx])
            if len(curr_window) == window_size:
                x_data.append([[curr_window[i][j] for j in range(1, 5)] for i in range(window_size)])
                y_data.append([curr_window[i][5] for i in range(window_size)])
                y_window_data.append(1 if sum(y_data[-1]) > 0 else 0)
            idx += 1
    return x_data, y_data, y_window_data


def reshape_data(dataset_data: List[List[int | float]], window_size: int = 30) -> List[List[int | float]]:
    reshaped_data: List[List[int | float]] = []
    for i in range(len(dataset_data)):
        reshaped_data.append([
            [dataset_data[i][j][0] for j in range(window_size)],  #x
            [dataset_data[i][j][1] for j in range(window_size)],  #y
            [dataset_data[i][j][2] for j in range(window_size)],  #v
            [dataset_data[i][j][3] for j in range(window_size)]   #a
        ])
    return reshaped_data
