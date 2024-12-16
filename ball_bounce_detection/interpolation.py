import numpy as np
from math import ceil


def traj_diff(delta_x, delta_y):
    return np.arctan2(delta_y, delta_x) * 180 / np.pi


def fix_one(data, ind):
    data[ind - 1] = ((data[ind][0] + data[ind - 2][0]) / 2, (data[ind][1] + data[ind - 2][1]) / 2)


def fix_two(data, ind):
    data[ind - 2] = (data[ind - 3][0] + (data[ind][0] - data[ind - 3][0]) / 2, data[ind - 3][1] + (data[ind][1] - data[ind - 3][1]) / 2)
    fix_one(data, ind)


def fix_three_to_six(data, ind, n):
    if ind - n - 2 < 0 or data[ind - n - 2] == (-1, -1):
        return
    if ind + 1 >= len(data) or data[ind + 1] == (-1, -1):
        return
    
    last_two = [data[ind - n - 2], data[ind - n - 1]]
    first_two = [data[ind], data[ind + 1]]

    start_phi = traj_diff(last_two[1][0] - last_two[0][0], last_two[1][1] - last_two[0][1])
    start_x, start_y = last_two[1][0], last_two[1][1]
    end_phi = traj_diff(first_two[0][0] - first_two[1][0], first_two[0][1] - first_two[1][1])
    end_x, end_y = first_two[0][0], first_two[0][1]

    for i in range(ceil((n - 2) / 2)):
        prev_x, prev_y, prev_phi = start_x, start_y, start_phi
        start_phi += (end_phi - start_phi) / (n - 2 * i)
        start_x += (end_x - start_x) * np.cos(start_phi * np.pi / 180) / (n - 2 * i)
        start_y += (end_y - start_y) * np.cos(start_phi * np.pi / 180) / (n - 2 * i)

        end_phi += (prev_phi - end_phi) / (n - 2 * i)
        end_x += (prev_x - end_x) * np.cos(end_phi * np.pi / 180) / (n - 2 * i)
        end_y += (prev_y - end_y) * np.cos(end_phi * np.pi / 180) / (n - 2 * i)


        data[ind - n + i] = (start_x, start_y)
        data[ind - 1 - i] = (end_x, end_y)

    if n % 2 == 1:
        fix_one(data, ind - ceil((n - 2) / 2))
    else:
        fix_two(data, ind - ceil((n - 2) / 2))


def interpolation(data):
    ind: int = 2
    minus_1: int = 0
    while ind < len(data):
        if data[ind][0] == -1 and minus_1 == 0:
            minus_1 = 1
        elif data[ind][0] == -1 and minus_1 > 0:
            minus_1 += 1
        else:
            if minus_1 == 1:
                fix_one(data, ind)
            if minus_1 == 2:
                fix_two(data, ind)
            minus_1 = 0
        ind += 1

    for i in range(2, len(data) - 1):
        if data[i - 1] == (-1, -1) and data[i + 1] == (-1, -1):
            data[i] = (-1, -1)

    minus_1: int = 0
    ind = 2
    while ind < len(data):
        if data[ind][0] == -1 and minus_1 == 0:
            minus_1 = 1
        elif data[ind][0] == -1 and minus_1 > 0:
            minus_1 += 1
        else:
            if 3 <= minus_1 <= 10:
                fix_three_to_six(data, ind, minus_1)
            minus_1 = 0            
        ind += 1