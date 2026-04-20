
#CTED_RIU4_LQP1_4


#  --- si(x) function, returns 0 or 1 for LQP_i (including the d(x) part)


def threshold_func(delta, mode, T1=2, T2=5):
    if mode == 'LQP1':
        return (delta >= T2).astype(np.uint8)
    elif mode == 'LQP2':
        return (delta <= -T2).astype(np.uint8)
    elif mode == 'LQP3':
        return (abs(delta) <= T1).astype(np.uint8)
    elif mode == 'LQP4':
        return (abs(delta) > T1).astype(np.uint8)
    else:
        raise ValueError("Invalid mode. Choose from 'LQP1', 'LQP2', 'LQP3', 'LQP4'.")




import numpy as np
from scipy.ndimage import map_coordinates
from math import ceil


def circular_1_segments(pattern):
    P = len(pattern)
    extended = pattern + pattern
    segments = []
    count = 0
    for i in range(2 * P):
        if extended[i] == 1:
            count += 1
        elif count > 0:
            segments.append(count)
            count = 0
    if count > 0:
        segments.append(count)
    if pattern[0] == 1 and pattern[-1] == 1 and len(segments) > 1:
        segments[0] += segments.pop()
    return sorted(segments)


# a part of riu4-LQP design

def compute_index(P, pattern):
    segments = circular_1_segments(pattern)
    if len(segments) != 2:
        raise ValueError("T1 == 4 but did not find exactly two 1-segments.")
    X, Y = segments
    if X == 1:
        return Y
    else:
        return sum((P - 3 - 2 * (n - 1)) + (Y - X + 1) for n in range(1, X))



def compute_RIU4_LQP(image, R=1, P=8, mode='LQP1', T1=2, T2=5):
    """
    Compute the RIU4-LQP{1,2,3,4} descriptor for a grayscale image.

    Parameters:
        image: 2D grayscale NumPy array
        R: Radius of neighborhood (e.g. 1)
        P: Number of neighbors (e.g. 8)
        mode: 'LQP1', 'LQP2', 'LQP3', or 'LQP4'
        T1: Threshold for LQP3 and LQP4
        T2: Threshold for LQP1 and LQP2

    Returns:
        2D array of same shape as input, with RIU4-LQP codes
    """
    H, W = image.shape
    I_c = image.astype(np.float32)
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    s_patterns = np.zeros((H, W, P), dtype=np.uint8)

    for p in range(P):
        theta = 2 * np.pi * p / P
        dx = R * np.cos(theta)
        dy = -R * np.sin(theta)
        coords = np.array([Y + dy, X + dx])
        I_n = map_coordinates(I_c, coords, order=1, mode='reflect')
        delta = I_n - I_c
        s_patterns[:, :, p] = threshold_func(delta, mode, T1, T2)

    output = np.zeros((H, W), dtype=np.uint8)
    uniform_threshold = ceil((P ** 2 + 11) / 4) - 1

    for i in range(H):
        for j in range(W):
            pattern = s_patterns[i, j, :].tolist()
            transitions = sum(abs(pattern[p] - pattern[p - 1]) for p in range(P))
            T_val = transitions

            if T_val == 0 or T_val == 2:
                output[i, j] = sum(pattern)
            elif T_val == 4:
                try:
                    idx = compute_index(P, pattern)
                    output[i, j] = P + idx
                except:
                    output[i, j] = uniform_threshold
            else:
                output[i, j] = uniform_threshold

    return output











