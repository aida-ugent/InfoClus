import numpy as np

v = np.array([[1, 2, 3], [3, 4, 5]])
data = np.array([[1, 2, 3], [3, 1, 5], [4, 6, 7], [7, 8, 9]])

RADIUS = 5

def get_neighbor(point: np.ndarray, data_ld: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(data_ld - point, axis=1)
    mask = distances <= RADIUS
    print(mask)
    return data_ld[mask]


neighbor = get_neighbor(data[0], data)
print(neighbor)
print(np.argmin(data[1]))