import cv2
import numpy as np
from scipy.special import erf
from collections import deque

def diamond_square(matrix, size, seed, roughness, x, y, step):
    half = step // 2
    if half < 1:
        return

    # Diamond step
    a = matrix[x][y]
    b = matrix[x + step][y]
    c = matrix[x][y + step]
    d = matrix[x + step][y + step]

    average = (a + b + c + d) / 4.0
    matrix[x + half][y + half] = average + np.random.uniform(-1, 1) * roughness

    def get_value(x, y):
        if x < 0 or x >= size or y < 0 or y >= size:
            return 0
        else:
            return matrix[x][y]

    # Square step
    matrix[x + half][y] = (a + b + get_value(x+half, y-half) + get_value(x+half, y+half)) / 4.0
    matrix[x][y + half] = (a + c + get_value(x-half, y+half) + get_value(x+half, y+half)) / 4.0
    matrix[x + step][y + half] = (b + d + get_value(x+half, y+half) + get_value(x+step+half, y+half)) / 4.0
    matrix[x + half][y + step] = (c + d + get_value(x+half, y+half) + get_value(x+half, y+step+half)) / 4.0

    # Recurse into the four quadrants
    diamond_square(matrix, size, seed, roughness, x, y, half)
    diamond_square(matrix, size, seed, roughness, x + half, y, half)
    diamond_square(matrix, size, seed, roughness, x, y + half, half)
    diamond_square(matrix, size, seed, roughness, x + half, y + half, half)

def generate_terrain(n, roughness, seed):
    if not (n-1 > 0 and ((n-1) & (n-2) == 0)):
        raise ValueError("n-1 should be a power of 2")

    # Initialize the grid
    matrix = np.zeros((n, n))
    matrix[0][0] = seed
    matrix[0][n - 1] = seed
    matrix[n - 1][0] = seed
    matrix[n - 1][n - 1] = seed

    step = n - 1

    diamond_square(matrix, n, seed, roughness, 0, 0, step)
    matrix[0][0] = (matrix[0][1] + matrix[1][0]) // 2
    matrix[0][n-1] = (matrix[0][n-2] + matrix[1][n-1]) // 2
    matrix[n-1][0] = (matrix[n-2][0] + matrix[n-1][1]) // 2
    matrix[n-1][n-1] = (matrix[n-2][n-1] + matrix[n-1][n-2]) // 2
    return matrix

def sig_mat(terrain, ratio=0.5):
    terrain -= 127
    return 127 * \
           ((np.tanh(terrain * 2 / 127) * ratio +
             erf(terrain * 2.5 / 127) * (1 - ratio)) + 1)

def add_noise_on_planar_section(terrain, min_region_size=100, mean=0, sigma=5, diff_thresh=0):
    height, width = terrain.shape
    visited = np.zeros((height, width))

    def bfs(terrain, start=(0, 0)):
        if visited[start]:
            return []

        rows, cols = terrain.shape
        queue = [start]
        visited[start] = True
        regions = []

        while queue:
            current_node = queue.pop(-1)
            regions.append(current_node)
            neighbors = []
            row, col = current_node
            if row > 0 and not visited[row - 1, col] \
                    and abs(terrain[current_node] - terrain[row - 1, col]) <= diff_thresh:
                neighbors.append((row - 1, col))
                visited[row - 1, col] = True

            if row < rows - 1 and not visited[row + 1, col] \
                    and abs(terrain[current_node] - terrain[row + 1, col]) <= diff_thresh:
                neighbors.append((row + 1, col))
                visited[row + 1, col] = True

            if col > 0 and not visited[row, col - 1] \
                    and abs(terrain[current_node] - terrain[row, col - 1]) <= diff_thresh:
                neighbors.append((row, col - 1))
                visited[row, col - 1] = True

            if col < cols - 1 and not visited[row, col + 1] \
                    and abs(terrain[current_node] - terrain[row, col - 1]) <= diff_thresh:
                neighbors.append((row, col + 1))
                visited[row, col + 1] = True

            queue.extend(neighbors)
        return np.array(regions)

    for i in range(height):
        for j in range(width):
            regions = bfs(terrain, start=(i, j))
            if len(regions) >= min_region_size:
                regions = np.array(regions)
                row_indices = regions[:, 0]
                col_indices = regions[:, 1]

                noise = np.random.normal(mean, sigma, len(regions))
                terrain[row_indices, col_indices] = np.maximum(0, terrain[row_indices, col_indices] + noise)

    return terrain
