import heapq
import random
import matplotlib.pyplot as plt
import numpy as np

# Constants
GRID_SIZE = 16
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)

# Generate Maze (Harry Potter Style)
def generate_maze(size, obstacle_ratio=0.3):
    maze = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_ratio and (i, j) not in [START, GOAL]:
                maze[i][j] = 1

    # Add Harry Potter-themed elements (walls that shift randomly)
    shifting_walls = random.sample([(i, j) for i in range(size) for j in range(size) if maze[i][j] == 0], size // 2)
    for x, y in shifting_walls:
        if random.random() < 0.5:
            maze[x][y] = 1  # Make wall shift

    return maze

# Heuristic Function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Algorithm
def astar(maze, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []

# Visualization
def visualize_maze(maze, path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='binary')
    if path:
        for (x, y) in path:
            plt.plot(y, x, 'go')
    plt.plot(START[1], START[0], 'bs')  # Start point
    plt.plot(GOAL[1], GOAL[0], 'rs')    # Goal point
    plt.grid(True)
    plt.show()

# Main Function
def main():
    maze = generate_maze(GRID_SIZE)
    path = astar(maze, START, GOAL)
    visualize_maze(maze, path)

if __name__ == "__main__":
    main()
