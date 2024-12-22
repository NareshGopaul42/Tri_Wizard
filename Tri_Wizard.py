import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
GRID_SIZE = 16
OBSTACLE_PROB = 0.3  # Probability of placing extra obstacles
random.seed(None)       # Just for reproducibility; you can remove this


# -----------------------------
# MAZE GENERATION
# -----------------------------
def generate_maze(size, obstacles=0.3):
    """
    Creates a random maze using DFS-based carving plus additional loops,
    then adds random obstacles while maintaining connectivity
    between (0,0) and (size-1, size-1).
    """
    maze = np.ones((size, size), dtype=int)  # 1=wall, 0=free
    stack = [(0, 0)]
    maze[0, 0] = 0

    # Helper to get neighbors 2 steps away (carving)
    def get_neighbors(x, y):
        nbrs = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                nbrs.append((nx, ny))
        return nbrs

    # DFS Carving
    while stack:
        x, y = stack[-1]
        nbrs = get_neighbors(x, y)
        if nbrs:
            nx, ny = random.choice(nbrs)
            # Carve path
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Add random loops
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if maze[i, j] == 1 and random.random() < 0.1:
                maze[i, j] = 0

    # Now place additional obstacles but ensure connectivity from (0,0) to (size-1,size-1)
    maze = add_obstacles(maze, obstacles, (0, 0), (size - 1, size - 1))
    return maze


def is_connected(maze, start, goal):
    """
    BFS to check if there's a path from start to goal in the grid.
    """
    queue = deque([start])
    visited = set([start])
    n = len(maze)

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < n:
                if maze[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


def add_obstacles(maze, obstacle_prob, start, goal):
    """
    Randomly adds extra walls (1) while preserving connectivity
    between 'start' and 'goal'. If adding a wall blocks the path,
    it reverts that wall.
    """
    size = len(maze)
    retries = 0
    max_retries = 200
    while retries < max_retries:
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        # Don't block around the goal
        if abs(i - goal[0]) <= 1 and abs(j - goal[1]) <= 1:
            retries += 1
            continue

        if maze[i, j] == 0 and random.random() < obstacle_prob:
            maze[i, j] = 1
            if not is_connected(maze, start, goal):
                # revert
                maze[i, j] = 0
            else:
                retries = 0
        retries += 1
    return maze


# -----------------------------
# RANDOM POSITIONS
# -----------------------------
def generate_random_positions(maze, count):
    """
    Picks 'count' unique random positions in open cells (where maze==0).
    Returns a list of (x,y) tuples.
    """
    positions = []
    tries = 0
    max_tries = 500
    n = len(maze)
    while len(positions) < count and tries < max_tries:
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        if maze[x, y] == 0 and (x, y) not in positions:
            positions.append((x, y))
        tries += 1
    return positions


# -----------------------------
# A* ALGORITHM (Full Path)
# -----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_full_path(maze, start, goal):
    """
    Standard A* that returns the ENTIRE path from start to goal, or empty if none.
    """
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze):
                if maze[nx, ny] == 0:  # passable
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (f_score[(nx, ny)], (nx, ny)))
    return []  # no path


# -----------------------------
# VISUALIZE SINGLE-AGENT PATH
# -----------------------------
def visualize_single_path(maze, start, goal, path):
    """
    Plots the maze (0=free, 1=wall) with a single path from start -> goal.
    """
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap='binary', vmin=0, vmax=1)
    ax.plot(start[1], start[0], 'bs', label='Start')
    ax.plot(goal[1], goal[0], 'rs', label='Goal')

    if len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(ys, xs, 'g-', label='Path')

    ax.set_title("Single-Agent A* Result")
    ax.legend()
    plt.show()


# -----------------------------
# TURN-BY-TURN A* (Next Step)
# -----------------------------
def astar_next_step(maze, start, goal, blocked):
    """
    A* that returns only the NEXT step toward the goal,
    treating 'blocked[x][y]' as off-limits.
    If no path, returns None.
    """
    if start == goal:
        return start

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if len(path) > 1:
                return path[1]  # the next step
            return start

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze):
                # must be free in the maze AND not blocked
                if maze[nx, ny] == 0 and not blocked[nx][ny]:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (f, (nx, ny)))
    return None


# -----------------------------
# TURN-BASED ADVERSARIAL CLASS
# -----------------------------
class TurnBasedAdversarial:
    """
    Orchestrates two agents (A and B) moving in a maze turn-by-turn.
    - Each agent blocks the cells it has visited so the other cannot enter them.
    - Agent A moves one step, then B moves one step, etc.
    """
    def __init__(self, maze, startA, startB, goal):
        self.maze = maze
        self.posA = startA
        self.posB = startB
        self.goal = goal

        # Track visited cells
        self.visited_by_A = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.visited_by_B = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

        self.visited_by_A[startA[0], startA[1]] = True
        self.visited_by_B[startB[0], startB[1]] = True

        self.turn = 'A'
        self.done = False

        # Setup plotting
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(maze, cmap='binary', vmin=0, vmax=1)
        # Plot initial positions
        self.ax.plot(startA[1], startA[0], 'bo', label='Agent A')
        self.ax.plot(startB[1], startB[0], 'ro', label='Agent B')
        self.ax.plot(goal[1], goal[0], 'gs', label='Goal')
        self.ax.legend()

    def update(self, frame):
        """
        Called each frame by FuncAnimation to move the active agent one step.
        """
        if self.done:
            return

        # Check if anyone reached the goal
        if self.posA == self.goal or self.posB == self.goal:
            self.done = True
            return

        if self.turn == 'A':
            # Agent A tries one step
            next_cell = astar_next_step(self.maze, self.posA, self.goal, blocked=self.visited_by_B)
            if next_cell and next_cell != self.posA:
                self.posA = next_cell
                self.visited_by_A[self.posA[0], self.posA[1]] = True
            self.turn = 'B'
        else:
            # Agent B tries one step
            next_cell = astar_next_step(self.maze, self.posB, self.goal, blocked=self.visited_by_A)
            if next_cell and next_cell != self.posB:
                self.posB = next_cell
                self.visited_by_B[self.posB[0], self.posB[1]] = True
            self.turn = 'A'

        self.ax.clear()
        self.ax.imshow(self.maze, cmap='binary', vmin=0, vmax=1)

        # Plot positions
        self.ax.plot(self.posA[1], self.posA[0], 'bo', label='Agent A')
        self.ax.plot(self.posB[1], self.posB[0], 'ro', label='Agent B')
        self.ax.plot(self.goal[1], self.goal[0], 'gs', label='Goal')

        # Optionally show visited cells as light markers
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.visited_by_A[x,y]:
                    self.ax.plot(y, x, marker='.', color='blue', alpha=0.2)
                if self.visited_by_B[x,y]:
                    self.ax.plot(y, x, marker='.', color='red', alpha=0.2)

        self.ax.set_title(f"Turn: {self.turn}")
        self.ax.legend()


# -----------------------------
# DEMO 1: Single-Agent
# -----------------------------
def single_agent_demo():
    """
    1) Generate a maze
    2) Pick a single random start and single random goal
    3) Use A* to get the entire path
    4) Visualize the resulting path
    """
    maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)
    positions = generate_random_positions(maze, 2)
    if len(positions) < 2:
        print("Not enough open cells found for start & goal!")
        return

    start, goal = positions[0], positions[1]
    print(f"Single-Agent Scenario:\n  Start = {start}\n  Goal  = {goal}")

    path = astar_full_path(maze, start, goal)
    if not path:
        print("No path found!")
    visualize_single_path(maze, start, goal, path)


# -----------------------------
# DEMO 2: Turn-Based Adversarial
# -----------------------------
def adversarial_agents_turn_based():
    """
    1) Generate a maze
    2) Pick random starts for A and B, and a random goal
    3) Animate turn-by-turn movement with blocked visitation.
    """
    maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)

    positions = generate_random_positions(maze, 3)
    if len(positions) < 3:
        print("Not enough open cells for A, B, and goal!")
        return
    startA, startB, goal = positions[0], positions[1], positions[2]
    print(f"Turn-Based Adversarial:\n  Agent A = {startA}\n  Agent B = {startB}\n  Goal    = {goal}")

    game = TurnBasedAdversarial(maze, startA, startB, goal)
    ani = FuncAnimation(game.fig, game.update, frames=300, interval=300, repeat=False)
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Run whichever scenario you want:
    
    # 1) Single-Agent Demo
    single_agent_demo()
    
    # 2) Turn-Based Adversarial Demo
    adversarial_agents_turn_based()
