import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
import time

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
GRID_SIZE = 16
OBSTACLE_PROB = 0.3
random.seed(None)  # Remove or set a fixed seed for reproducibility

# -----------------------------
# MAZE GENERATION
# -----------------------------
def generate_maze(size, obstacles=0.3):
    maze = np.ones((size, size), dtype=int)
    stack = [(0, 0)]
    maze[0, 0] = 0

    def get_neighbors(x, y):
        nbrs = []
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                nbrs.append((nx, ny))
        return nbrs

    # First path carving
    while stack:
        x, y = stack[-1]
        nbrs = get_neighbors(x, y)
        if nbrs:
            nx, ny = random.choice(nbrs)
            maze[(x + nx)//2, (y + ny)//2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Start second path from middle of left side
    mid = size // 2
    if maze[mid, 1] == 1:  # If wall exists here
        stack = [(mid, 0)]
        maze[mid, 0] = 0
        
        while stack:  # Carve second path
            x, y = stack[-1]
            nbrs = get_neighbors(x, y)
            if nbrs:
                nx, ny = random.choice(nbrs)
                maze[(x + nx)//2, (y + ny)//2] = 0
                maze[nx, ny] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    # Add more connections between paths
    for i in range(1, size-1):
        for j in range(1, size-1):
            if maze[i, j] == 1:
                open_neighbors = sum(1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] 
                                  if 0 <= i+dx < size and 0 <= j+dy < size 
                                  and maze[i+dx][j+dy] == 0)
                if open_neighbors >= 2 and random.random() < 0.3:
                    maze[i, j] = 0

    return maze  # Return maze before adding obstacles

def is_connected(maze, start, goal):
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
    size = len(maze)
    retries = 0
    max_retries = 100
    
    while retries < max_retries:
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        
        # Keep larger area around goal clear
        if abs(i - goal[0]) <= 2 and abs(j - goal[1]) <= 2:
            retries += 1
            continue
            
        if maze[i, j] == 0 and random.random() < obstacle_prob:
            maze[i, j] = 1
            # Check if still connected
            if not is_connected(maze, start, goal):
                maze[i, j] = 0
            
        retries += 1
    return maze

def generate_random_positions(maze, count):
    positions = []
    tries = 0
    max_tries = 500
    n = len(maze)
    while len(positions) < count and tries < max_tries:
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if maze[x, y] == 0 and (x, y) not in positions:
            positions.append((x, y))
        tries += 1
    return positions

# -----------------------------
# HELPER: Heuristic
# -----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -----------------------------
# FULL A* for final path reconstruction
# -----------------------------
def astar_full_path_entire(maze, start, goal):
    """
    A standard full A* to get the entire path once we're done.
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
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if maze[nx, ny] == 0:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (f_score[(nx, ny)], (nx, ny)))
    return []

# -----------------------------
# SINGLE-AGENT PARTIAL A* STEP
# -----------------------------
def astar_next_step_single(maze, start, goal, expansions_list):
    """
    Single-agent step-by-step.
    expansions_list is a single-element list so we can increment expansions.
    """
    if start == goal:
        return start

    open_list = []
    came_from = {}
    g_score = {start: 0}
    expansions = 0

    heapq.heappush(open_list, (0, start))

    while open_list:
        _, current = heapq.heappop(open_list)
        expansions += 1

        if current == goal:
            expansions_list[0] += expansions
            # reconstruct just next step
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if len(path) > 1:
                return path[1]
            return start

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if maze[nx, ny] == 0:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (f, (nx, ny)))

    expansions_list[0] += expansions
    return None

# -----------------------------
# SINGLE-AGENT SCENARIO (STEP-BY-STEP)
# -----------------------------
class SingleAgentScenario:
    """
    Step-by-step single agent. We track expansions + path length,
    show the agent's trail as it moves, and finally draw the full path in green.
    """
    def __init__(self, ax, maze):
        self.ax = ax
        self.maze = maze

        # Start & goal
        pos = generate_random_positions(maze, 2)
        self.start = pos[0]
        self.goal = pos[1]

        self.current = self.start
        self.done = False

        # We'll store expansions in a one-element list
        self.expansions = [0]
        self.path_len = 0

        # For showing the agent's TRIAL or "footsteps"
        self.agent_trail = [self.start]

        # We will store the final path
        self.final_path = []

        # Plot initial
        self.ax.imshow(self.maze, cmap='binary', vmin=0, vmax=1)
        self.ax.plot(self.start[1], self.start[0], 'bs', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'rs', label='Goal')

    def update(self):
        if self.done:
            return

        # Check if we've arrived
        if self.current == self.goal:
            self.done = True
            self.final_path = astar_full_path_entire(self.maze, self.start, self.goal)
            self.draw()
            return

        # Attempt one step
        step = astar_next_step_single(self.maze, self.current, self.goal, self.expansions)
        if step and step != self.current:
            self.current = step
            self.path_len += 1
            self.agent_trail.append(step)
        else:
            # Stuck or no path
            self.done = True
            self.final_path = astar_full_path_entire(self.maze, self.start, self.goal)

        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.maze, cmap='binary', vmin=0, vmax=1)
        self.ax.plot(self.start[1], self.start[0], 'bs', label='Start')
        self.ax.plot(self.goal[1], self.goal[0], 'rs', label='Goal')

        # Mark the trail as it moves
        if len(self.agent_trail) > 1:
            xs = [p[0] for p in self.agent_trail]
            ys = [p[1] for p in self.agent_trail]
            self.ax.plot(ys, xs, 'y.-', label='Trail')  # agent's route so far

        # Current agent position
        self.ax.plot(self.current[1], self.current[0], 'go', label='Agent')

        # If done, draw the final path in green
        if self.done and self.final_path:
            xs = [p[0] for p in self.final_path]
            ys = [p[1] for p in self.final_path]
            self.ax.plot(ys, xs, 'g-', label='Final Path')

        self.ax.set_title(f"Single Agent: steps={self.path_len}, expansions={self.expansions[0]}")
        self.ax.legend()

# -----------------------------
# ADVERSARIAL PARTIAL A* STEP
# (skip cells visited by opponent)
# -----------------------------
def adversarial_next_step(maze, start, goal, blocked, expansions_list):
    """
    Each agent tries to move 1 step closer to the goal,
    but cannot step on cells 'blocked' by the opponent.
    expansions_list to track expansions.
    """
    if start == goal:
        return start

    open_list = []
    came_from = {}
    g_score = {start: 0}
    expansions = 0

    heapq.heappush(open_list, (0, start))

    while open_list:
        _, current = heapq.heappop(open_list)
        expansions += 1

        if current == goal:
            expansions_list[0] += expansions
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if len(path) > 1:
                return path[1]
            return start

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                # Maze cell must be free and not visited by opponent
                if maze[nx, ny] == 0 and not blocked[nx, ny]:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_list, (f, (nx, ny)))

    expansions_list[0] += expansions
    return None

# -----------------------------
# ADVERSARIAL SCENARIO
# -----------------------------
class AdversarialScenario:
    """
    Two agents, A and B, turn-based. Each blocks the other’s visited cells.
    Steps are partial A* calls. We track expansions, path length, etc.
    """
    def __init__(self, ax, maze):
        self.ax = ax
        self.maze = maze

        pos = generate_random_positions(maze, 3)
        self.startA = pos[0]
        self.startB = pos[1]
        self.goal  = pos[2]

        self.posA = self.startA
        self.posB = self.startB

        self.visitedA = np.zeros_like(maze, dtype=bool)
        self.visitedB = np.zeros_like(maze, dtype=bool)
        self.visitedA[self.posA[0], self.posA[1]] = True
        self.visitedB[self.posB[0], self.posB[1]] = True

        self.turn = 'A'
        self.done = False

        self.expansionsA = [0]
        self.expansionsB = [0]
        self.path_lenA = 0
        self.path_lenB = 0

        # We'll store partial "trails" for each agent
        self.trailA = [self.posA]
        self.trailB = [self.posB]

    def update(self):
        if self.done:
            return

        # If either agent is at the goal, scenario ends
        if self.posA == self.goal or self.posB == self.goal:
            self.done = True
            return

        if self.turn == 'A':
            nxt = adversarial_next_step(self.maze, self.posA, self.goal, self.visitedB, self.expansionsA)
            if nxt and nxt != self.posA:
                self.posA = nxt
                self.trailA.append(nxt)
                self.visitedA[self.posA[0], self.posA[1]] = True
                self.path_lenA += 1
            self.turn = 'B'
        else:
            nxt = adversarial_next_step(self.maze, self.posB, self.goal, self.visitedA, self.expansionsB)
            if nxt and nxt != self.posB:
                self.posB = nxt
                self.trailB.append(nxt)
                self.visitedB[self.posB[0], self.posB[1]] = True
                self.path_lenB += 1
            self.turn = 'A'

        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.maze, cmap='binary', vmin=0, vmax=1)

        # Draw partial "trails"
        if len(self.trailA) > 1:
            xsA = [p[0] for p in self.trailA]
            ysA = [p[1] for p in self.trailA]
            self.ax.plot(ysA, xsA, 'b.-', alpha=0.5, label='A trail')

        if len(self.trailB) > 1:
            xsB = [p[0] for p in self.trailB]
            ysB = [p[1] for p in self.trailB]
            self.ax.plot(ysB, xsB, 'r.-', alpha=0.5, label='B trail')

        # Agents
        self.ax.plot(self.posA[1], self.posA[0], 'bo', 
                     label=f"A expansions={self.expansionsA[0]}")
        self.ax.plot(self.posB[1], self.posB[0], 'ro', 
                     label=f"B expansions={self.expansionsB[0]}")

        # Goal
        self.ax.plot(self.goal[1], self.goal[0], 'gs', label='Goal')

        # Mark visited cells
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.visitedA[x, y]:
                    self.ax.plot(y, x, marker='.', color='blue', alpha=0.15)
                if self.visitedB[x, y]:
                    self.ax.plot(y, x, marker='.', color='red', alpha=0.15)

        self.ax.set_title(f"A steps={self.path_lenA}, B steps={self.path_lenB}, Turn={self.turn}")
        self.ax.legend()

def write_metrics_to_file(metrics_data, filename="metrics.txt"):
    """
    Write metrics to a file in a structured format for later analysis.

    metrics_data format:
    {
        'scenario_type': str,  # 'single' or 'adversarial'
        'maze_number': int,
        'time_taken': float,
        'total_expansions': int,
        'path_length': int,
        'reached_goal': bool,
        'agent_a_expansions': Optional[int],  # Only for adversarial
        'agent_b_expansions': Optional[int],  # Only for adversarial
        'agent_a_path_length': Optional[int], # Only for adversarial
        'agent_b_path_length': Optional[int]  # Only for adversarial
    }
    """
    with open(filename, 'a') as f:
        # Write header if file is empty
        if f.tell() == 0:
            f.write("---- METRICS LOG ----\n\n")

        f.write(f"Scenario: {metrics_data['scenario_type']}\n")
        f.write(f"Maze Number: {metrics_data['maze_number']}\n")
        f.write(f"Time Taken: {metrics_data['time_taken']:.3f} seconds\n")
        f.write(f"Total Expansions: {metrics_data['total_expansions']}\n")
        f.write(f"Path Length: {metrics_data['path_length']}\n")
        f.write(f"Reached Goal: {'Yes' if metrics_data['reached_goal'] else 'No'}\n")

        if metrics_data['scenario_type'] == 'adversarial':
            f.write(f"Agent A Expansions: {metrics_data.get('agent_a_expansions', 'N/A')}\n")
            f.write(f"Agent B Expansions: {metrics_data.get('agent_b_expansions', 'N/A')}\n")
            f.write(f"Agent A Path Length: {metrics_data.get('agent_a_path_length', 'N/A')}\n")
            f.write(f"Agent B Path Length: {metrics_data.get('agent_b_path_length', 'N/A')}\n")

        f.write("-" * 30 + "\n\n")


# -----------------------------
# SCENARIO RUNNERS
# -----------------------------
def run_single_agent_3_mazes():
    start = time.time()

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    # Maximize
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except AttributeError:
        try:
            manager.frame.Maximize(True)
        except AttributeError:
            try:
                manager.window.state('zoomed')
            except AttributeError:
                print("Maximize not supported")

    fig.subplots_adjust(left=0.025, right=0.78, top=0.92, bottom=0.2, wspace=0.3)

    scenarios = []
    for i in range(3):
        maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)
        scenarios.append(SingleAgentScenario(axes[i], maze))
        axes[i].set_title(f"Single-Agent Maze #{i+1}")

    def update_all(frame):
        for sc in scenarios:
            sc.update()

    ani = FuncAnimation(fig, update_all, frames=400, interval=300, repeat=False)
    plt.suptitle("Single-Agent A* on 3 Mazes (Step-by-Step)", x=0.403)
    plt.show()

    # Record metrics for each maze
    for i, sc in enumerate(scenarios, 1):
        start = time.time()
        while not sc.done:  # Simulate until done
            sc.update()
        done_time = time.time() - start

        metrics = {
            'scenario_type': 'single',
            'maze_number': i,
            'time_taken': done_time,  # Exact time for each maze
            'total_expansions': sc.expansions[0],
            'path_length': sc.path_len,
            'reached_goal': sc.done and sc.current == sc.goal
        }
        write_metrics_to_file(metrics)


def run_adversarial_3_mazes():
    start = time.time()

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    # Maximize
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except AttributeError:
        try:
            manager.frame.Maximize(True)
        except AttributeError:
            try:
                manager.window.state('zoomed')
            except AttributeError:
                print("Maximize not supported")

    fig.subplots_adjust(left=0.025, right=0.78, top=0.92, bottom=0.2, wspace=0.3)

    scenarios = []
    for i in range(3):
        maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)
        scenarios.append(AdversarialScenario(axes[i], maze))
        axes[i].set_title(f"Adversarial Maze #{i+1}")

    def update_all(frame):
        for sc in scenarios:
            sc.update()

    ani = FuncAnimation(fig, update_all, frames=400, interval=300, repeat=False)
    plt.suptitle("Adversarial Agents on 3 Mazes (Turn-by-Turn)", x=0.403)
    plt.show()

    # Record metrics for each maze
    for i, sc in enumerate(scenarios, 1):
        start = time.time()
        while not sc.done:  # Simulate until done
            sc.update()
        done_time = time.time() - start

        # Metrics for Agent A
        metrics_a = {
            'scenario_type': 'adversarial',
            'maze_number': i,
            'time_taken': done_time,  # Exact time for this maze
            'total_expansions': sc.expansionsA[0],
            'path_length': sc.path_lenA,
            'reached_goal': sc.done and sc.posA == sc.goal
        }
        write_metrics_to_file(metrics_a)

        # Metrics for Agent B
        metrics_b = {
            'scenario_type': 'adversarial',
            'maze_number': i,
            'time_taken': done_time,  # Same time, different expansions
            'total_expansions': sc.expansionsB[0],
            'path_length': sc.path_lenB,
            'reached_goal': sc.done and sc.posB == sc.goal
        }
        write_metrics_to_file(metrics_b)


def clear_metrics_file(filename="metrics.txt"):
    open(filename, 'w').close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Clear previous metrics
    clear_metrics_file()

    # 1) Single-Agent
    run_single_agent_3_mazes()

    # 2) Adversarial
    run_adversarial_3_mazes()