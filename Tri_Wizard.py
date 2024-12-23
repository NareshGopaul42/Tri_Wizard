import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
import time
import pickle

def save_maze(filename, maze):
    """
    Save the maze to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(maze, f)


def load_maze(filename):
    """
    Load the maze from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


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
    Logs steps and summaries for both single-agent and adversarial scenarios in a grouped format.
    """
    with open(filename, 'a') as f:
        # Write header for the first log
        if f.tell() == 0:
            f.write("---- METRICS LOG ----\n\n")

        # Log steps (grouped per agent)
        if 'step' in metrics_data:
            agent = metrics_data.get('agent', 'Single-Agent')
            f.write(f"Agent {agent} - Step {metrics_data['step']}: Expanded {metrics_data['expansions']} nodes at {metrics_data['node']} "
                    f"in {metrics_data['time_taken']:.3f} seconds.\n")


        # Final summary for each agent
        else:
            f.write(f"Scenario: {metrics_data['scenario_type']}\n")
            f.write(f"Maze Number: {metrics_data['maze_number']}\n")
            f.write(f"Agent: {metrics_data.get('agent', 'N/A')}\n")
            f.write(f"Time Taken: {metrics_data['time_taken']:.3f} seconds\n")
            f.write(f"Total Expansions: {metrics_data['total_expansions']}\n")
            f.write(f"Path Length: {metrics_data['path_length']}\n")
            f.write(f"Reached Goal: {'Yes' if metrics_data['reached_goal'] else 'No'}\n")
            f.write("-" * 30 + "\n\n")


# -----------------------------
# SCENARIO RUNNERS
# -----------------------------
def run_single_agent_3_mazes():
    """
    Run single-agent scenarios for 3 mazes sequentially.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
    plt.suptitle("Single-Agent A* on 3 Mazes", x=0.403)
    print("Running Single-Agent Scenarios...")

    for i in range(3):
        print(f"Running Single-Agent for Maze {i + 1}")
        maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)
        scenario = SingleAgentScenario(axes[i], maze)

        # Track time
        start_time = time.time()
        step_counter = 0

        while not scenario.done:
            scenario.update()
            if scenario.done:
                break
            step_counter += 1
            step_time = time.time() - start_time

            # Log each step
            write_metrics_to_file({
                'step': step_counter,
                'agent': 'Single',  # Specify single-agent
                'expansions': scenario.expansions[0],
                'node': scenario.current,
                'time_taken': step_time
            })


        # Final metrics
        total_time = time.time() - start_time
        reached_goal = scenario.done and scenario.current == scenario.goal

        # Log summary
        write_metrics_to_file({
            'scenario_type': 'single',
            'maze_number': i + 1,
            'agent': 'Single',
            'time_taken': total_time,
            'total_expansions': scenario.expansions[0],
            'path_length': scenario.path_len,
            'reached_goal': reached_goal
        })
    
    plt.show()


def run_adversarial_3_mazes():
    """
    Run adversarial scenarios for 3 mazes sequentially.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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

    print("Running Adversarial Scenarios...")

    for i in range(3):
        print(f"Running Adversarial for Maze {i + 1}")
        maze = generate_maze(GRID_SIZE, OBSTACLE_PROB)
        scenario = AdversarialScenario(axes[i], maze)

        # Separate step counters for each agent
        step_A = 0
        step_B = 0

        # Track time
        start_time = time.time()

        # Separate logs for Agent A and Agent B
        while not scenario.done:
            # Track the agent that just moved
            prev_turn = scenario.turn  # Save BEFORE updating

            # Update the scenario (perform the move for the current turn)
            scenario.update()
            if scenario.done:
                break
            step_time = time.time() - start_time  # AFTER the move, measure time

            # Log the move for the previous turn
            if prev_turn == 'A':  # Agent A just moved
                step_A += 1
                write_metrics_to_file({
                    'step': step_A,
                    'agent': 'A',
                    'expansions': scenario.expansionsA[0],
                    'node': scenario.posA,
                    'time_taken': step_time
                })
            else:  # Agent B just moved
                step_B += 1
                write_metrics_to_file({
                    'step': step_B,
                    'agent': 'B',
                    'expansions': scenario.expansionsB[0],
                    'node': scenario.posB,
                    'time_taken': step_time
                })

        # Final summary for Agent A
        total_time = time.time() - start_time
        write_metrics_to_file({
            'scenario_type': 'adversarial',
            'maze_number': i + 1,
            'agent': 'A',
            'time_taken': total_time,
            'total_expansions': scenario.expansionsA[0],
            'path_length': scenario.path_lenA,
            'reached_goal': scenario.posA == scenario.goal
        })

        # Final summary for Agent B
        write_metrics_to_file({
            'scenario_type': 'adversarial',
            'maze_number': i + 1,
            'agent': 'B',
            'time_taken': total_time,
            'total_expansions': scenario.expansionsB[0],
            'path_length': scenario.path_lenB,
            'reached_goal': scenario.posB == scenario.goal
        })


    plt.suptitle("Adversarial Agents on 3 Mazes", x=0.403)
    plt.show()


def clear_metrics_file(filename="metrics.txt"):
    """
    Clears the metrics file before starting simulations.
    """
    open(filename, 'w').close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Clear previous metrics
    clear_metrics_file()

    # Run Single-Agent Scenarios (All 3 Mazes First)
    run_single_agent_3_mazes()

    # Run Adversarial Scenarios (All 3 Mazes After Single-Agent)
    run_adversarial_3_mazes()
