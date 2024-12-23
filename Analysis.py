import pandas as pd
import matplotlib.pyplot as plt

# Load metrics data
filename = "metrics.txt"

def load_metrics(filename):
    metrics = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            # Skip the header or divider lines
            if line.startswith("----") or line == "":
                continue
            if line.startswith("Scenario"):
                # Read each metric block
                scenario = line.split(": ")[1]
                maze_number = int(lines[i+1].split(": ")[1])
                time_taken = float(lines[i+2].split(": ")[1].split()[0])
                total_expansions = int(lines[i+3].split(": ")[1])
                path_length = int(lines[i+4].split(": ")[1])
                reached_goal = 1 if lines[i+5].split(": ")[1] == "Yes" else 0

                if scenario == 'adversarial':
                    agent = lines[i+6].split(": ")[1]  # Agent A or B
                    agent_expansions = int(lines[i+7].split(": ")[1])
                    agent_path_length = int(lines[i+8].split(": ")[1])

                    metrics.append({
                        'Scenario': scenario,
                        'MazeNumber': maze_number,
                        'Agent': agent,
                        'TimeTaken': time_taken,
                        'TotalExpansions': agent_expansions,
                        'PathLength': agent_path_length,
                        'ReachedGoal': reached_goal
                    })
                else:
                    metrics.append({
                        'Scenario': scenario,
                        'MazeNumber': maze_number,
                        'TimeTaken': time_taken,
                        'TotalExpansions': total_expansions,
                        'PathLength': path_length,
                        'ReachedGoal': reached_goal
                    })
    return pd.DataFrame(metrics)

# Read data
data = load_metrics(filename)

# ---------------------
# VISUALIZATIONS
# ---------------------

# 1. Time Taken per Maze - Bar Chart
plt.figure(figsize=(10, 6))
for scenario in data['Scenario'].unique():
    subset = data[data['Scenario'] == scenario]
    plt.bar(subset['MazeNumber'], subset['TimeTaken'], label=scenario)
plt.xlabel('Maze Number')
plt.ylabel('Time Taken (seconds)')
plt.title('Time Taken per Maze')
plt.legend()
plt.grid(True)
plt.show()

# 2. Total Expansions per Maze - Bar Chart
plt.figure(figsize=(10, 6))
for scenario in data['Scenario'].unique():
    subset = data[data['Scenario'] == scenario]
    plt.bar(subset['MazeNumber'], subset['TotalExpansions'], label=scenario)
plt.xlabel('Maze Number')
plt.ylabel('Total Expansions')
plt.title('Total Expansions per Maze')
plt.legend()
plt.grid(True)
plt.show()

# 3. Path Length per Maze - Bar Chart
plt.figure(figsize=(10, 6))
for scenario in data['Scenario'].unique():
    subset = data[data['Scenario'] == scenario]
    plt.bar(subset['MazeNumber'], subset['PathLength'], label=scenario)
plt.xlabel('Maze Number')
plt.ylabel('Path Length')
plt.title('Path Length per Maze')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------
# ANALYSIS
# ---------------------

# 1. Average Time Taken
avg_time = data.groupby(['Scenario'])['TimeTaken'].mean()
print("Average Time Taken:")
print(avg_time)

# 2. Efficiency (Expansions per Second)
data['Efficiency'] = data['TotalExpansions'] / data['TimeTaken']
plt.figure(figsize=(10, 6))
for scenario in data['Scenario'].unique():
    subset = data[data['Scenario'] == scenario]
    plt.bar(subset['MazeNumber'], subset['Efficiency'], label=scenario)
plt.xlabel('Maze Number')
plt.ylabel('Efficiency (Nodes Expanded per Second)')
plt.title('Efficiency per Maze')
plt.legend()
plt.grid(True)
plt.show()
print("\nEfficiency (Expansions per Second):")
print(data[['Scenario', 'MazeNumber', 'Efficiency']])

# 3. Success Rate
success_rate = data.groupby(['Scenario'])['ReachedGoal'].mean() * 100
print("\nSuccess Rate (%):")
print(success_rate)
