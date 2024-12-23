import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

# Function to parse metrics including steps for each agent
def parse_metrics(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Patterns
    step_pattern = r"Agent ([\w\s-]+) - Step (\d+): Expanded (\d+) nodes at \((\d+), (\d+)\) in ([0-9.]+) seconds\."
    summary_pattern = r"Scenario: (\w+)\nMaze Number: (\d+)\nAgent: ([\w\s-]+)\nTime Taken: ([0-9.]+) seconds\nTotal Expansions: (\d+)\nPath Length: (\d+)\nReached Goal: (Yes|No)"
    
    # Data containers
    summary_data = []
    step_data = []

    # Split scenarios
    scenarios = data.split('------------------------------')
    for scenario in scenarios:
        # Initialize defaults to handle cases without summaries
        maze_number = -1  # Default invalid maze number
        scenario_type = ""
        agent = "Unknown"

        # Parse summary
        summary = re.search(summary_pattern, scenario)
        if summary:
            scenario_type, maze_number, agent, time_taken, total_expansions, path_length, reached_goal = summary.groups()
            summary_data.append({
                'Scenario': scenario_type,
                'Maze': int(maze_number),
                'Agent': agent.strip(),
                'Time': float(time_taken),
                'Expansions': int(total_expansions),
                'Path Length': int(path_length),
                'Goal Reached': reached_goal
            })
            maze_number = int(maze_number)  # Update maze number from summary

        # Parse steps for each scenario
        steps = re.findall(step_pattern, scenario)
        for step in steps:
            step_data.append({
                'Maze': int(maze_number),  # Uses updated maze_number or -1 if missing
                'Agent': step[0].strip(),
                'Step': int(step[1]),
                'Nodes Expanded': int(step[2]),
                'Time Elapsed': float(step[5])
            })

    return pd.DataFrame(summary_data), pd.DataFrame(step_data)

# Function to set a specific seaborn palette and cycle through it
def set_seaborn_palette():
    # Use seaborn's Set2 or tab10 palette for distinct colors
    return sns.color_palette("Set2", n_colors=10)  # You can change to 'tab10', 'Paired', etc.

# Function to add data points to graphs
def add_data_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=10)

# Single-Agent Analysis with Aesthetic Colors
def analyze_single_agent(df):
    # Filter single-agent scenarios (those labeled as 'single' in the 'Scenario' column)
    single_df = df[df['Scenario'] == 'single']

    # If no single-agent data exists, skip the analysis
    if single_df.empty:
        print("No single-agent data found. Skipping single-agent analysis.")
        return

    # Get a 10-color seaborn palette
    colors = set_seaborn_palette()

    # Create a cycle iterator for the colors so that they will be used repeatedly and randomly
    color_cycle = cycle(colors)

    # Plotting the data for each individual single-agent scenario
    ax = single_df.plot(x='Maze', y='Time', kind='bar', color=[next(color_cycle) for _ in range(len(single_df))], legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Time Taken Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Time (seconds)')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()

    # Nodes Expanded with Seaborn Palette
    ax = single_df.plot(x='Maze', y='Expansions', kind='bar', color=[next(color_cycle) for _ in range(len(single_df))], legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Nodes Expanded Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Nodes Expanded')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()

    # Path Length with Seaborn Palette
    ax = single_df.plot(x='Maze', y='Path Length', kind='bar', color=[next(color_cycle) for _ in range(len(single_df))], legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Path Length Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Path Length (steps)')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()

# Adversarial Analysis with Aesthetic Colors
def analyze_adversarial(df):
    adversarial_df = df[df['Scenario'] == 'adversarial']

    # Get unique maze numbers
    maze_numbers = adversarial_df['Maze'].unique()

    # Prepare data to be plotted for all mazes on one graph
    maze_expansions = []
    maze_path_lengths = []
    maze_labels = []

    for maze_number in maze_numbers:
        maze_data = adversarial_df[adversarial_df['Maze'] == maze_number]
        
        # Get nodes expanded and path length for both agents
        agent_A_data = maze_data[maze_data['Agent'] == 'A']
        agent_B_data = maze_data[maze_data['Agent'] == 'B']

        # Append data to the lists
        maze_expansions.append([agent_A_data['Expansions'].values[0], agent_B_data['Expansions'].values[0]])
        maze_path_lengths.append([agent_A_data['Path Length'].values[0], agent_B_data['Path Length'].values[0]])
        maze_labels.append(f'Maze {maze_number}')

    # Get a 10-color seaborn palette
    colors = set_seaborn_palette()

    # Create a cycle iterator for the colors so that they will be used repeatedly and randomly
    color_cycle = cycle(colors)

    # Grouped Bar Chart for Nodes Expanded with Seaborn Colors
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # width of the bars

    x = range(len(maze_labels))

    ax.bar(x, [x[0] for x in maze_expansions], width, label='Agent A (Nodes Expanded)', color=next(color_cycle))
    ax.bar([p + width for p in x], [x[1] for x in maze_expansions], width, label='Agent B (Nodes Expanded)', color=next(color_cycle))

    plt.xticks([p + width / 2 for p in x], maze_labels)
    plt.title('Adversarial: Nodes Expanded Comparison Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Nodes Expanded')
    add_data_labels(ax)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Grouped Bar Chart for Path Lengths with Seaborn Colors
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, [x[0] for x in maze_path_lengths], width, label='Agent A (Path Length)', color=next(color_cycle))
    ax.bar([p + width for p in x], [x[1] for x in maze_path_lengths], width, label='Agent B (Path Length)', color=next(color_cycle))

    plt.xticks([p + width / 2 for p in x], maze_labels)
    plt.title('Adversarial: Path Length Comparison Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Path Length (steps)')
    add_data_labels(ax)
    plt.legend()
    plt.grid(True)
    plt.show()

# Cumulative Nodes Expanded vs Time for Single-Agent Scenarios (Group on One Graph)
def plot_single_agents_over_time(step_df):
    # Filter data for single-agent scenarios (those labeled as 'single' in the 'Scenario' column)
    single_df = step_df[step_df['Agent'] == 'Single']

    # Set the color palette for plotting
    colors = set_seaborn_palette()

    # Create a cycle iterator for the colors so that they will be used repeatedly and randomly
    color_cycle = cycle(colors)

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # For each single-agent maze, plot the cumulative nodes over time
    for maze_number in range(1, 4):
        maze_data = single_df[single_df['Maze'] == maze_number]

        # Cumulative sum of nodes expanded for the current agent
        maze_data['Cumulative Nodes'] = maze_data['Nodes Expanded'].cumsum()

        # Plot the cumulative nodes vs. time for the current agent with a seaborn color
        plt.plot(maze_data['Time Elapsed'], maze_data['Cumulative Nodes'], label=f'Maze {maze_number} (Agent Single)', color=next(color_cycle))

    # Customize the plot
    plt.title('Cumulative Nodes Expanded Over Time for Single-Agent Scenarios')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Nodes Expanded')
    plt.legend(title='Maze Number')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Cumulative Nodes Expanded vs Time for Adversarial Scenarios (Separate for Each Maze)
def plot_adversarial_agents_over_time(step_df):
    # Define the different mazes (1, 2, and 3)
    maze_numbers = [1, 2, 3]
    
    # Set the color palette for plotting
    colors = set_seaborn_palette()

    # Create a cycle iterator for the colors so that they will be used repeatedly
    color_cycle = cycle(colors)

    for maze_number in maze_numbers:
        # Filter data for the current maze and adversarial agents (A and B)
        maze_data = step_df[step_df['Maze'] == maze_number]
        agents = ['A', 'B']

        # Initialize the plot for the current maze
        plt.figure(figsize=(12, 8))

        # For each adversarial agent (A and B), plot the cumulative nodes over time
        for agent in agents:
            agent_data = maze_data[maze_data['Agent'] == agent]

            # Cumulative sum of nodes expanded for the current agent
            agent_data['Cumulative Nodes'] = agent_data['Nodes Expanded'].cumsum()

            # Plot the cumulative nodes vs. time for the current agent with a seaborn color
            plt.plot(agent_data['Time Elapsed'], agent_data['Cumulative Nodes'], label=f'Agent {agent}', color=next(color_cycle))

        # Customize the plot for the current maze
        plt.title(f'Cumulative Nodes Expanded Over Time for Adversarial Agents in Maze {maze_number}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative Nodes Expanded')
        plt.legend(title='Agent')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Main Execution
if __name__ == '__main__':
    # Path to metrics file
    file_path = 'metrics.txt'

    # Parse data
    summary_df, step_df = parse_metrics(file_path)

    # Analyze Single-Agent Mazes
    analyze_single_agent(summary_df)

    # Analyze Adversarial Mazes (Grouped Bar for All Mazes)
    analyze_adversarial(summary_df)
    plot_single_agents_over_time(step_df)

    # Analyze Steps for All Agents (Maze by Maze)
    plot_adversarial_agents_over_time(step_df)
