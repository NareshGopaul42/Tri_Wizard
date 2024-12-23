import re
import matplotlib.pyplot as plt
import pandas as pd

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

# Function to add data points to graphs
def add_data_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
        
# Single-Agent Analysis
def analyze_single_agent(df):
    # Filter single-agent scenarios (those labeled as 'single' in the 'Scenario' column)
    single_df = df[df['Scenario'] == 'single']

    # If no single-agent data exists, skip the analysis
    if single_df.empty:
        print("No single-agent data found. Skipping single-agent analysis.")
        return

    # Plotting the data for each individual single-agent scenario
    # Time Taken
    ax = single_df.plot(x='Maze', y='Time', kind='bar', color='blue', legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Time Taken Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Time (seconds)')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()

    # Nodes Expanded
    ax = single_df.plot(x='Maze', y='Expansions', kind='bar', color='green', legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Nodes Expanded Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Nodes Expanded')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()

    # Path Length
    ax = single_df.plot(x='Maze', y='Path Length', kind='bar', color='purple', legend=False, figsize=(10, 6))
    plt.title('Single-Agent: Path Length Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Path Length (steps)')
    add_data_labels(ax)
    plt.grid(True)
    plt.show()



# Adversarial Analysis (Grouped Bar for All Agents in Each Maze)
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

    # Create a grouped bar chart for Nodes Expanded
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # width of the bars

    # Set the positions for the bars (side by side)
    x = range(len(maze_labels))

    ax.bar(x, [x[0] for x in maze_expansions], width, label='Agent A (Nodes Expanded)', color='blue')
    ax.bar([p + width for p in x], [x[1] for x in maze_expansions], width, label='Agent B (Nodes Expanded)', color='green')

    plt.xticks([p + width / 2 for p in x], maze_labels)
    plt.title('Adversarial: Nodes Expanded Comparison Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Nodes Expanded')
    add_data_labels(ax)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a grouped bar chart for Path Lengths
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, [x[0] for x in maze_path_lengths], width, label='Agent A (Path Length)', color='blue')
    ax.bar([p + width for p in x], [x[1] for x in maze_path_lengths], width, label='Agent B (Path Length)', color='green')

    plt.xticks([p + width / 2 for p in x], maze_labels)
    plt.title('Adversarial: Path Length Comparison Across Mazes')
    plt.xlabel('Maze Number')
    plt.ylabel('Path Length (steps)')
    add_data_labels(ax)
    plt.legend()
    plt.grid(True)
    plt.show()



# Step-by-Step Expansion Analysis for All Agents (Separate by Maze with Data Labels)
def analyze_steps_all_agents(summary_df, step_df):
    adversarial_df = summary_df[summary_df['Scenario'] == 'adversarial']

    # Get unique maze numbers
    maze_numbers = adversarial_df['Maze'].unique()

    # Loop over each maze and plot for each agent's step-by-step expansion
    for maze_number in maze_numbers:
        maze_data = adversarial_df[adversarial_df['Maze'] == maze_number]
        
        plt.figure(figsize=(10, 6))
        for _, agent in maze_data.iterrows():
            # Get step data for each agent in this maze
            agent_steps = step_df[(step_df['Maze'] == agent['Maze']) & (step_df['Agent'].str.contains(agent['Agent']))]
            line, = plt.plot(agent_steps['Step'], agent_steps['Nodes Expanded'], marker='o', label=f'{agent["Agent"]}')
            
            # Add data labels to the line graph
            for i, txt in enumerate(agent_steps['Nodes Expanded']):
                plt.text(agent_steps['Step'].iloc[i], txt, str(txt), fontsize=9, ha='right', color=line.get_color())

        plt.title(f'Adversarial: Step-by-Step Expansions in Maze {maze_number}')
        plt.xlabel('Step')
        plt.ylabel('Nodes Expanded')
        plt.legend()
        plt.grid(True)
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

    # Analyze Steps for All Agents (Maze by Maze)
    analyze_steps_all_agents(summary_df, step_df)
