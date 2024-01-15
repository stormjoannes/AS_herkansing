"""In this file we will iterate over values using Temporal difference learning, SARSA and Q-learning (SARSAMAX)"""

import matplotlib.pyplot as plt
import math
import numpy as np


class Agent:

    def __init__(self, position: tuple, Maze: classmethod, Policy: classmethod, delta_threshold: float):
        """Set all class values"""
        self.position = position
        self.maze = Maze
        self.policy = Policy
        self.delta_threshold = delta_threshold

    def value_iteration(self):
        """
        Iterating over the maze grid and calculate each value for each iteration.
        """
        # Make sure the first time iteration it will always go into the while loop.
        delta = self.delta_threshold + 1
        iteration = 0
        while delta >= self.delta_threshold:
            # print(f"Begin delta: {delta}")
            # Set delta low so that first loop delta will always be overwritten.
            delta = 0
            for state in self.maze.grid:
                # Check if state isn't a terminal state.
                if state not in self.maze.terminal_states:
                    # Gets the coords of the states surrounding the current state.
                    new_value = self.policy.select_action(state, iteration)

                    self.maze.grid[state].append(new_value)

                    # Get difference from old to new value
                    new_delta = abs(self.maze.grid[state][iteration] - new_value)
                    delta = new_delta if new_delta > delta else delta
                else:
                    # If state is terminal state, new value is always zero.
                    self.maze.grid[state].append(0)

            iteration += 1

    def temporal_difference(self, discount: float, learning_rate: float, epochs: int):
        """
        Functioneren van een agent op basis van de temporal difference learning methode.

            Parameters:
                discount(float): Amount of discount for the agent, impacts the importance of values/rewards
                learning_rate(float): The rate of how fast the agent learns
                epochs(int): Amount of times the loop should loop
        """
        for epoch in range(epochs):
            # Optimal steps
            episode = [0, 2, 0, 0, 3, 3]
            for step in episode:
                next_state = self.maze.stepper(self.position, step)

                if next_state not in self.maze.terminal_states:
                    value = self.maze.grid[self.position][-1]
                    reward, next_value = self.maze.rewards[next_state], self.maze.grid[next_state][-1]
                    new_value = value + learning_rate * (reward + (discount * next_value) - value)
                    self.maze.grid[self.position].append(new_value)
                    self.position = next_state

                else:
                    print("Terminal state: ", next_state)
                    self.position = (3, 2)
                    self.print_iteration(-1)

                    # Add last iteration value of episodes to dict episodes (rounded with 2 decimals)
                    for cor in self.maze.grid:
                        self.maze.episodes[cor].append(round(self.maze.grid[cor][-1], 2))

        self.plot_values(2, 5, f"_temporal_difference_{discount}")

    def sarsa(self, discount: float, learning_rate: float, epsilon: float, epochs: int):
        """
        Functioneren van een agent op basis van de SARSA methode.

            Parameters:
                discount(float): Amount of discount for the agent, impacts the importance of values/rewards
                learning_rate(float): The rate of how fast the agent learns
                epsilon(float): Current epsilon
                epochs(int): Amount of times the loop should loop
        """
        for epoch in range(epochs):
            print("epoch: ", epoch)
            state = self.position
            print('\n')
            print(state)
            action = self.policy.decide_action_value(discount, epsilon, self.policy.value_func(self.maze.surrounding_states(state), discount))

            print("Hieroooo", self.maze.surrounding_values_per_coords)

            while state not in self.maze.terminal_states:
                print(state)
                next_position = self.maze.stepper(self.position, action)

                next_surr_states = self.maze.surrounding_states(next_position)
                next_surr_values = self.policy.value_func(next_surr_states, discount)

                next_action = self.policy.decide_action_value(discount, epsilon, next_surr_values)
                print(next_action)

                # Update maze surrounding values ------------- IMPORTANT ----------
                print('gabber', action, self.maze.surrounding_values_per_coords)
                print('between', next_action)
                print('second', next_surr_values)
                # next_surr_values can consist of only 3 values if corner, next action 3 will crash it
                # print('dropper', next_surr_values[next_action])
                self.maze.surrounding_values_per_coords[state][action][0] = self.maze.surrounding_values_per_coords[state][action][0] + learning_rate * (self.maze.rewards[next_position] + discount * self.maze.surrounding_values_per_coords[next_position][next_action][0] - self.maze.surrounding_values_per_coords[state][action][0])

                action = next_action
                self.position = next_position
                state = next_position
            self.position = (3, 2)

        self.plot_sarsa_values(f"SARSA_{discount}")
        self.plot_sarsa_directions(f"SARSA_{discount}")

    def q_learning(self, discount: float, learning_rate: float, epsilon: float, epochs: int):
        """
        Functioneren van een agent op basis van de q-learning (SARSAMAX) methode.

            Parameters:
                discount(float): Amount of discount for the agent, impacts the importance of values/rewards
                learning_rate(float): The rate of how fast the agent learns
                epsilon(float): Current epsilon
                epochs(int): Amount of times the loop should loop
        """
        for epoch in range(epochs):
            state = self.position
            while state not in self.maze.terminal_states:
                action = self.policy.decide_action_value(discount, epsilon, self.policy.value_func(self.maze.surrounding_states(state), discount))
                next_position = self.maze.stepper(self.position, action)

                next_surr_states = self.maze.surrounding_states(next_position)
                next_surr_values = self.policy.value_func(next_surr_states, discount)

                best_action_value = max(next_surr_values)

                self.maze.surrounding_values_per_coords[state][action][0] = self.maze.surrounding_values_per_coords[state][action][0] + learning_rate * (self.maze.rewards[next_position] + discount * best_action_value - self.maze.surrounding_values_per_coords[state][action][0])

                self.position = next_position
                state = next_position

            self.position = (3, 2)

        self.plot_sarsa_values(f"Q-learning_{discount}")
        self.plot_sarsa_directions(f"Q-learning_{discount}")

    def print_iteration(self, iteration: int):
        """
        Quick hardcoded print to show values for each iteration.
        """
        values = []
        for coord in self.maze.grid:
            values.append(self.maze.grid[coord][iteration])

        values = [round(value, 1) for value in values]

        print("Iteration: ", iteration)
        print(values[0: 4])
        print(values[4: 8])
        print(values[8: 12])
        print(values[12: 16], "\n")

    def plot_sarsa_values(self, plt_name: str):
        """Plot the SARSA and SARSAMAX last iteration surround values for each position."""
        print(self.maze.surrounding_values_per_coords)

        values = np.array([np.array(self.maze.surrounding_values_per_coords[key]) for key in self.maze.surrounding_values_per_coords.keys()])

        action_names = ["Up", "Down", "Left", "Right"]

        fig, ax = plt.subplots(figsize=(4, 6))  # Adjust the figsize parameter as needed

        ax.imshow(values[:, :, 0], cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names)
        ax.set_yticks(range(len(self.maze.surrounding_values_per_coords.keys())))
        ax.set_yticklabels([str(key) for key in self.maze.surrounding_values_per_coords.keys()])
        ax.set_xlabel('Next position direction value')
        ax.set_ylabel('Positions')
        ax.set_title(f'Surrounding values after {plt_name}')

        # Show actual values in each block
        for i in range(len(self.maze.surrounding_values_per_coords.keys())):
            for j in range(len(action_names)):
                ax.annotate(str(round(values[i, j, 0], 2)),
                            xy=(j, i),
                            ha='center', va='center',
                            color='w' if values[i, j, 0] < 30 else 'black')  # White text for negative values

        plt.savefig(f'../images/AS_{plt_name}_visualization.png')
        plt.show()

    def plot_sarsa_directions(self, plt_name: str):
        """Plot the SARSA and SARSAMAX directions based on the highest values of the surrounding states."""
        # Extract directions and values
        directions = ["Up", "Down", "Left", "Right"]
        values = np.zeros((4, 4, 4))

        for i in range(4):
            for j in range(4):
                values[i, j, :] = [self.maze.surrounding_values_per_coords[(i, j)][k][0] for k in range(4)]

        # Determine the direction with the highest value for each coordinate
        max_directions = np.argmax(values, axis=2)

        # Plot the heatmap without borders
        fig, ax = plt.subplots()
        ax.imshow(max_directions, cmap='viridis')

        # Hide grid lines
        ax.set_xticks(np.arange(4) - 0.5, minor=True)
        ax.set_yticks(np.arange(4) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=1)

        # Add labels
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        plt.xlabel('X-axis position')
        plt.ylabel('Y-axis position')
        plt.title('Highest Value Direction for Each Coordinate')

        # Display the values on the heatmap
        for i in range(4):
            for j in range(4):
                if (i, j) not in self.maze.terminal_states:
                    ax.text(j, i, directions[max_directions[i, j]], ha='center', va='center', color='w')
                else:
                    ax.text(j, i, 'terminal', ha='center', va='center', color='w')

        plt.savefig(f'../images/AS_{plt_name}_directions_visualization.png')
        plt.show()

    def plot_values(self, tot_fig_rows: int, tot_fig_columns: int, plt_name: str):
        """
        Plot the values in a state transition matrix
        """
        iterations = len(self.maze.episodes[0, 0]) - 1
        rows = math.ceil(math.sqrt(len(self.maze.episodes)))
        cols = math.ceil(math.sqrt(len(self.maze.episodes)))

        fig, axs = plt.subplots(tot_fig_rows, tot_fig_columns, figsize=(10, 9))
        fig.suptitle('Heatmaps for Each Iteration')

        print(self.maze.episodes)

        cl = -1
        nextt = False
        for i in range(iterations + 1):
            values = []
            cl += 1
            for r in range(rows):
                row_values = []
                for c in range(cols):
                    key = (r, c)
                    row_values.append(self.maze.episodes[key][i])
                values.append(row_values)

            rw = 0 if i <= (iterations/2) else 1
            if i > (iterations/2) and not nextt:
                cl = 0
                nextt = True
            # cl =
            print([rw, cl])
            ax = axs[rw, cl]
            # ax = axs[i // tot_fig_rows, i % tot_fig_columns]
            ax.imshow(values, cmap='viridis', interpolation='nearest')

            for r in range(rows):
                for c in range(cols):
                    color = 'black' if values[r][c] > 30 else 'white'
                    ax.text(c, r, values[r][c], ha='center', va='center', color=color)

            ax.set_title(f'Iteration {i}')
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.set_xticklabels([str(j) for j in range(cols)])
            ax.set_yticklabels([str(rows - j - 1) for j in range(rows)][::-1])  # Reversed y-axis tick labels

        plt.tight_layout()
        plt.savefig(f'../images/AS{plt_name}_visualization.png')
        plt.show()
