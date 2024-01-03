"""In this file we will iterate over values"""

import matplotlib.pyplot as plt
import math


class Agent:

    def __init__(self, position, Maze, Policy, delta_threshold):
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

                    # --- TEST ---
                    # print(self.maze.surrounding_states(self.position))

                    self.maze.grid[state].append(new_value)

                    # Get difference from old to new value
                    new_delta = abs(self.maze.grid[state][iteration] - new_value)
                    # print("new delta: ", new_delta, ", current delta: ", delta, ", state: ", state)
                    delta = new_delta if new_delta > delta else delta
                else:
                    # If state is terminal state, new value is always zero.
                    self.maze.grid[state].append(0)

            # print(f"End delta: {delta}")
            # self.print_iteration(iteration)
            iteration += 1

    def temporal_difference(self, discount: float, learning_rate: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epochs(int): ...
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

    def sarsa(self, discount: float, learning_rate: float, epsilon: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epsilon(float): ...
                epochs(int): ...
        """
        for epoch in range(epochs):
            state = self.position
            print('\n')
            print(state)
            action = self.policy.decide_action_value(state, discount, epsilon, self.policy.value_func(self.maze.surrounding_states(state), discount))
            while state not in self.maze.terminal_states:
                print(state)
                next_position = self.maze.stepper(self.position, action)

                c_surr_states = self.maze.surrounding_states(state)
                c_surr_values = self.policy.value_func(c_surr_states, discount)

                next_surr_states = self.maze.surrounding_states(next_position)
                next_surr_values = self.policy.value_func(next_surr_states, discount)

                next_action = self.policy.decide_action_value(next_position, discount, epsilon, next_surr_values)

                # self.pos[3][action] = c_surr_values[action] + learning_rate * (self.maze.rewards[next_position] + discount * next_surr_values[next_action] - c_surr_values[action])

                action = next_action
                self.position = next_position
                state = next_position
            self.position = (3, 2)

        # self.plot_values()

    def q_learning(self, discount: float, learning_rate: float, epsilon: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epochs(int): ...
        """
        for epoch in range(epochs):
            state = self.position
            while state not in self.maze.terminal_states:
                action = self.policy.decide_action_value(state, discount, epsilon, self.policy.value_func(self.maze.surrounding_states(state), discount))
                next_position = self.maze.stepper(self.position, action)

                c_surr_states = self.maze.surrounding_states(state)
                c_surr_values = self.policy.value_func(c_surr_states, discount)

                next_surr_states = self.maze.surrounding_states(next_position)
                next_surr_values = self.policy.value_func(next_surr_states, discount)

                best_action_value = max(next_surr_values)

                # self.pos[3][action] = c_surr_values[action] + learning_rate * (self.maze.rewards[next_position] + discount * next_surr_values[next_action] - c_surr_values[action])

                self.position = next_position

            self.position = (3, 2)

        # self.plot_values()

    def print_iteration(self, iteration):
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

    def plot_values(self, tot_fig_rows, tot_fig_columns, plt_name):
        """
        Plot the values in a state transition matrix
        """
        iterations = len(self.maze.grid[0, 0])
        rows = math.ceil(math.sqrt(len(self.maze.grid)))
        cols = math.ceil(math.sqrt(len(self.maze.grid)))

        fig, axs = plt.subplots(tot_fig_rows, tot_fig_columns, figsize=(10, 9))
        fig.suptitle('Heatmaps for Each Iteration')

        for i in range(iterations):
            values = []
            for r in range(rows):
                row_values = []
                for c in range(cols):
                    key = (r, c)
                    row_values.append(self.maze.grid[key][i])
                values.append(row_values)

            ax = axs[i // tot_fig_rows, i % tot_fig_columns]
            ax.imshow(values, cmap='viridis', interpolation='nearest')

            for r in range(rows):
                for c in range(cols):
                    color = 'black' if values[r][c] > 30 else 'white'
                    ax.text(c, r, values[r][c], ha='center', va='center', color=color)

            ax.set_title(f'Iteration {i + 1}')
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.set_xticklabels([str(j) for j in range(cols)])
            ax.set_yticklabels([str(rows - j - 1) for j in range(rows)][::-1])  # Reversed y-axis tick labels

        plt.tight_layout()
        plt.savefig(f'../images/AS{plt_name}_visualization.png')
        plt.show()
