"""In this file we will iterate over values"""


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
            print(f"Begin delta: {delta}")
            # Set delta low so that first loop delta will always be overwritten.
            delta = 0
            for state in self.maze.grid:
                # Check if state isn't an terminal state.
                if state not in self.maze.terminal_states:
                    # Gets the coords of the states surrounding the current state.
                    new_value = self.policy.select_action(state, iteration)

                    self.maze.grid[state].append(new_value)
                    new_delta = abs(self.maze.grid[state][iteration] - new_value)
                    delta = new_delta if new_delta > delta else delta
                else:
                    # If state is terminal state, new value is always zero.
                    self.maze.grid[state].append(0)

            print(f"End delta: {delta}")
            self.print_iteration(iteration)
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
                    value = self.maze.grid[tuple(self.position)][-1]
                    reward, next_value = self.maze.rewards[next_state], self.maze.grid[next_state][-1]
                    new_value = value + learning_rate * (reward + (discount * next_value) - value)
                    self.maze.grid[tuple(self.position)].append(new_value)
                    self.position = list(next_state)

                else:
                    print("Terminal state: ", next_state)

    def sarsa(self, discount: float, learning_rate: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epochs(int): ...
        """
        for epoch in range(epochs):
            value = self.maze.grid[tuple(self.position)][-1]
            pass

    def q_learning(self, discount: float, learning_rate: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epochs(int): ...
        """
        for epoch in range(epochs):
            value = self.maze.grid[tuple(self.position)][-1]
            pass

    def choose_action(self):
        """
        Choose where the next step will be.
        """
        return self.policy.select_action()

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
