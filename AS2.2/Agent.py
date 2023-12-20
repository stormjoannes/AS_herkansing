"""In this file we will iterate over values"""


class Agent:

    def __init__(self, position, Maze, Policy, delta_threshold):
        self.position = position
        self.maze = Maze
        self.policy = Policy
        self.delta_threshold = delta_threshold
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, 1], 3: [0, -1]}

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
            delta = -1
            for position in self.maze.grid:
                # Check if position isn't an terminal state.
                if position not in self.maze.terminal_states:
                    # Gets the coords of the states surrounding the current position.
                    new_value = self.policy.select_action(position, iteration)

                    self.maze.grid[position].append(new_value)
                    new_delta = abs(self.maze.grid[position][iteration] - new_value)
                    delta = new_delta if new_delta > delta else delta
                else:
                    # If state is terminal state, new value is always zero.
                    self.maze.grid[position].append(0)

            print(f"End delta: {delta}")
            self.print_iteration(iteration)
            iteration += 1

    def temporal_difference(self, discount: float, learning_rate: float):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
        """
        episode = [0, 2, 0, 0, 3, 3]
        for step in episode:
            next_state = self.maze.stepper(self.position, step)
            print(next_state)

            if next_state not in self.maze.terminal_states:
                value = self.maze.grid[tuple(self.position)][-1]
                reward, next_value = self.maze.rewards[next_state], self.maze.grid[next_state][-1]
                new_value = value + learning_rate * (reward + (discount * next_value) - value)
                self.maze.grid[tuple(self.position)].append(new_value)
                self.position = list(next_state)
                print(self.position)

            else:
                print("Terminal state: ", next_state)

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
