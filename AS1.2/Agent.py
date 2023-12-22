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

            print(f"End delta: {delta}")
            self.print_iteration(iteration)
            iteration += 1

    def action(self):
        """
        Move agent through the maze
        """
        print("Current position: ", self.position)
        surr_states = self.maze.surrounding_states(self.position)
        action = self.policy.choose_action(self.position, surr_states)

        # als de waarde van de beste surrounding state niet beter is dan huidige waarde, terminal
        if not action:
            print("Terminal stage reached")
        else:
            # Action ontbreekt nog
            new_pos = self.maze.stepper(self.position, action[0])
            self.position = new_pos
            print("new position", self.position)

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
