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
            c_state = self.position
            print('\n')
            print(c_state)
            action = self.policy.decide_action_value(c_state, discount, epsilon, self.policy.value_func(self.maze.surrounding_states(c_state), discount))
            while c_state not in self.maze.terminal_states:
                print(c_state)
                next_position = self.maze.stepper(self.position, action)

                c_surr_states = self.maze.surrounding_states(c_state)
                c_surr_values = self.policy.value_func(c_surr_states, discount)

                next_surr_states = self.maze.surrounding_states(next_position)
                next_surr_values = self.policy.value_func(next_surr_states, discount)

                next_action = self.policy.decide_action_value(next_position, discount, epsilon, next_surr_values)

                # self.pos[3][action] = c_surr_values[action] + learning_rate * (self.maze.rewards[next_position] + discount * next_surr_values[next_action] - c_surr_values[action])

                action = next_action
                self.position = next_position
                c_state = next_position
            self.position = (3, 2)

    def q_learning(self, discount: float, learning_rate: float, epochs: int):
        """
        ...

            Parameters:
                discount(float): ...
                learning_rate(float): ...
                epochs(int): ...
        """
        for epoch in range(epochs):
            value = self.maze.grid[self.position][-1]
            pass

    def choose_action(self):
        """
        Choose where the next step will be.
        """
        return self.policy.select_action()

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
