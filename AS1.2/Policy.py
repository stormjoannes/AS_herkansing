"""In this file we define our policy class"""


class Policy:

    def __init__(self, maze):
        """
        Defines the values of class Policy.
        """
        self.maze = maze
        self.discount = 1

    def select_action(self, position, iteration):
        """
        Select max value state, from states touching the given state.
        """

        opt_1 = self.maze.stepper(position, 0)  # up
        opt_2 = self.maze.stepper(position, 1)  # down
        opt_3 = self.maze.stepper(position, 2)  # right
        opt_4 = self.maze.stepper(position, 3)  # left
        options = [opt_1, opt_2, opt_3, opt_4]

        new_value = self.monte_carlo(options, iteration)

        return new_value

    def monte_carlo(self, options, iteration):
        new_value = max(self.maze.rewards[options[0]] + (self.discount * self.maze.grid[options[0]][iteration]),
                        self.maze.rewards[options[1]] + (self.discount * self.maze.grid[options[1]][iteration]),
                        self.maze.rewards[options[2]] + (self.discount * self.maze.grid[options[2]][iteration]),
                        self.maze.rewards[options[3]] + (self.discount * self.maze.grid[options[3]][iteration]))
        return new_value

    def choose_action(self, position, surr_states):
        #represent decide_action
        highest_val = 0
        for state in surr_states:
            if self.maze.grid[state][-1] > highest_val:
                highest_val = self.maze.grid[state][-1]
                highest_state = state

        action = best_state - position[i]
        return action

