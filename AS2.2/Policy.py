"""In this file we define our policy class"""
import random


class Policy:

    def __init__(self, maze):
        """
        Defines the values of class Policy.

            Parameters:
                maze(class): current maze class
        """
        self.maze = maze
        self.discount = 1

    def select_action(self, position, iteration):
        """
        Select max value state, from states touching the given state.

            Parameters:
                position(tuple): Combination of x and y-axis position
                iteration(int): Current iteration

            Return:
                new_values(float): New value of the position
        """

        opt_1 = self.maze.stepper(position, 0)  # up
        opt_2 = self.maze.stepper(position, 1)  # down
        opt_3 = self.maze.stepper(position, 2)  # right
        opt_4 = self.maze.stepper(position, 3)  # left
        options = [opt_1, opt_2, opt_3, opt_4]

        new_value = self.monte_carlo(options, iteration)

        return new_value

    def monte_carlo(self, options, iteration):
        """
        Calculate the values of the surrounding states, and select the highest.

            Parameters:
                options(list): Position of the surrounding states
                iteration(int): Iteration to get the value of current iteration

            Return:
                new_value(float): New value of the position
        """
        new_value = max(self.maze.rewards[options[0]] + (self.discount * self.maze.grid[options[0]][iteration]),
                        self.maze.rewards[options[1]] + (self.discount * self.maze.grid[options[1]][iteration]),
                        self.maze.rewards[options[2]] + (self.discount * self.maze.grid[options[2]][iteration]),
                        self.maze.rewards[options[3]] + (self.discount * self.maze.grid[options[3]][iteration]))
        return new_value

    def choose_action(self, position, surr_states) -> int:
        """
        Check which action is needed to get to the best surrounding state.

            Parameters:
                position(tuple): Current position
                surr_states(list): Values from surrounding states

            Return:
                action(int): Action to take
        """
        highest_state = max(surr_states, key=surr_states.get)

        pos_action = [highest_state[0] - position[0], highest_state[1] - position[1]]
        action = [key for key, action_position in self.maze.actions.items() if action_position == pos_action]
        return action

    # -------------------- CHANGE ----------------------
    def decide_action_value(self, state, discount, epsilon, surr_values):
        """
        Decide action, depending on the epsilon it can differ if it is random or not.

            Parameters:
                state(): Current state
                discount(float): Current discount
                epsilon(float): Current epsilon
                surr_values(list): Surrounding values of state

            Return:
                action(int): Chosen action to take
        """
        # x% chance to land in epsilon aka random
        rd_num = round(random.random(), 2)
        if rd_num < epsilon:
            choice = random.choice([0, 1, 2, 3])
            return choice

        else:
            # get correct action by finding the highest Q value and return correct action
            # if more than one action have the same max value, pick first
            max_action = max(surr_values)
            greedy_action = surr_values.index(max_action)

            return greedy_action

    def value_func(self, next_states, discount):
        """
        Calculate the new values for the given next states.

            Parameters:
                next_states(list):
                discount(float): Current discount

            Return:
                next_values(list): All values of the given next states
        """
        next_values = []
        for coord in next_states:
            val = self.maze.rewards[coord] + discount * self.maze.grid[coord][-1]
            next_values.append(val)

        return next_values
