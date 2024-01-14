"""In this file we define create our maze"""


class Maze:

    def __init__(self):
        """Set all class values"""
        self.rewards = {}
        self.grid = {}
        self.terminal_states = []
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.episodes = self.fillDict([[] for _ in range(16)])
        self.surrounding_values_per_coords = self.fillDict([[] for _ in range(16)])

    def stepper(self, position, action):
        """
        Get the new position based on the action

            Parameters:
                position(tuple): Current x and y-axis position
                aciton(int): The chosen action to take

            Return:
                tuple: Next x and y-axis position
        """
        movement = self.actions[action]
        new_position = (position[0] + movement[0], position[1] + movement[1])

        if new_position in self.grid:
            return new_position
        else:
            return position

    def surrounding_states(self, position):
        """
        Get the surrounding states for current position

            Parameters:
                position(tuple): Current x and y-axis position

            Return:
                states(list): list with for each state the surrounding rewards and values
        """
        states = {}
        for action in self.actions.keys():
            state = self.stepper(position, action)
            states[state] = [self.rewards[state], self.grid[state][-1]]
        return states

    def surrounding_values(self):
        """
        Fill the surround values dictionary with the last iteration value in the grid
        """
        for position in self.surrounding_values_per_coords:
            for action in self.actions.keys():
                state = self.stepper(position, action)
                self.surrounding_values_per_coords[state].append([self.grid[state][-1]])

    def fillDict(self, value, sizeHorizontal=4, sizeVertical=4):
        """
        Filling a dictionary with coördinates.

            Parameters:
                value(float):
                sizeHorizontal(int): Horizontal size of the maze
                sizeVertical(int): Vertical size of the maze

            Return:
                dict(dict): Dictionary with each position defined
        """
        index = 0
        dict = {}
        for vertical in range(sizeVertical):
            for horizontal in range(sizeHorizontal):
                coord = (vertical, horizontal)
                dict[coord] = value[index]
                index += 1

        return dict

    def create_maze_values(self):
        """
        Create the values and rewards in a dictionary.
        """
        values = [[0] for _ in range(16)]

        rewards = [-1, -1, -1, 40,
                   -1, -1, -10, -10,
                   -1, -1, -1, -1,
                   10, -2, -1, -1]

        self.terminal_states = [(0, 3), (3, 0)]

        self.grid, self.rewards = self.fillDict(values), self.fillDict(rewards)


