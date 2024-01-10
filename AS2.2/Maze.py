"""In this file we define create our maze"""


class Maze:

    def __init__(self):
        self.rewards = {}
        self.grid = {}
        self.terminal_states = []
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.episodes = self.fillDict([[] for _ in range(16)])

    def stepper(self, position, action):
        movement = self.actions[action]
        new_position = (position[0] + movement[0], position[1] + movement[1])

        if new_position in self.grid:
            return new_position
        else:
            return position

    def surrounding_states(self, position):
        """
        """
        states = {}
        for action in self.actions.keys():
            state = self.stepper(position, action)
            states[state] = [self.rewards[state], self.grid[state][-1]]
        return states

    def fillDict(self, value, sizeHorizontal=4, sizeVertical=4):
        """
        Filling a dictionary with coördinates.
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
        values = [[0] for _ in range(16)]

        rewards = [-1, -1, -1, 40,
                   -1, -1, -10, -10,
                   -1, -1, -1, -1,
                   10, -2, -1, -1]

        self.terminal_states = [(0, 3), (3, 0)]

        self.grid, self.rewards = self.fillDict(values), self.fillDict(rewards)


