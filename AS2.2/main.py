"""In this file we make sure everything is connected and run properly"""

from Maze import Maze
from Agent import Agent
from Policy import Policy

delta_threshold = 0.01
start_position = (3, 2)

maze = Maze()
maze.position = start_position
maze.create_maze_values()

policy = Policy(maze)

agent = Agent(start_position, maze, policy, delta_threshold)
agent.value_iteration()

# agent.temporal_difference(0.5, 1, 10)
agent.temporal_difference(0.5, 0.5, 10)
# agent.sarsa(0.25, 1, 0.1, 2000)
# agent.sarsa(0.25, 0.9, 0.1, 20000)
# agent.q_learning(0.25, 1, 0.1, 50000)
# agent.q_learning(0.5, 0.9, 0.1, 50000)
