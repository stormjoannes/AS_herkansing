import random
import numpy as np


class Memory:

    def __init__(self, batch_size: int, max_memory_size: int):
        """
        wasd

            Parameters:
                 batch_size(int): Amount of rows data for each batch
                 max_memory_size(int): Max amount of steps the agent will remember
        """
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.deque = []

    def store(self, transition: tuple):
        """
        Remove begin of memory, to prevent training on bad data

            Parameter:
            transition(tuple): A tuple with arrays of ...
        """
        print(len(self.deque))
        if len(self.deque) > self.max_memory_size:
            del self.deque[0]
        self.deque.append(transition)

    def sample(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Take sample of size batch size from the memory

            Return:
                np.ndarray: Sates, actions, rewards, next_states and terminated from the sample transitions
        """
        # print("length ", len(self.deque))
        batch = random.sample(self.deque, self.batch_size)
        states, actions, rewards, next_states, terminated = [], [], [], [], []

        for transitie in batch:
            states.append(transitie[0])
            actions.append(transitie[1])
            rewards.append(transitie[2])
            next_states.append(transitie[3])
            terminated.append(transitie[4])

        return np.array([states]), np.array(actions), np.array(rewards), np.array([next_states]), np.array(terminated)
