import random
import numpy as np


class Memory:

    def __init__(self, batch_size, max_memory_size):
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.deque = []

    def store(self, transition):
        """
        Remove begin of memory, to prevent training on bad data
        """
        print(len(self.deque))
        if len(self.deque) > self.max_memory_size:
            del self.deque[0]
        self.deque.append(transition)

    def sample(self):
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
