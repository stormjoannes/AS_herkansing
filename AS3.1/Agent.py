import numpy as np


class Agent:

    def __init__(self, policy: classmethod, memory: int, discount: float):
        """
        Set class values

            Parameters:
                 policy(classmethod): Current policy class
                 memory(int): Max amount of memory that is stored
                 discount(int): Amount of discount for the agent, impacts the importance of values/rewards
        """
        self.policy = policy
        self.memory = memory
        self.discount = discount

    def train(self):
        """
        Predict actions, calculate new values and train the model.
        """
        states, actions, rewards, next_states, terminated = self.memory.sample()
        actions_states = self.policy.model.predict(states)
        actions_next_states = self.policy.model.predict(next_states)
        q_value = np.copy(actions_states)

        # Loop for each row and calculate action state value
        for row, action in zip(range(len(actions_states[0])), actions):
            if terminated[row]:
                actions_next_states[0][row][action] = rewards[row]
            else:
                actions_next_states[0][row][action] = rewards[row] + self.discount * np.max(actions_next_states[0][row])

        self.policy.model.train_on_batch(states, q_value)
        self.policy.decay()
