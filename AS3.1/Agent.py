import numpy as np


class Agent:

    def __init__(self, policy: classmethod, memory: int, discount: float):
        self.policy = policy
        self.memory = memory
        self.discount = discount

    def train(self):
        states, actions, rewards, next_states, terminated = self.memory.sample()
        actions_states = self.policy.model.predict(states)
        actions_next_states = self.policy.model.predict(next_states)
        q_value = np.copy(actions_states)

        for row, action in zip(range(len(actions_states[0])), actions):
            # for each row, only calculate the new action value that was chosen by the nn
            # if terminated, value = reward, else bellman equation
            if terminated[row]:
                actions_next_states[0][row][action] = rewards[row]
            else:
                actions_next_states[0][row][action] = rewards[row] + self.discount * np.max(actions_next_states[0][row])

        # print(states, 'hoi', states.shape)
        # print(q_value, 'doei', q_value.shape)
        self.policy.model.train_on_batch(states, q_value)

        # self.policy.decay()
