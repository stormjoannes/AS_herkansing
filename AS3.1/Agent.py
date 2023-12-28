
class Agent:

    def __init__(self, policy, memory, discount):
        self.policy = policy
        self.memory = memory
        self.discount = discount

    def train(self):
        pass
