import gym
from Agent import Agent
from Policy import Policy
from Memory import Memory
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

episodes = 1000
max_steps = 1000
max_memory_size = 10000
batch_size = 32
learning_rate = 0.1
scores = []
average_scores = []

memory = Memory(batch_size, max_memory_size)
policy = Policy(1)
agent = Agent(policy, memory, 0.99)
actions = env.action_space.n
dimensions = env.observation_space.shape
print(dimensions)
agent.policy.setup_model(dimensions[0], actions, learning_rate)
print("setup model")


for episode in range(episodes):
    observation, info = env.reset()
    steps = 0
    score = 0
    terminated = False
    while steps < max_steps and not terminated:
        # print('1')
        action = agent.policy.select_action(observation)
        # print('2')
        next_observation, reward, terminated, truncated, info = env.step(action)
        # print('3')
        steps += 1
        score += reward
        # print('4')
        transition = (observation, action, reward, next_observation, terminated)
        # print('5')
        agent.memory.store(transition)
        # print('6')
        observation = next_observation
        # Gather 2 iterations data so batch size of 32 can always be filled
        if episode > 4:
            agent.train()

    scores.append(score)
    avg = np.mean(scores[-100:])
    average_scores.append(avg)

    print("episode", episode, "score %.2f" % score, "average score %.2f" % avg, "epsilon %.2f" % agent.policy.epsilon, "steps", steps)

print(scores)
print(average_scores)

env.close()

plt.plot(np.arange(episodes), np.array(scores), label="Score")
plt.plot(np.arange(episodes), np.array(average_scores), label="Average score")
plt.xticks(np.arange(0, episodes+1, 100))
plt.xlabel("episode")
plt.ylabel("score")
plt.legend()
plt.savefig(f'../images/AS_3.1_visualization.png')
plt.show()
