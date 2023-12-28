import gymnasium as gym
from Agent import Agent
from Policy import Policy
from Memory import Memory

env = gym.make("LunarLander-v2",
               ender_mode="human")
observation, info = env.reset(seed=42)

policy = Policy(1)

episodes = 100
max_steps = 2000

for index in range(episodes):
    observation, info = env.reset()
    steps = 0
    terminated = False
    while steps < 2000 and not terminated:
        steps += 1
        action = agent.policy.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        transition = (observation, action, reward, next_observation, terminated)
        agent.memory.store(transition)
        observation = next_observation
        # Gather 2 iterations data before training
        if index > 2:
            agent.train()


env.close()