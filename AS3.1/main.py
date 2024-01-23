import gym
from Agent import Agent
from Policy import Policy
from Memory import Memory
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

episodes = 1000
max_steps_episode = 2000
max_memory_size = 32000
batch_size = 32
learning_rate = 0.001
scores = []
average_scores = []

# Initialize replay memory D to capacity N
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
    while steps < max_steps_episode and not terminated:
        # With probability epsilon select a random action a_t, otherwise select a_t = argmax_a Q(φ_t, a; θ)
        action = agent.policy.select_action(observation)
        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        next_observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        score += reward
        transition = (observation, action, reward, next_observation, terminated)
        # Store transition φ_t, a_t, r_t, φ_t+1 in D
        agent.memory.store(transition)
        observation = next_observation
        # Gather 4 iterations data so batch size of 32 can always be filled
        if episode > 4:
            agent.train()

    print('\n', "Scores: ", scores)
    print("AVG_Scores: ", average_scores, '\n')
    scores.append(score)
    # Gemiddelde van laatste 50 episodes for smoothness
    last_scores = np.mean(scores[-50:])
    average_scores.append(last_scores)
    print("episode", episode, f"score {score}", f"average score {last_scores}")

env.close()

plt.plot(np.arange(episodes), np.array(average_scores), label="Average score")
plt.xticks(np.arange(0, episodes+1, 100))
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.savefig(f'../images/AS_3.1_visualization.png')
plt.show()
