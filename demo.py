import gym
import random
# d = gym.envs.registry
# for i in d:
#     print(i)
# Pong-v0
# PongDeterministic-v0
# PongNoFrameskip-v0
# Pong-v4
# PongDeterministic-v4
# PongNoFrameskip-v4

env = gym.make('PongNoFrameskip-v4', render_mode='human')
env.seed(0)
observation, info = env.reset()
while True:
    env.render()
    action = random.randint(0, 5)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break


    

