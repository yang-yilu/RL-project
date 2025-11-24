import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
done = False

for _ in range(200):
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        state = env.reset()

env.close()
print("Mario env OK")
