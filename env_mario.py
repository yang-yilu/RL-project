# env_mario.py
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

def make_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)     # (H,W,1)
    env = ResizeObservation(env, (84, 84))             # (84,84,1)
    env = FrameStack(env, 4)                           # (84,84,4)
    return env
