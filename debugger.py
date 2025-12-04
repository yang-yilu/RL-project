import time
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from a3c_mario5 import GrayResizeObs, FrameStack  # reuse your wrappers

def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayResizeObs(env)
    env = FrameStack(env, k=4)

    print("SIMPLE_MOVEMENT:", SIMPLE_MOVEMENT)

    for ep in range(3):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            x = info.get("x_pos", -1)
            print(f"[RANDOM] ep={ep} step={steps} action={action} ({SIMPLE_MOVEMENT[action]}) "
                  f"x_pos={x} reward={reward:.3f} done={done}")
            steps += 1
            env.render()
            time.sleep(1/60.0)
    env.close()

if __name__ == "__main__":
    main()
