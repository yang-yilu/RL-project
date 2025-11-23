# test_dqn.py
import numpy as np
import torch
from env_mario import make_mario_env
from dqn_model import DQN


def _process_obs(obs):
    obs = np.array(obs)

    # If FrameStack gives (K, H, W, C) -> (H, W, C*K)
    if obs.ndim == 4:
        obs = np.transpose(obs, (1, 2, 3, 0))  # (K,H,W,C)->(H,W,C,K)
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)  # (H,W,C*K)

    return obs


def _reset_env(env):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return _process_obs(obs)


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 4:
        next_obs, r, done, info = out
    else:
        next_obs, r, terminated, truncated, info = out
        done = terminated or truncated
    return _process_obs(next_obs), r, done, info


def test(model_path="mario_dqn.pt", episodes=5, render=True):
    env = make_mario_env()  # should be the SAME wrappers as training
    n_actions = env.action_space.n

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    q = DQN(n_actions).to(device)
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()

    ep_rewards = []
    ep_lengths = []
    ep_xpos = []

    for ep in range(episodes):
        obs = _reset_env(env)
        done = False
        total_r = 0
        steps = 0
        max_x = 0

        while not done:
            steps += 1

            with torch.no_grad():
                inp = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
                qs = q(inp).cpu().numpy()
            a = int(qs.argmax())

            obs, r, done, info = _step_env(env, a)
            total_r += r

            # info usually contains x_pos (progress). If present, track it.
            if isinstance(info, dict) and "x_pos" in info:
                max_x = max(max_x, info["x_pos"])

            if render:
                env.render()

        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        ep_xpos.append(max_x)

        print(f"Episode {ep+1}: reward={total_r:.1f}, steps={steps}, max_x={max_x}")

    env.close()

    print("\n=== Summary ===")
    print("Avg reward:", np.mean(ep_rewards))
    print("Avg steps:", np.mean(ep_lengths))
    if any(ep_xpos):
        print("Avg max_x:", np.mean(ep_xpos))


if __name__ == "__main__":
    test()
