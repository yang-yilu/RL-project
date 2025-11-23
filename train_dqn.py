# train_dqn.py
import random
import numpy as np
import torch
import torch.nn.functional as F

from env_mario import make_mario_env
from dqn_model import DQN
from replay_buffer import ReplayBuffer


def _process_obs(obs):
    obs = np.array(obs)

    # If FrameStack gives (K, H, W, C) -> convert to (H, W, C*K)
    if obs.ndim == 4:
        # assume stack is first axis
        # (K,H,W,C) -> (H,W,C,K)
        obs = np.transpose(obs, (1, 2, 3, 0))
        # (H,W,C,K) -> (H,W,C*K)
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)

    # If it gives (H, W, C, K) already, just flatten last two dims
    elif obs.ndim == 4 and obs.shape[-1] <= 4:
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)

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



def train():
    env = make_mario_env()
    n_actions = env.action_space.n

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    q = DQN(n_actions).to(device)
    q_tgt = DQN(n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q.train()
    q_tgt.eval()

    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    buf = ReplayBuffer(100000)

    gamma = 0.99
    batch_size = 32
    start_learn = 10000
    target_update = 5000
    max_ep_steps = 5000   # avoids infinite “stuck” episodes

    eps, eps_min, eps_decay = 1.0, 0.1, 1e-6
    total_steps = 0
    episode = 0

    while total_steps < 1_000_000:
        obs = _reset_env(env)
        done = False
        ep_reward = 0
        ep_steps = 0

        while not done and ep_steps < max_ep_steps:
            ep_steps += 1
            total_steps += 1

            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    qs = q(np.expand_dims(obs, 0)).cpu().numpy()
                a = int(qs.argmax())

            next_obs, r, done, info = _step_env(env, a)

            buf.push(obs, a, r, next_obs, done)
            obs = next_obs
            ep_reward += r

            eps = max(eps_min, eps - eps_decay)

            if len(buf) >= start_learn:
                s, a_b, r_b, ns, d_b = buf.sample(batch_size)

                s  = torch.tensor(s,  dtype=torch.float32, device=device)
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                a_b = torch.tensor(a_b, dtype=torch.int64, device=device)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
                d_b = torch.tensor(d_b, dtype=torch.float32, device=device)

                q_sa = q(s).gather(1, a_b.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    q_next = q_tgt(ns).max(1)[0]
                    target = r_b + gamma * q_next * (1 - d_b)

                loss = F.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

                if total_steps % target_update == 0:
                    q_tgt.load_state_dict(q.state_dict())
                    print("target updated")

        episode += 1
        print(f"ep {episode} | steps {total_steps} | reward {ep_reward:.1f} | eps {eps:.3f}")

        if episode % 20 == 0:
            torch.save(q.state_dict(), "mario_dqn.pt")
            print("saved mario_dqn.pt")

    env.close()


if __name__ == "__main__":
    train()
