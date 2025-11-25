# eval_a3c_mario.py

import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from a3c_mario5 import (
    ActorCriticNet,
    make_mario_env,
    preprocess_state,
    SAVE_PATH,
)


def make_eval_env(use_shaped_env: bool):
    """
    Evaluation env:
      - use_shaped_env=True  -> still uses ProgressRewardWrapper
      - use_shaped_env=False -> raw NES reward, no shaping
    """
    # train=False ensures we don't use training-only settings
    return make_mario_env(train=False, use_shaped_env=use_shaped_env)


def evaluate(
    num_episodes: int = 10,
    render: bool = True,
    use_shaped_env: bool = False,
    use_sampling: bool = False,
    use_random_policy: bool = False,
):
    """
    - use_shaped_env = True  -> run with shaped rewards (ProgressRewardWrapper)
      use_shaped_env = False -> run with raw NES rewards.
    - use_sampling = True    -> sample from policy (stochastic)
      use_sampling = False   -> greedy argmax
    - use_random_policy      -> ignore network and pick random actions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_eval_env(use_shaped_env=use_shaped_env)
    n_actions = env.action_space.n

    model = ActorCriticNet(num_actions=n_actions).to(device)
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_base_rewards = []
    all_shaped_rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_base = 0.0    # raw env reward
        ep_shaped = 0.0  # shaped reward if available

        while not done:
            if use_random_policy:
                action = env.action_space.sample()
            else:
                s = preprocess_state(state).to(device)
                with torch.no_grad():
                    logits, _ = model(s)
                    if use_sampling:
                        probs = F.softmax(logits, dim=-1)
                        m = Categorical(probs)
                        action = m.sample().item()
                    else:
                        # Greedy
                        action = torch.argmax(logits, dim=-1).item()

            next_state, reward, done, info = env.step(action)

            base_r = info.get("base_reward", reward)
            shaped_r = info.get("shaped_raw", None)

            ep_base += float(base_r)
            if shaped_r is not None:
                ep_shaped += float(shaped_r)

            state = next_state

            if render:
                env.render()
                time.sleep(1 / 60.0)

        all_base_rewards.append(ep_base)
        all_shaped_rewards.append(ep_shaped)

        if ep_shaped != 0.0:
            print(
                f"Episode {ep}: base_reward = {ep_base:.1f}, "
                f"shaped_reward = {ep_shaped:.1f}"
            )
        else:
            print(f"Episode {ep}: base_reward = {ep_base:.1f}")

    env.close()

    avg_base = sum(all_base_rewards) / len(all_base_rewards)
    if any(r != 0.0 for r in all_shaped_rewards):
        avg_shaped = sum(all_shaped_rewards) / len(all_shaped_rewards)
        print(
            f"\nAverage base reward over {num_episodes} eps: {avg_base:.1f} "
            f"(avg shaped: {avg_shaped:.1f})"
        )
    else:
        print(f"\nAverage base reward over {num_episodes} eps: {avg_base:.1f}")


if __name__ == "__main__":
    # EDIT THESE FLAGS TO TEST DIFFERENT THINGS
    evaluate(
        num_episodes=5,
        render=True,
        use_shaped_env=False,   # False = raw game reward
        use_sampling=True,     # False = greedy; True = stochastic sampling
        use_random_policy=False # True = ignore model, purely random actions
    )
