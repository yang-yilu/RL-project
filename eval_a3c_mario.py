# eval_a3c_mario.py
import time
import torch

from a3c_mario import (
    ActorCriticNet,
    make_mario_env,
    preprocess_state,
    SAVE_PATH,
)

def evaluate(num_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create env (same wrappers as during training)
    env = make_mario_env()
    n_actions = env.action_space.n

    # Build model and load weights
    model = ActorCriticNet(num_actions=n_actions).to(device)
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            s = preprocess_state(state).to(device)

            with torch.no_grad():
                logits, value = model(s)
                # Greedy action (argmax over policy)
                action = torch.argmax(logits, dim=-1).item()

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state

            if render:
                env.render()
                # slow it down a bit so you can see it
                time.sleep(1 / 60.0)

        all_rewards.append(ep_reward)
        print(f"Episode {ep}: reward = {ep_reward:.1f}")

    env.close()

    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.1f}")


if __name__ == "__main__":
    evaluate(num_episodes=100, render=True)
