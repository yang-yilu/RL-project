import numpy as np
from collections import deque
import cv2

# Try to import classic Gym; fall back to Gymnasium if needed.
try:
    import gym
    from gym import spaces
except ImportError:  # Gym not installed, use Gymnasium
    import gymnasium as gym
    from gymnasium import spaces

try:
    import gym_super_mario_bros
except ImportError:
    gym_super_mario_bros = None


def _process_frame_mario(frame):
    """Pre-process raw NES frame into (1, 84, 84) float32 array.

    This matches the original A3C Mario implementation:
    - reshape to (224, 256, 3) if needed
    - convert to grayscale with fixed RGB weights
    - resize to (84, 84)
    - add a channel dimension
    """
    if frame is None:
        return np.zeros((1, 84, 84), dtype=np.float32)

    frame = np.array(frame)

    # Older nes-py sometimes gives a flat array.
    if frame.ndim == 1:
        frame = frame.reshape(224, 256, 3)

    # Some versions give 240x256; crop to 224 like the original code.
    if frame.shape[0] == 240:
        frame = frame[16:240, :, :]

    frame = frame.astype(np.float32)
    frame = (
        frame[:, :, 0] * 0.2126
        + frame[:, :, 1] * 0.6152
        + frame[:, :, 2] * 0.0722
    )

    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = np.reshape(frame, (1, 84, 84))
    return frame.astype(np.float32)


class ProcessFrameMario(gym.Wrapper):
    """Reward shaping + frame preprocessing for Mario.

    This wrapper converts raw RGB frames to 1x84x84 grayscale and applies
    dense reward shaping based on distance, time, status and score.
    It is compatible with both old Gym and new Gymnasium-style APIs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 84, 84),
            dtype=np.float32,
        )
        # Shaping state
        self.prev_distance = 0.0
        self.prev_time = 400.0
        self.prev_status = 0.0
        self.prev_score = 0.0

    def step(self, action):
        """Gym / Gymnasium compatible step.

        Supports both:
        - obs, reward, done, info
        - obs, reward, terminated, truncated, info
        """
        result = self.env.step(action)

        if isinstance(result, tuple) and len(result) == 5:
            obs, _env_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _env_reward, done, info = result

        info = info or {}
        distance = float(info.get("distance", 0.0))
        time_left = float(info.get("time", self.prev_time))
        status = float(info.get("player_status", self.prev_status))
        score = float(info.get("score", self.prev_score))

        # === Reward shaping (adjusted for stable learning) ===
        reward = 0.0

        # Encourage positive progress, clip huge jumps.
        # FIX: Distance reward encourages forward movement (max +2.0 per step)
        reward += min(max(distance - self.prev_distance, 0.0), 2.0)

        # Time penalty to discourage standing still.
        # FIX: Soft time penalty (-0.01 per time unit lost) prevents overwhelming other signals
        reward += (self.prev_time - time_left) * -0.01

        # Reward for status improvements (e.g., power-ups).
        # FIX: Status reward encourages power-ups (e.g., small -> big = +5.0)
        reward += (status - self.prev_status) * 5.0

        # Reward for score increases.
        # FIX: Small score reward (0.025 per point) provides additional signal
        reward += (score - self.prev_score) * 0.025

        # Terminal bonus/penalty based on progress.
        # FIX: Reduced terminal penalty from ±50.0 to ±15.0 to prevent overwhelming learning signal
        # Level complete threshold is ~3225 pixels (end of level 1-1)
        if done:
            if distance > 3225:
                reward += 15.0  # Reduced from 50.0: level complete bonus
            else:
                reward -= 15.0  # Reduced from 50.0: early termination penalty

        # Update shaping state.
        self.prev_distance = distance
        self.prev_time = time_left
        self.prev_status = status
        self.prev_score = score

        processed_obs = _process_frame_mario(obs)
        return processed_obs, reward, done, info

    def reset(self, **kwargs):
        """Gym / Gymnasium-compatible reset.

        Handles both:
        - obs
        - (obs, info)
        and always returns a processed (1,84,84) observation.
        """
        self.prev_distance = 40.0
        self.prev_status = 2.0
        self.prev_score = 0.0
        self.prev_time = 400.0

        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _info = result
        else:
            obs = result

        return _process_frame_mario(obs)


class BufferSkipFrames(gym.Wrapper):
    """Skip frames and stack the last `skip` processed frames.

    Output shape: (skip, 84, 84)
    """

    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self.skip = skip
        self.buffer = deque(maxlen=skip)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(skip, 84, 84),
            dtype=np.float32,
        )

    def step(self, action):
        """Gym / Gymnasium-compatible step with frame skipping."""
        total_reward = 0.0
        self.buffer.clear()
        done = False
        info = {}
        obs = None

        for _ in range(self.skip):
            result = self.env.step(action)
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
                step_done = bool(terminated or truncated)
            else:
                obs, reward, step_done, info = result

            self.buffer.append(obs)
            total_reward += reward
            done = done or step_done

            if step_done:
                break

        # Pad with zeros if we terminated early in the skip window.
        while len(self.buffer) < self.skip:
            self.buffer.append(np.zeros_like(obs))

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (self.skip, 84, 84))
        return frame.astype(np.float32), total_reward, done, info

    def reset(self, **kwargs):
        self.buffer.clear()
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _info = result
        else:
            obs = result

        for _ in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (self.skip, 84, 84))
        return frame.astype(np.float32)


class NormalizedEnv(gym.Wrapper):
    """
    Simple, stable observation normalization for A3C Mario.

    Converts raw 0–255 uint8 frames to float32 in [0, 1]. We intentionally
    avoid running mean/std whitening here, because the old implementation
    can produce NaNs on newer NumPy versions.
    """

    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        # These are kept in case you want to experiment later, but not used now.
        self.state_mean = 0.0
        self.state_std = 1.0
        self.alpha = 0.9999
        self.eps = 1e-8
        self.num_steps = 0

    def observation(self, observation):
        if observation is None:
            return observation

        # Stable, simple normalization: scale 0–255 pixels to [0, 1].
        observation = np.asarray(observation, dtype=np.float32)
        observation = observation / 255.0
        return observation


def wrap_mario(env):
    """Apply preprocessing stack to a Mario env."""
    env = ProcessFrameMario(env)
    #env = NormalizedEnv(env)
    env = BufferSkipFrames(env)
    return env


def create_mario_env(env_id):
    """Create a Super Mario Bros env with wrappers.

    Works for both:
    - old gym-based gym-super-mario-bros
    - newer gymnasium-based versions with apply_api_compatibility
    """
    if gym_super_mario_bros is not None:
        try:
            # Newer API: get classic (obs, reward, done, info) on top of Gymnasium.
            env = gym_super_mario_bros.make(
                env_id,
                apply_api_compatibility=True,
                render_mode="rgb_array",
            )
        except TypeError:
            # Older versions don't support apply_api_compatibility.
            env = gym_super_mario_bros.make(env_id)
    else:
        env = gym.make(env_id)

    env = wrap_mario(env)
    return env
