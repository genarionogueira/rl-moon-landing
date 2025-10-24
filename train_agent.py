"""Train a Double DQN agent for LunarLander-v2 (PyTorch).

Educational overview
--------------------
This script wires up the training loop around the Agent from agent_class.py.
It logs rewards, saves weights/statistics, draws a simple chart, and records a
short "beginning of training" clip to illustrate early behavior.

Saves:
- OUTPUT_DIR/lunarlander_dqn_weights.pt
- OUTPUT_DIR/lunarlander_training_stats.npz (arrays: ep_rewards, losses)
- OUTPUT_DIR/lunarlander_training_chart.png (rewards curve + moving average)
- OUTPUT_DIR/lunarlander_beginning_training.mp4 (early training clip)
"""

import os
from pathlib import Path
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import gymnasium as gym
from agent_class import Agent, AgentConfig


def _record_clip(agent: Agent, seconds: int = 12, eps: float = 1.0, fps: int = 30):
    """Record a short greedy/ε-greedy rollout as an MP4 clip.

    We set the environment to rgb_array mode so that env.render() returns frames
    (H, W, 3) which are collected and written to disk with imageio.
    """
    env_r = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, info = env_r.reset(seed=123)
    frames = []
    target_frames = seconds * fps
    while len(frames) < target_frames:
        action = agent.act(obs, eps=eps)
        obs, reward, terminated, truncated, info = env_r.step(action)
        frames.append(env_r.render())
        if terminated or truncated:
            obs, info = env_r.reset()
    env_r.close()
    return frames


def train(total_episodes: int = 800):
    """Main training loop.

    - Linear epsilon decay across 90% of episodes
    - Learning happens inside agent.maybe_learn() when enough data exists
    - We print progress every 20 episodes for visibility
    """
    output_dir = os.environ.get("OUTPUT_DIR", str(Path(__file__).parent / "output"))
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("LunarLander-v2")
    # Observation semantics (LunarLander-v2 returns an 8-D float vector):
    #   obs = [
    #     x,                  # horizontal position (0 = pad center, negative = left, positive = right)
    #     y,                  # vertical position (roughly 0 at the landing pad)
    #     vx,                 # horizontal velocity
    #     vy,                 # vertical velocity (negative = falling)
    #     angle,              # lander angle (0 = upright)
    #     angular_velocity,   # angular velocity
    #     left_leg_contact,   # 1.0 if left leg is in contact with ground, else 0.0
    #     right_leg_contact   # 1.0 if right leg is in contact with ground, else 0.0
    #   ]
    # Example (typical at the very beginning, values vary with seed):
    #   obs ≈ [0.0, 1.4, 0.0, -0.2, 0.05, 0.0, 0.0, 0.0]
    #
    # Actions (Discrete(4)):
    #   0 → do nothing
    #   1 → fire left orientation engine (rotates/tilts right)
    #   2 → fire main engine (thrust up)
    #   3 → fire right orientation engine (rotates/tilts left)
    
    # The policy’s job is to combine these to slow descent, center over the pad,
    # keep angle near 0, and touch down with both legs gently.
    # Step API returns: (next_obs, reward, terminated, truncated, info)
    #   - reward: shaped reward encouraging gentle, centered landings
    #   - terminated: True if episode ends by success/failure
    #   - truncated: True if time limit reached
    #   - info: auxiliary diagnostics
    obs, info = env.reset(seed=0)
    cfg = AgentConfig(obs_dim=env.observation_space.shape[0], num_actions=env.action_space.n)
    agent = Agent(cfg)

    ep_rewards = []
    losses = []

    epsilon_start, epsilon_end = 1.0, 0.02
    epsilon_decay_episodes = total_episodes * 0.9

    early_clip_saved = False
    for ep in range(total_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_episodes))
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # epsilon-greedy: random with prob ε, otherwise argmax Q(s,·)
            action = agent.act(obs, eps=epsilon)
            # apply action in env → next state, reward, done flags
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Gymnasium splits episode end into termination (goal/fail) or truncation (time limit)
            done = terminated or truncated
            # store transition (s, a, r, s', done) in replay buffer
            agent.push(obs, action, reward, next_obs, float(done))
            # if enough data and at train interval: sample batch, Double DQN update, return loss
            loss = agent.maybe_learn()
            # an update happened this step
            if loss is not None:
                # log training loss for charting/analysis later
                losses.append(loss)
            # move agent's current state pointer forward
            obs = next_obs
            # accumulate episode return for logging
            ep_reward += reward
        ep_rewards.append(ep_reward)
        # Save an "early training" clip after the first episode finishes
        if not early_clip_saved:
            try:
                frames = _record_clip(agent, seconds=12, eps=1.0, fps=30)
                imageio.mimsave(f"{output_dir}/lunarlander_beginning_training.mp4", frames, fps=30, macro_block_size=None)
                early_clip_saved = True
                print(f"Saved early training clip to {output_dir}/lunarlander_beginning_training.mp4")
            except Exception as e:
                print(f"Failed to save early training clip: {e}")

        if (ep + 1) % 20 == 0:
            print(f"[Train] Ep {ep+1}/{total_episodes} | reward={ep_reward:.1f} | eps={epsilon:.3f}")

    env.close()
    # Save weights and stats (write to /tmp then move atomically into OUTPUT_DIR)
    weights_path = f"{output_dir}/lunarlander_dqn_weights.pt"
    try:
        tmp_path = "/tmp/lunarlander_dqn_weights.pt"
        agent.save(tmp_path)
        os.replace(tmp_path, weights_path)
    except Exception as e:
        print(f"Primary save failed ({e}). Attempting direct write to {weights_path}...")
        agent.save(weights_path)
    np.savez_compressed(f"{output_dir}/lunarlander_training_stats.npz", ep_rewards=np.array(ep_rewards), losses=np.array(losses))
    print(f"Saved weights to {weights_path}")

    # Save a training chart
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(ep_rewards, label='Episode reward', alpha=0.7, linewidth=1.6)
        if len(ep_rewards) >= 20:
            window = 20
            mov_avg = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, window-1+len(mov_avg)), mov_avg, label='Moving avg (20)', color='C1', linewidth=2.0)
        ax.set_ylabel('Reward')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/lunarlander_training_chart.png")
        plt.close(fig)
        print(f"Saved training chart to {output_dir}/lunarlander_training_chart.png")
    except Exception as e:
        print(f"Failed to save training chart: {e}")


if __name__ == "__main__":
    train()


