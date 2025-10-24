"""Run a trained LunarLander agent and save a video.

Educational notes
-----------------
This script loads the saved PyTorch weights and runs a greedy (ε=0) policy
to create a short MP4 of the agent landing the lander.

Usage (after training):
  python run_agent.py
"""

import os
from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
from agent_class import Agent, AgentConfig


def run_video(seconds: int = 20):
    """Render a greedy rollout video from the trained agent."""
    output_dir = os.environ.get("OUTPUT_DIR", str(Path(__file__).parent / "output"))
    os.makedirs(output_dir, exist_ok=True)
    weights_path = f"{output_dir}/lunarlander_dqn_weights.pt"

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    cfg = AgentConfig(obs_dim=env.observation_space.shape[0], num_actions=env.action_space.n)
    agent = Agent(cfg)
    agent.load(weights_path)

    fps = 30
    frames = []
    target_frames = seconds * fps
    while len(frames) < target_frames:
        action = agent.act(obs, eps=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    out = f"{output_dir}/lunarlander_trained_agent.mp4"
    imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
    print(f"Saved video to {out}")


if __name__ == "__main__":
    run_video()


