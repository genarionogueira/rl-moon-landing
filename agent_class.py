"""PyTorch Double DQN agent for LunarLander-v2.

Educational overview
--------------------
This file contains a minimal-yet-robust Double DQN implementation in PyTorch.
It is deliberately written in a clear, modular style for teaching:

- ReplayBuffer stores transitions (s, a, r, s', done) and supports sampling.
- Agent holds the online Q-network and the target Q-network, the optimizer,
  and the learning routine implementing Double DQN with soft target updates.
- We keep the network small (two hidden layers) so training is fast on CPU.

Files in this project
---------------------
- agent_class.py: defines the Agent and supporting utilities
- train_agent.py: trains the agent, logs rewards/losses, saves weights
- run_agent.py: loads saved weights and records a video of greedy behavior

Core algorithmic choices
------------------------
- Double DQN: action selection uses the online network, action evaluation uses
  the target network to reduce overestimation bias.
- Soft target updates (Polyak averaging): small incremental updates from online
  to target make learning more stable than periodic hard copies.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _mlp(in_dim: int, out_dim: int) -> nn.Module:
    """Create a small MLP mapping observations to Q-values.

    in_dim:  observation dimension (LunarLander has 8 floats)
    out_dim: number of discrete actions (LunarLander has 4 actions)
    """
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, out_dim),
    )


class ReplayBuffer:
    """Simple ring buffer for experience replay.

    We store fixed-size arrays for observations, actions, rewards, next
    observations, and done flags. Sampling returns PyTorch tensors ready for
    training on CPU/GPU.
    """
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity,), dtype=np.int64)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, o, a, r, no, d):
        i = self.idx % self.capacity
        self.obs[i] = o
        self.acts[i] = a
        self.rews[i] = r
        self.next_obs[i] = no
        self.dones[i] = d
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Uniformly sample a minibatch of transitions.

        Returns tensors (o, a, r, no, d) with shapes:
          - o, no: [batch, obs_dim]
          - a:     [batch]
          - r, d:  [batch]
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        o = torch.from_numpy(self.obs[idxs])
        a = torch.from_numpy(self.acts[idxs])
        r = torch.from_numpy(self.rews[idxs])
        no = torch.from_numpy(self.next_obs[idxs])
        d = torch.from_numpy(self.dones[idxs])
        return o, a, r, no, d


@dataclass
class AgentConfig:
    obs_dim: int
    num_actions: int
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.005              # soft target update rate
    buffer_capacity: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 4


class Agent:
    """Double DQN agent with soft (Polyak) target updates.

    Typical usage:
      - Construct with AgentConfig
      - Call act(obs, eps) to pick actions (epsilon-greedy)
      - Push transitions into the buffer and call maybe_learn() each step
      - Periodically save() to persist weights
    """
    def __init__(self, cfg: AgentConfig, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = _mlp(cfg.obs_dim, cfg.num_actions).to(self.device)
        self.q_target = _mlp(cfg.obs_dim, cfg.num_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber
        self.cfg = cfg
        self.rb = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)
        self.total_steps = 0

    def act(self, obs: np.ndarray, eps: float) -> int:
        """Epsilon-greedy action selection on a single observation."""
        if np.random.rand() < eps:
            return np.random.randint(0, self.cfg.num_actions)
        with torch.no_grad():
            q = self.q(torch.from_numpy(obs).float().to(self.device).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    def push(self, o, a, r, no, d):
        """Insert one transition into the replay buffer."""
        self.rb.add(o, a, r, no, d)

    def _soft_update(self):
        """Polyak-averaging: q_target ← (1-τ)·q_target + τ·q."""
        with torch.no_grad():
            for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def maybe_learn(self) -> float | None:
        """Perform one learning step if conditions are met.

        Conditions:
          - Enough samples in the replay buffer (learning_starts)
          - Train every cfg.train_freq environment steps

        Returns the scalar loss if an update happened, otherwise None.
        """
        self.total_steps += 1
        if self.rb.size < self.cfg.learning_starts:
            return None
        if self.total_steps % self.cfg.train_freq != 0:
            return None

        o, a, r, no, d = self.rb.sample(self.cfg.batch_size)
        o = o.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        no = no.to(self.device)
        d = d.to(self.device)

        # Double DQN target and loss
        # --------------------------
        # Goal: Make the online Q-network produce values that satisfy the
        # Bellman optimality equation. We build a one-step bootstrapped target
        # for the taken actions and then minimize the discrepancy (TD error)
        # between the current Q-value and that target.
        #
        # 1) Target construction (Double DQN):
        #    - Select the greedy next action using the ONLINE net:
        #        a* = argmax_a Q_online(s', a)
        #    - Evaluate that action’s value using the TARGET net:
        #        Q_target(s', a*)
        #    - One-step target:
        #        y = r + γ · (1 - done) · Q_target(s', a*)
        with torch.no_grad():
            next_online_q = self.q(no)
            next_actions = torch.argmax(next_online_q, dim=1)
            next_target_q = self.q_target(no)
            max_next = next_target_q.gather(1, next_actions.view(-1, 1)).squeeze(1)
            target = r + self.cfg.gamma * (1.0 - d) * max_next

        # 2) Current prediction for the taken actions:
        curr_q = self.q(o).gather(1, a.view(-1, 1)).squeeze(1)

        # 3) TD loss: we minimize a robust (Huber) loss between prediction and
        #    target. This behaves like L2 near zero and like L1 for large
        #    residuals, which stabilizes training when targets are noisy.
        #    Intuition: “Make today’s Q-value look like what we’d expect if we
        #    took the best action next and then continued optimally.”
        loss = self.loss_fn(curr_q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # Mild gradient clipping helps prevent rare exploding gradients
        nn.utils.clip_grad_value_(self.q.parameters(), 1.0)
        self.opt.step()

        self._soft_update()
        return float(loss.item())

    def save(self, path: str | os.PathLike):
        """Save online network weights to the given path."""
        torch.save(self.q.state_dict(), path)

    def load(self, path: str | os.PathLike):
        """Load weights and sync target network to online network."""
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q.state_dict())


