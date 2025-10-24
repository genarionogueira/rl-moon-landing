# LunarLander Double DQN (PyTorch)

## A clean, seminar-friendly implementation of a Double DQN agent for `Gymnasium`'s `LunarLander-v2`. It trains on CPU, logs learning progress, and renders videos of (a) the beginning of training and (b) the final trained agent.

## What you get

- PyTorch Double DQN with replay buffer and soft target updates
- Training script that saves weights/stats, exports a training chart, and records an early-training clip
- Run script that records a greedy rollout of the trained agent
- Docker and Poetry workflows

Artifacts written to `./output`:

- `lunarlander_dqn_weights.pt`
- `lunarlander_training_stats.npz`
- `lunarlander_training_chart.png`
- `lunarlander_beginning_training.mp4`
- `lunarlander_trained_agent.mp4`

---

## Quick start

### Run with Docker

```
docker compose build --no-cache && docker compose up
```

This will train the agent and then render a final video automatically. All outputs will be in `./output`.

### Run locally with Poetry

1. Install deps

```
poetry install
```

2. Train

```
env OUTPUT_DIR=./output poetry run train_lander
```

3. Render trained agent video

```
env OUTPUT_DIR=./output poetry run run_lander
```

---

## Educational notes (LunarLander-v2)

Observation (8 floats):

```
[x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
```

- `x, y` — lander position (x≈0, y≈0 near the pad)
- `vx, vy` — velocities (vy<0 means falling)
- `angle, angular_velocity` — attitude and rotation rate (0 is upright)
- `left_leg_contact, right_leg_contact` — 1.0 if that leg touches ground else 0.0

Actions (Discrete 4):

- `0` do nothing
- `1` fire left orientation engine (tilts right)
- `2` fire main engine (thrust up)
- `3` fire right orientation engine (tilts left)

Learning (high level):

- Interact with ε-greedy exploration; store transitions in replay buffer.
- Double DQN target: select next action via online net, evaluate via target net; minimize Huber loss to the one-step target.
- Soft target updates (Polyak averaging) for stability.

---

## Troubleshooting

- Docker “no space left on device”: free Docker disk space (`docker system prune -a --volumes`) or increase Docker Desktop disk size.
- Platform mismatch (ARM vs AMD64): if you’re not on ARM, remove `platform: linux/arm64` from `docker-compose.yml`.
- Box2D build issues: ensure `build-essential` and `swig` are available (Dockerfile installs these).
- Output write errors: ensure `./output` exists and is writeable; trainer saves to `/tmp` then moves files atomically.

---

## Customize

- Training length: edit `total_episodes` in `train_agent.py`.
- Exploration: adjust epsilon decay/floor in `train_agent.py`.
- Network size: change hidden sizes in `_mlp()` in `agent_class.py`.
- Soft update rate: change `tau` in `AgentConfig`.

---

## Acknowledgements

- Inspired by: https://github.com/juliankappler/lunar-lander
- `LunarLander-v2` environment from `Gymnasium`.
