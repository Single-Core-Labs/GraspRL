# GraspRL

GraspRL is a Python-first, simulation-first toolkit that turns robot demonstrations and simulated environments into trained, evaluated and exportable robot-arm control policies.

## Installation

```bash
pip install -e .[dev]
```

## Quick Start

```bash
# Validate demonstration data
grasp-rl data validate path/to/data.json

# Train a policy
grasp-rl train --config configs/ppo_reach.yaml --env robotarm/reach-v0

# Evaluate a checkpoint
grasp-rl eval --checkpoint runs/latest/checkpoint.pt --episodes 100
```

## License

Apache-2.0
