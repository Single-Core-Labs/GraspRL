# Frontier RL — PRD, PoC and Build Prompts

v0.1 draft · working product name · open-core robotics RL library

## 1. Product Requirements Document (PRD)

### 1.1 Summary

Frontier RL is a Python-first, simulation-first toolkit that turns robot demonstrations and simulated environments into trained, evaluated and exportable robot-arm control policies. It wraps established RL implementations and adds the parts teams usually rebuild themselves: data contracts, standard robot environments, reproducible evaluation, safety constraints and deployment packaging.

### 1.2 Problem

Robotics teams that want RL policies for arms face a fragmented workflow. Data formats differ per robot, environments are bespoke, training runs are hard to resume or reproduce, evaluation is ad hoc, and there is no clean boundary between a learned policy and the safety layer that must wrap it. General RL libraries solve the algorithm, not this workflow.

### 1.3 Target users

- Robotics startups building arm products.
- University and corporate research labs.
- Embodied AI teams needing a reproducible training and evaluation baseline.
- Individual developers prototyping on a workstation (RTX 4050, 6 GB VRAM, 16 GB RAM, or CPU only).

### 1.4 Goals

1. A new developer goes from install to a trained, evaluated reaching policy in under one hour of effort on a laptop-class machine.
2. Every run is reproducible and resumable, with a self-contained experiment directory.
3. Safety limits are enforced outside the learned policy.
4. Components (simulator, algorithm, robot adapter) are replaceable behind documented interfaces.

### 1.5 Non-goals for v0.1

- Reimplementing PPO or SAC from scratch.
- Image-based policies, distributed training, sim-to-real transfer, multi-agent RL, online learning.
- Direct real-hardware control. Hardware support is a later, explicitly enabled integration.
- Any unmeasured claim such as a fixed accuracy or latency target.

### 1.6 User stories

- As a developer, I validate a demonstration file and get clear errors for bad units, timestamps or shapes.
- As a developer, I train a PPO policy on a reaching task with one command and a YAML config.
- As a developer, I resume a training run that crashed without losing progress.
- As a researcher, I evaluate a checkpoint on held-out seeds and get success rate and position-error statistics.
- As an integrator, I load a policy through a Python SDK and receive a proposed action that my own controller validates.
- As a team lead, I compare runs through versioned configs, logs and reports.

### 1.7 Functional requirements

**P0 (required for v0.1)**

| ID | Requirement |
| --- | --- |
| F1 | Versioned JSON schemas for trajectory-only and state-action data, with a validator (dimensions, finite values, units, frames, timestamps) |
| F2 | Episode-level train/validation/test splitting; source data never mutated; preprocessing logged |
| F3 | Gymnasium-compliant `RobotArmEnv` base with named observations, explicit action modes, `terminated` vs `truncated` handled correctly |
| F4 | One lightweight mathematical trajectory-tracking env, clearly labelled as non-physical |
| F5 | One MuJoCo-backed arm reaching env (`robotarm/reach-v0`) |
| F6 | Composable, configurable reward terms with documented scales |
| F7 | Safety layer: joint, workspace and action limits enforced independently of reward; NaN/stale action rejection |
| F8 | PPO training via a wrapped established library, YAML config, CSV logging |
| F9 | Resumable checkpoints (model, optimizer, RNG state, config, versions) |
| F10 | Evaluation module: success rate, mean and p95 position error, constraint violations, JSON report |
| F11 | Python policy SDK: `load`, `predict`, `reset`, `metadata`, with checkpoint validation |
| F12 | CLI: `data validate`, `train`, `eval` |

**P1 (stretch within v0.1 if time allows)**

| ID | Requirement |
| --- | --- |
| F13 | Behavior cloning from paired state-action demonstrations |
| F14 | ONNX export with numerical agreement test |
| F15 | `serve` command: optional HTTP inference endpoint |
| F16 | Optional TensorBoard logging |

**Later (post v0.1)**

SAC, manipulation tasks, vision policies, enterprise registry and access control, hardware adapters.

### 1.8 Non-functional requirements

- Supported Python 3.10 to 3.12 on Linux; macOS and Windows best effort.
- State-based MLP training must fit on 6 GB VRAM or run CPU-only.
- Reproducibility: same config and seed give evaluation results within a documented tolerance on CPU. GPU determinism is not guaranteed.
- Test coverage on schema validation, safety limits and checkpoint resume.
- Apache-2.0 core; dependency licenses verified before publishing.

### 1.9 Success metrics (measure first, then set targets)

| Metric | v0.1 criterion |
| --- | --- |
| Clean install | Works in a fresh virtual environment |
| Env correctness | Gymnasium `check_env` passes |
| Training | PPO completes the reference reaching task |
| Success rate | Reported over 100 held-out episodes |
| Tracking quality | Mean and p95 end-effector error in metres |
| Safety | Zero unhandled hard-limit violations in the benchmark suite |
| Recovery | Resume from `latest.pt` continues training |
| Inference | p50 and p95 latency reported with hardware details |
| Reproducibility | Repeat run matches within documented tolerance |

The reaching success threshold is set after the baseline is measured, not before.

### 1.10 Risks

| Risk | Mitigation |
| --- | --- |
| MuJoCo arm model takes longer than planned | Start with a small self-authored arm model; adopt a third-party model later after checking its license |
| Wrapped PPO makes full resume hard | Wrap checkpoint save/load, and test resume on day one of training work |
| Reward shaping unstable | Normalize terms, start with distance plus small action penalty, add terms one at a time |
| Nondeterminism breaks repeatability claims | Promise CPU repeatability only, with tolerances |
| Name conflict | Check PyPI, GitHub and trademark databases before publishing |
| Scope creep into SAC and vision | Keep them on the later list until the core benchmark passes |

### 1.11 Milestones

| Week | Outcome |
| --- | --- |
| 1 | Repo, schemas, validator, lightweight env, start MuJoCo arm |
| 2 | MuJoCo reach env, rewards, safety, PPO training, resume |
| 3 | Evaluation, report, SDK, benchmark vs a simple controller |
| 4 | CLI polish, docs, examples, packaging, reproducible benchmark |

### 1.12 Open questions

- Wrap Stable-Baselines3 directly, or write a thin custom PPO for full state control?
- Which real arm (if any) is the first integration target?
- Final project name and license check outcome.
- Which action mode is the default for the reaching benchmark: joint position targets or joint deltas?

## 2. Proof of Concept (PoC)

### 2.1 PoC goal

Prove one thing end to end: a single command trains a PPO policy on a simulated arm reaching task, saves a resumable checkpoint, evaluates it on held-out seeds, and prints a report. Everything else is out of scope for the PoC.

### 2.2 PoC scope

**In scope**

- One MuJoCo arm (3 to 6 joints) defined in a small MJCF file in the repo.
- `robotarm/reach-v0`: random target in a reachable workspace, joint-delta or position-target actions.
- PPO through Stable-Baselines3, config-driven.
- Checkpoint save and resume.
- Evaluation over 100 seeds with success rate and error statistics.
- Minimal safety clipping and a test that proves limits hold.
- Minimal CLI: `train` and `eval`.

**Out of scope**

SAC, behavior cloning, ONNX, HTTP service, image observations, real hardware, enterprise features.

### 2.3 Recommended stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.11 |
| Env API | Gymnasium |
| Physics | MuJoCo Python bindings |
| RL | Stable-Baselines3 (PyTorch) |
| Config and schemas | Pydantic v2 and YAML |
| CLI | Typer |
| Tests | pytest |
| Lint and types | ruff, mypy |
| Packaging | pyproject.toml |

### 2.4 Repository layout

```
frontier-rl/
├── pyproject.toml
├── README.md
├── LICENSE
├── configs/
│   └── ppo_reach.yaml
├── src/frontier_rl/
│   ├── data/          # schemas, validator, splits
│   ├── envs/          # base env, tracking env, mujoco reach env, assets/
│   ├── rewards/       # composable reward terms
│   ├── safety/        # limits, action validation
│   ├── algorithms/    # PPO wrapper, checkpointing
│   ├── evaluation/    # metrics, report
│   ├── policies/      # RLPolicy SDK
│   └── cli.py
├── tests/
└── examples/
```

### 2.5 PoC acceptance checklist

1. `pip install -e .` succeeds in a clean environment.
2. `pytest` passes, including env checks and safety tests.
3. `frontier-rl train --config configs/ppo_reach.yaml` runs and writes `runs/<id>/` with logs and checkpoints.
4. Killing training and restarting with `--resume` continues from the last checkpoint.
5. `frontier-rl eval --checkpoint ... --episodes 100` writes `reports/evaluation.json`.
6. A random policy and a simple proportional controller are evaluated for comparison.
7. Results and versions are recorded in the run directory.

## 3. Step-by-step build prompts

Use these with Claude Code (or any coding agent) in an empty repository, in order. Do not move to the next prompt until the checks for the current one pass. Paste Prompt 0 once into a `CLAUDE.md` file so every later session keeps the context.

### Prompt 0 — Project context (paste into CLAUDE.md)

```
You are helping me build "Frontier RL" (working name), an open-core Python library for reinforcement learning on robot arms. Apache-2.0 core.

Rules for the whole project:
- Python 3.11, src layout, package name frontier_rl, pyproject.toml.
- Gymnasium API for environments (reset/step, terminated vs truncated).
- MuJoCo Python bindings for physics. Do not call a math-only trajectory replay a physical simulation.
- Wrap Stable-Baselines3 for PPO. Do not reimplement PPO.
- Pydantic v2 for configs and schemas. Typer for the CLI. pytest for tests.
- Safety limits (joint, workspace, action bounds) are enforced outside the learned policy and independent of reward.
- Must run on CPU; GPU is optional (RTX 4050, 6 GB VRAM target). Single environment, small MLP.
- Never claim accuracy, latency or success numbers that were not measured. Print measured values only.
- Keep every module small, typed, tested and documented with docstrings.
- After each task: run pytest and ruff, summarize what changed, list anything unfinished.
```

### Prompt 1 — Scaffold the repository

```
Create the repository skeleton for frontier_rl exactly as follows: pyproject.toml (dependencies: gymnasium, mujoco, numpy, torch, stable-baselines3, pydantic>=2, pyyaml, typer, pandas; dev: pytest, ruff, mypy), src/frontier_rl with subpackages data, envs, rewards, safety, algorithms, evaluation, policies, a cli.py with a Typer app exposing an empty `frontier-rl` command group, tests/, configs/, examples/, README.md, an Apache-2.0 LICENSE, and a .gitignore (runs/, __pycache__, .venv).
Add a smoke test that imports the package and calls `frontier-rl --help`.
Checks: `pip install -e .[dev]`, `pytest`, `ruff check .` all pass.
```

### Prompt 2 — Data schemas and validator

```
Implement frontier_rl.data. Use Pydantic models for two versioned schemas (schema_version "1.0"): (A) trajectory-only (robot_id, coordinate_frame, position_unit, angle_unit, trajectories with episode_id, timestamp_ms, eef_position, eef_velocity, joint_positions) and (B) state-action demonstrations (robot_id, action_mode, episodes with steps containing timestamp_ms, state{joint_positions, joint_velocities}, action, terminated).
Write a validator that checks: schema version, consistent dimensions, finite values (no NaN/inf), strictly increasing timestamps within an episode, known units and frames, non-empty episodes. Return a structured report listing every problem with episode id and index instead of stopping at the first error. Never mutate input data.
Add an episode-level train/val/test split function (seeded, never splits inside an episode).
Add the CLI command `frontier-rl data validate PATH` returning a non-zero exit code on failure.
Write tests with valid and deliberately broken fixtures.
```

### Prompt 3 — Base environment and lightweight tracking env

```
Implement frontier_rl.envs.base.RobotArmEnv, a gymnasium.Env subclass with: named structured observations (joint_positions, joint_velocities, eef_position, target, previous_action) flattened only in the returned Box; configurable action_mode (joint_position_target, joint_delta, eef_delta) with explicit scale, control_dt and saturation; correct terminated (success or unrecoverable failure) versus truncated (time limit); an info dict with distance, success and constraint-violation flags.
Then implement a lightweight math-only environment `robotarm/track-v0` that follows a reference trajectory with simple kinematics. Document clearly that it is not physical simulation.
Register both with gymnasium. Tests: gymnasium.utils.env_checker.check_env passes, seeding is deterministic, spaces have the expected shapes, truncation fires at max_episode_steps.
```

### Prompt 4 — MuJoCo reaching environment

```
Create a small MJCF model of a 3-joint arm (self-authored, no third-party assets) in src/frontier_rl/envs/assets/arm3.xml with position actuators, joint limits, and a site on the end effector. Implement `robotarm/reach-v0` on top of RobotArmEnv: sample a reachable target each reset (seeded), step the simulation with frame_skip derived from control_dt, expose joint positions and velocities, eef position and target. Success means eef within a configurable tolerance of the target.
Support joint_delta and joint_position_target action modes.
Tests: check_env passes, reset is deterministic for a given seed, a zero action does not produce NaNs, random rollouts of 1000 steps never crash, and eef position matches the MuJoCo site position.
```

### Prompt 5 — Rewards and safety layer

```
Implement frontier_rl.rewards as composable terms (distance, velocity error, smoothness/jerk, action magnitude, success bonus, failure penalty), each with a configurable weight and documented scale, combined by a RewardComposer built from config. Log each term separately in info.
Implement frontier_rl.safety: a SafetyLayer that clips actions to configured bounds, rejects NaN/inf actions, enforces joint position and velocity limits and workspace bounds, and records every intervention in info. The environment must call the safety layer independently of the reward.
Tests: reward terms are finite and scaled as documented, safety clips and counts violations, NaN actions raise or are replaced by a safe fallback (configurable), and no reward setting can disable the safety layer.
```

### Prompt 6 — PPO training with resumable checkpoints

```
Implement frontier_rl.algorithms.ppo as a thin wrapper around stable-baselines3 PPO. Load a Pydantic config from YAML (seed, environment, policy hidden_dims [256,256], algorithm params, training total_timesteps/evaluation_interval/checkpoint_interval/device). Create configs/ppo_reach.yaml with the defaults: lr 3e-4, gamma 0.99, gae_lambda 0.95, clip 0.2, epochs 4, minibatch 64, rollout 2048.
A run creates runs/<run_id>/ with config.yaml, environment.json, versions.json (python, torch, mujoco, sb3, gymnasium), logs/training.csv, checkpoints/latest and best.
Checkpoints must include model, optimizer state, num_timesteps, and RNG states (python, numpy, torch), written atomically. Add `--resume RUN_DIR` that continues training from latest.
Add CLI: `frontier-rl train --config ... --env robotarm/reach-v0 [--resume DIR]`.
Tests: a 2000-step training smoke test, then a resume test that proves num_timesteps continues rather than restarts.
```

### Prompt 7 — Evaluation and report

```
Implement frontier_rl.evaluation: run a checkpoint for N episodes on held-out seeds distinct from training seeds, deterministic action. Compute success rate, mean and p95 end-effector position error (metres), mean episode return, constraint violation count, and action saturation rate. Write reports/evaluation.json including config hash, checkpoint path, seeds and package versions.
Implement two baselines under identical conditions: a random policy and a proportional joint controller toward the target.
Add CLI: `frontier-rl eval --checkpoint ... --episodes 100`, printing a compact table comparing the policy against the baselines.
Tests: metrics on a scripted fake rollout match hand-computed values.
```

### Prompt 8 — Policy SDK

```
Implement frontier_rl.policies.RLPolicy with load(path, device="auto"), predict(observation, deterministic=True), reset(), metadata(). The loader validates schema version, observation and action specs and normalization metadata and raises clear errors on mismatch. predict accepts either a named-observation dict or a flat array and returns the proposed action only; document that a separate adapter must validate and translate it for a controller.
Tests: load a checkpoint produced by the smoke training run, predictions are finite and shape-correct, a corrupted checkpoint raises a descriptive error. Add examples/run_policy.py.
```

### Prompt 9 — Run the benchmark and measure

```
Write scripts/benchmark.py that trains the reference reaching task from configs/ppo_reach.yaml with seed 42, evaluates over 100 held-out episodes, compares with the random and proportional baselines, and measures p50/p95 inference latency (batch size 1, CPU, and GPU if available) including hardware info. Save everything under runs/ and print a markdown summary table.
Do not claim any target. If success rate is low, report it and list the next three reward or config changes to try.
```

### Prompt 10 — Stretch: behavior cloning, ONNX, serve

```
Only after Prompt 9 passes. Add (a) behavior cloning on state-action demonstration files using MSE loss with episode-level splits and train-set normalization stats, usable to initialize the PPO actor; (b) `frontier-rl export --format onnx` with a test comparing native and ONNX outputs within 1e-5; (c) an optional FastAPI `frontier-rl serve` exposing POST /v1/policies/{policy_id}/predict and GET /health, rejecting malformed or non-finite inputs. The server must never send hardware commands.
```

### Prompt 11 — Docs, CI, release prep

```
Write the README quickstart (install, validate data, train, resume, eval), architecture diagram in Mermaid, a CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md and a versioning policy. Add GitHub Actions running ruff, mypy and pytest on Python 3.10-3.12. Generate a THIRD_PARTY_LICENSES summary from installed dependencies and flag anything that is not permissive. Add a note that safety clipping alone cannot guarantee robot safety.
```

## 4. How to use this plan

1. Create the repo and paste Prompt 0 into `CLAUDE.md`.
2. Run Prompts 1 to 9 in order. Review each diff and run the checks yourself.
3. Stop at Prompt 9: that is the PoC. Decide on SAC, hardware and enterprise features only after you have measured baseline results.
4. Check the name and licenses before any public release.
