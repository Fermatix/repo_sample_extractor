## Repository Overview

A MuJoCo-based robot arm simulation and control system. The project simulates a dual-arm robot (leader/follower configuration) with support for inverse kinematics, reinforcement learning (SAC), and AI-agent-driven control via LLM APIs. Written entirely in Python, using `mujoco` for physics simulation, `torch` for RL/neural network components, and standard library threading for concurrency. Build system: `pixi`/`pyproject.toml`.

## Repository Structure

```
.
├── reach_env.py                    # Gymnasium-style RL environment wrapper
├── test_model.py                   # SAC model inference test
├── pyproject.toml                  # Project metadata & dependencies
├── AGENTS.md / agent_memory*.md    # AI agent interaction logs (not code)
├── src/
│   ├── configs/
│   │   ├── so100.py                # Configuration classes
│   │   └── main_follower.json      # JSON configuration data
│   └── so100_mujoco_sim/           # Main library package
│       ├── simulate.py             # Core simulation loop
│       ├── arm_control.py          # Arm state & control logic
│       ├── agent_service.py        # LLM-based agent service API
│       ├── ik_core.py              # Inverse kinematics solver
│       ├── update_thread.py        # Async status update thread
│       ├── mujoco_viewport.py      # MuJoCo viewport wrapper
│       └── xml/sim_scene.xml       # MuJoCo scene definition
├── scripts/
│   ├── gemini_er_control.py        # Gemini API-based control script
│   ├── real_agent_service.py       # Real robot agent service script
│   ├── real_camera.py              # Camera capture for real robot
│   └── download.py                 # HuggingFace model download utility
└── tests/
    ├── test_agent_service.py       # Agent service unit tests
    └── test_arm_control.py         # Arm control tests (minimal)
```

## What Was Sampled

- **17 files** saved, covering ~3704 LOC (trimmed of trailing blanks).
- **Business logic (7 files):** `simulate.py`, `arm_control.py`, `reach_env.py`, `mujoco_viewport.py`, `scripts/real_agent_service.py`, `scripts/real_camera.py`, `scripts/gemini_er_control.py`
- **API/service layer (2 files):** `agent_service.py`, `scripts/gemini_er_control.py`
- **Utility (2 files):** `ik_core.py`, `download.py`
- **Infrastructure (4 files):** `update_thread.py`, `src/configs/so100.py`, `src/configs/main_follower.json`, `src/so100_mujoco_sim/xml/sim_scene.xml`
- **Tests (3 files):** `test_agent_service.py`, `test_arm_control.py`, `test_model.py`
- **Test-to-prod ratio:** ~7% — the test suite is thin, reflecting the early-stage nature of the project.

## Notes

- **Small single-project repo.** The entire codebase is ~4820 Python LOC total. All substantive source files were sampled. The remaining LOC gap (vs. the 5000 LOC target) is because `save_sample` strips trailing blank lines, and the rest of the files are either empty (`__init__.py`, `run_sac_control.py`) or non-code (markdown docs, binary model files, lock file).
- **Qualitative observations:** The codebase is functional but shows signs of rapid prototyping — mixed patterns (dataclasses, callbacks, global state), minimal error handling in spots, some dead/commented-out code, and thin test coverage. The project has 2 contributors (one with ~69 commits, another with ~9).
- **Skipped:** Empty `__init__.py` files, `run_sac_control.py` (0 LOC), binary files (model zip, screenshots), markdown docs, `pixi.lock`, `.env.example`, `.gitignore`, `.gitattributes`.
- **No PII, secrets, or organizational identifiers** were found in the sampled files.