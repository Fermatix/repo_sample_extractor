#!/usr/bin/env python
"""Drive the SO-100 MuJoCo sim with Google Gemini Robotics-ER 1.6.

This launches the existing `agent_service.py` JSON sim as a subprocess and lets
Gemini Robotics-ER 1.6 control the arm through low-level function-calling tools
(observe / move_gripper_to / open_gripper / close_gripper / reset_scene / done).

Gemini is given NO prior knowledge of the cube positions, the coordinate axes, or
any grasp strategy -- it must discover all of that from the camera images and the
numeric end-effector feedback, the same way a human (or Claude) would by probing.

Usage:
    # put GEMINI_API_KEY=... in a .env file at the repo root, then:
    pixi run agent-gemini
    # or:
    .pixi/envs/default/bin/python scripts/gemini_er_control.py [--max-steps N]

Requires: google-genai (added via `pixi add --pypi google-genai`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PYTHON = PROJECT_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
AGENT_SERVICE = PROJECT_ROOT / "src" / "so100_mujoco_sim" / "agent_service.py"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-1.6-preview")
CAMERA_ORDER = ("top", "front", "left", "gripper")


# --------------------------------------------------------------------------- #
# .env loading (tiny manual parser -- avoids an extra dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# --------------------------------------------------------------------------- #
# Sim subprocess wrapper
# --------------------------------------------------------------------------- #
class SimClient:
    """Launches agent_service.py and talks to it over newline-delimited JSON."""

    def __init__(self, session_dir: Path, memory_path: Path) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._id = 0
        self.log_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.proc = subprocess.Popen(
            [
                str(ENV_PYTHON),
                str(AGENT_SERVICE),
                "--session-dir", str(session_dir.resolve()),
                "--memory", str(memory_path.resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    def _stderr_tail(self, n: int = 20) -> str:
        return "\n".join(self.stderr_lines[-n:])

    def wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "sim exited during startup. stderr:\n" + self._stderr_tail()
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.log_lines.append(line)  # e.g. a stray banner line
                continue
            if msg.get("cmd") == "ready":
                return
            self.log_lines.append(line)
        raise TimeoutError("timed out waiting for sim 'ready'")

    def send(self, cmd: dict) -> dict:
        self._id += 1
        cmd = dict(cmd, id=self._id)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "sim closed stdout (crashed?). stderr:\n" + self._stderr_tail()
                )
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                self.log_lines.append(line)
                continue
            if resp.get("id") == self._id:
                return resp
            self.log_lines.append(line)  # ready / unmatched

    # convenience wrappers -------------------------------------------------- #
    def reset(self) -> dict:
        return self.send({"cmd": "reset"})

    def observe(self) -> dict:
        return self.send({"cmd": "observe"})

    def move_gripper_to(self, x: float, y: float, z: float, steps: int = 60) -> dict:
        return self.send(
            {"cmd": "move_gripper_to", "position": [x, y, z], "steps": steps}
        )

    def act_jaw(self, value: float, steps: int = 40) -> dict:
        return self.send(
            {
                "cmd": "act",
                "mode": "absolute",
                "joints": {"Jaw": value},
                "max_delta": 0.85,
                "steps": steps,
            }
        )

    def close(self) -> None:
        try:
            self.send({"cmd": "close"})
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------- #
# Tool layer (maps Gemini function calls to sim commands)
# --------------------------------------------------------------------------- #
JAW_OPEN = 0.85
JAW_CLOSED = 0.1

FUNCTION_DECLARATIONS = [
    {
        "name": "observe",
        "description": (
            "Capture the current camera views (top, front, left, gripper) and "
            "report the current end-effector position [x, y, z] in world meters, "
            "the joint angles, and the gripper jaw opening. Returns the four "
            "images so you can see the scene."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "move_gripper_to",
        "description": (
            "Move the gripper/end-effector to a world-space position given in "
            "meters, using inverse kinematics. This does NOT change the gripper "
            "(jaw) opening. Reports whether the target was reached, the position "
            "error, the resulting end-effector position, and returns fresh camera "
            "views."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "world X in meters"},
                "y": {"type": "number", "description": "world Y in meters"},
                "z": {"type": "number", "description": "world Z in meters"},
                "steps": {
                    "type": "integer",
                    "description": "motion interpolation steps (default 60, range 1-1000)",
                },
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "open_gripper",
        "description": (
            "Open the gripper jaw. Returns the resulting jaw opening and fresh "
            "camera views."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "close_gripper",
        "description": (
            "Close the gripper jaw (to grasp). Returns the resulting jaw opening "
            "and fresh camera views."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "reset_scene",
        "description": (
            "Reset the entire scene back to its starting layout. Use only to "
            "recover from a state you cannot otherwise fix. Returns fresh views."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "done",
        "description": (
            "Call this when you have finished: either the red cube is stacked on "
            "the green cube (success=true), or you are giving up (success=false)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string", "description": "short summary of the outcome"},
            },
            "required": ["success", "summary"],
        },
    },
]


class ToolExecutor:
    def __init__(self, sim: SimClient) -> None:
        self.sim = sim

    @staticmethod
    def _feedback(observation: dict) -> dict:
        ee = observation["end_effector_pose"]["position"]
        joints = observation["joint_positions"]
        return {
            "end_effector_position": [round(v, 4) for v in ee],
            "jaw_opening": round(joints.get("Jaw", 0.0), 4),
            "joint_angles": {k: round(v, 4) for k, v in joints.items()},
            "step_count": observation.get("step_count"),
        }

    def execute(self, name: str, args: dict):
        """Returns (result_dict, screenshots_or_None)."""
        if name == "observe":
            obs = self.sim.observe()["observation"]
            return self._feedback(obs), obs.get("screenshots")

        if name == "move_gripper_to":
            x, y, z = float(args["x"]), float(args["y"]), float(args["z"])
            steps = int(args.get("steps", 60))
            steps = max(1, min(1000, steps))
            r = self.sim.move_gripper_to(x, y, z, steps)
            if not r.get("ok"):
                return {"ok": False, "error": r.get("error", "move failed")}, None
            obs = self.sim.observe()["observation"]
            fb = self._feedback(obs)
            fb.update(
                {
                    "ok": True,
                    "requested_target": [x, y, z],
                    "reached": r.get("reached"),
                    "position_error": round(float(r.get("position_error", 0.0)), 4),
                }
            )
            return fb, obs.get("screenshots")

        if name == "open_gripper":
            self.sim.act_jaw(JAW_OPEN)
            obs = self.sim.observe()["observation"]
            return self._feedback(obs), obs.get("screenshots")

        if name == "close_gripper":
            self.sim.act_jaw(JAW_CLOSED, steps=50)
            obs = self.sim.observe()["observation"]
            return self._feedback(obs), obs.get("screenshots")

        if name == "reset_scene":
            r = self.sim.reset()
            obs = r["observation"]
            return self._feedback(obs), obs.get("screenshots")

        return {"ok": False, "error": f"unknown tool '{name}'"}, None


# --------------------------------------------------------------------------- #
# Gemini agent
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are an autonomous controller for a 5-DOF SO-100 robot arm fitted with a \
parallel-jaw gripper, inside a MuJoCo physics simulation. On the table in front \
of the arm there are three cubes: one red, one green, and one blue.

YOUR TASK: pick up the RED cube and place it on top of the GREEN cube, so the red \
cube ends up resting stably on top of the green cube. Do not knock over or push \
away the blue cube. When the red cube is stacked on the green cube, call \
done(success=true, ...). If you become unable to make progress, call \
done(success=false, ...).

You can only affect the world through these tools:
- observe(): four camera views (top, front, left, gripper-mounted) plus the \
current end-effector position [x, y, z] in world meters, the joint angles, and \
the gripper jaw opening value.
- move_gripper_to(x, y, z): move the gripper to a world-space position in meters \
via inverse kinematics. It does not open or close the jaw. It reports whether it \
reached the target and the resulting end-effector position.
- open_gripper() / close_gripper(): open or close the jaw.
- reset_scene(): reset everything (recovery only).
- done(success, summary): finish.

Important: you are NOT told where the cubes are, which direction each coordinate \
axis points, or how high to grasp. Discover all of this yourself. The end-effector \
position is reported in the same world meters frame that move_gripper_to expects, \
so a reliable way to calibrate is to make a small deliberate test move with \
move_gripper_to and compare the before/after end-effector position and camera \
views -- that tells you which axis is left/right, which is forward/back, which is \
up/down, and roughly where each cube sits. Then plan the pick-and-place.

CRITICAL -- LOOK AT THE SCREENSHOTS AFTER EVERY ACTION. Every tool returns four \
fresh camera views (top, front, left, gripper). After each and every action you \
MUST actually examine all four images before you do anything else, and in your \
reasoning explicitly describe what each view now shows: where the gripper is, \
where the red/green/blue cubes are, whether the gripper is holding a cube, the \
gaps between the fingers and the cube, and relative heights. Do NOT decide your \
next move from the numeric end-effector value alone, and do NOT rely on what you \
expected to happen -- read the latest images and trust them over your assumptions.

Work one small step at a time: look at the four latest images and say what you \
see, decide a single move, execute it, then look again at the four new images and \
correct based on what actually changed.

Verify with your eyes, never assume:
- After you close the gripper and lift, look at the front and left views and \
confirm the red cube has actually risen off the table and is held between the \
fingers. If it is still sitting on the table, your grasp FAILED -- do not continue \
as if you are holding it; go back and attempt the pick again, adjusting your \
approach.
- Before you ever call done(success=true), look carefully at the top, front, and \
left views and confirm the red cube is physically resting on top of the green \
cube. If the cubes are still separate on the table, the task is NOT complete -- \
keep working. Be skeptical of your own progress: only claim success when the \
images themselves clearly show the red cube stacked on the green cube.

Think out loud briefly (describing the current images) before each tool call.
"""

INITIAL_USER_TEXT = """\
Here are the current camera views and state of the scene. Begin the task: pick up \
the red cube and stack it on top of the green cube. Start by observing/probing to \
understand the layout and the coordinate frame, then act step by step."""

RETRY_STATUS_HINTS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")


CAMERA_LABEL_SET = {f"[{view} camera]" for view in CAMERA_ORDER}


def image_parts(screenshots: dict | None) -> list:
    parts: list = []
    if not screenshots:
        return parts
    for view in CAMERA_ORDER:
        path = screenshots.get(view)
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            parts.append(types.Part(text=f"[{view} camera]"))
            parts.append(types.Part.from_bytes(data=data, mime_type="image/png"))
    return parts


def prune_old_images(contents: list, keep_last: int = 2) -> None:
    """Keep image bytes only for the most recent `keep_last` image-bearing turns.

    Older camera images (and their labels) are replaced by a tiny placeholder so
    the model's vision stays focused on the current state and the context/token
    cost does not balloon. Full *text* history (reasoning + numeric feedback) is
    retained. Only user turns carry images; model turns (with thought signatures)
    are never touched.
    """
    img_idxs = [
        i
        for i, c in enumerate(contents)
        if any(getattr(p, "inline_data", None) is not None for p in (c.parts or []))
    ]
    keep = set(img_idxs[-keep_last:]) if keep_last > 0 else set()
    for i in img_idxs:
        if i in keep:
            continue
        c = contents[i]
        kept_parts = []
        dropped = 0
        for p in c.parts or []:
            if getattr(p, "inline_data", None) is not None:
                dropped += 1
                continue
            if getattr(p, "text", None) in CAMERA_LABEL_SET:
                continue
            kept_parts.append(p)
        if dropped:
            kept_parts.append(types.Part(text=f"[{dropped} earlier camera images omitted to save context]"))
        contents[i] = types.Content(role=c.role, parts=kept_parts)


def generate_with_retry(client, model, contents, config, max_retries=6, min_interval=0.0, _last=[0.0]):
    # crude pacing for free-tier RPM limits
    if min_interval > 0:
        wait = min_interval - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            _last[0] = time.time()
            return resp
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            retriable = any(h in msg for h in RETRY_STATUS_HINTS)
            if not retriable or attempt == max_retries:
                raise
            print(f"  [retry] API error ({msg[:120]}...); sleeping {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def run(max_steps: int, min_interval: float) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: no GEMINI_API_KEY found. Put `GEMINI_API_KEY=...` in a .env file "
            "at the repo root (see .env.example) or export it in your shell.",
            file=sys.stderr,
        )
        return 2

    session_dir = PROJECT_ROOT / "agent_runs" / "gemini-task" / "sim"
    memory_path = PROJECT_ROOT / "agent_runs" / "gemini-task" / "gemini_memory.md"
    transcript_path = PROJECT_ROOT / "agent_runs" / "gemini-task" / "transcript.json"

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=FUNCTION_DECLARATIONS)],
        temperature=1.0,
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        # Force a real structured function call every turn (the model otherwise
        # sometimes writes its action as ```python text and wastes the turn).
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )

    print(f"Launching sim ({ENV_PYTHON.name}) ...", flush=True)
    sim = SimClient(session_dir, memory_path)
    transcript: list[dict] = []
    success = False
    summary = ""
    try:
        sim.wait_ready()
        print("Sim ready. Resetting scene.", flush=True)
        reset_obs = sim.reset()["observation"]
        executor = ToolExecutor(sim)
        init_feedback = ToolExecutor._feedback(reset_obs)
        print(f"Initial end-effector position: {init_feedback['end_effector_position']}", flush=True)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=INITIAL_USER_TEXT),
                    types.Part(text="Current state: " + json.dumps(init_feedback)),
                    *image_parts(reset_obs.get("screenshots")),
                ],
            )
        ]

        for step in range(1, max_steps + 1):
            print(f"\n===== step {step}/{max_steps} =====", flush=True)
            prune_old_images(contents, keep_last=2)
            resp = generate_with_retry(
                client, DEFAULT_MODEL, contents, config, min_interval=min_interval
            )
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None:
                print("  (no candidate returned; nudging)", flush=True)
                contents.append(types.Content(role="user", parts=[types.Part(
                    text="No response received. Please continue using the tools.")]))
                continue

            model_content = cand.content
            contents.append(model_content)
            parts = model_content.parts or []

            thought_text = "\n".join(
                p.text.strip()
                for p in parts
                if getattr(p, "thought", False) and getattr(p, "text", None)
            )
            answer_text = "\n".join(
                p.text.strip()
                for p in parts
                if not getattr(p, "thought", False) and getattr(p, "text", None)
            )
            if thought_text.strip():
                print(f"  [thinking] {thought_text.strip()}", flush=True)
            if answer_text.strip():
                print(f"GEMINI: {answer_text.strip()}", flush=True)

            function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            step_record = {
                "step": step,
                "thought": thought_text.strip(),
                "text": answer_text.strip(),
                "tool_calls": [],
            }

            if not function_calls:
                transcript.append(step_record)
                print("  (no tool call; nudging to act)", flush=True)
                contents.append(types.Content(role="user", parts=[types.Part(
                    text="Continue. Use the tools to act, or call done() when finished.")]))
                continue

            response_parts: list = []  # functionResponse parts (kept first)
            attached_images: list = []  # image parts (appended after)
            finished = False
            for fc in function_calls:
                name = fc.name
                args = dict(fc.args) if fc.args else {}
                print(f"  -> {name}({json.dumps(args)})", flush=True)

                if name == "done":
                    finished = True
                    success = bool(args.get("success", False))
                    summary = str(args.get("summary", ""))
                    step_record["tool_calls"].append({"name": name, "args": args})
                    response_parts.append(
                        types.Part.from_function_response(name=name, response={"ok": True})
                    )
                    continue

                try:
                    result, shots = executor.execute(name, args)
                except Exception as exc:  # noqa: BLE001
                    result, shots = {"ok": False, "error": str(exc)}, None
                print(f"     result: {json.dumps(result)[:300]}", flush=True)
                step_record["tool_calls"].append({"name": name, "args": args, "result": result})
                response_parts.append(types.Part.from_function_response(name=name, response=result))
                attached_images.extend(image_parts(shots))

            reply_parts = response_parts + attached_images
            transcript.append(step_record)
            transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
            contents.append(types.Content(role="user", parts=reply_parts))

            if finished:
                print(f"\nGemini called done(success={success}): {summary}", flush=True)
                break
        else:
            summary = f"Reached step cap ({max_steps}) without calling done()."
            print("\n" + summary, flush=True)

        # final look for the record
        final_obs = sim.observe()["observation"]
        transcript_path.write_text(
            json.dumps(
                {"steps": transcript, "final_state": ToolExecutor._feedback(final_obs),
                 "final_screenshots": final_obs.get("screenshots"),
                 "model_success": success, "model_summary": summary},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFinal screenshots: {final_obs.get('screenshots')}", flush=True)
        print(f"Transcript: {transcript_path}", flush=True)
    finally:
        sim.close()

    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drive the SO-100 sim with Gemini Robotics-ER 1.6.")
    p.add_argument("--max-steps", type=int, default=30, help="max agent iterations (default 30)")
    p.add_argument(
        "--min-interval",
        type=float,
        default=float(os.environ.get("GEMINI_MIN_INTERVAL", "0")),
        help="minimum seconds between model calls (free-tier pacing; default 0)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run(max_steps=args.max_steps, min_interval=args.min_interval)


if __name__ == "__main__":
    raise SystemExit(main())
