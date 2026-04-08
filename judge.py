"""
Evaluation script for LevDoom levels.

Usage:
    python judge.py [--student-path PATH] [--output PATH]

Loads the student's agent from student_agent.py and evaluates it across all
configured LevDoom levels, writing results to results.json.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
import random

import numpy as np
import levdoom
import gymnasium
from typing import Protocol


# ---------------------------------------------------------------------------
# Actor interface — students must implement this
# ---------------------------------------------------------------------------

class Actor(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...


# ---------------------------------------------------------------------------
# Model file size check
# ---------------------------------------------------------------------------

MAX_TOTAL_SIZE_MB = 51


def check_submission_size(student_path: str) -> None:
    """
    Enforce a total size limit on the student submission directory.

    Sums every file (including source code, weights, etc.).
    Raises RuntimeError if total exceeds MAX_TOTAL_SIZE_MB.
    """
    root = Path(student_path)
    total = 0
    files_found = []

    for f in root.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total += size
            files_found.append((f.relative_to(root), size))

    files_found.sort(key=lambda x: x[1], reverse=True)

    print(f"\nStudent submission files:")
    for name, size in files_found:
        print(f"  {name}: {size / 1024 / 1024:.2f} MB")
    print(f"  Total: {total / 1024 / 1024:.2f} MB (limit: {MAX_TOTAL_SIZE_MB} MB)")

    if total > MAX_TOTAL_SIZE_MB * 1024 * 1024:
        raise RuntimeError(
            f"Submission too large: {total / 1024 / 1024:.2f} MB "
            f"(limit: {MAX_TOTAL_SIZE_MB} MB)"
        )


# ---------------------------------------------------------------------------
# Student agent loader
# ---------------------------------------------------------------------------

def load_student_agent(student_path: str) -> "Actor":
    """
    Dynamically import student_agent.py and return an instantiated agent.

    The student module must expose:
      - StudentAgent(action_space)  — class with reset() and act(obs) -> int
    """
    agent_file = Path(student_path) / "student_agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"student_agent.py not found at {agent_file}")

    check_submission_size(student_path)

    sys.path.insert(0, str(Path(student_path).resolve()))

    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "StudentAgent"):
        raise ImportError("student_agent.py must define a StudentAgent class")

    try:
        env = gymnasium.make(LEVELS[0]["id"], render=False)
        agent = module.StudentAgent(env.action_space)
        env.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate student agent: {exc}") from exc

    if not callable(getattr(agent, "act", None)):
        raise TypeError("StudentAgent must implement act(obs) -> int")

    return agent


# ---------------------------------------------------------------------------
# Results serialiser
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_path: str) -> None:
    """Write evaluation results to a JSON file for the leaderboard step."""
    payload = {}
    for r in results:
        payload[r["level"]] = {
            "kills":  r["mean_kills"],
            "health": r["mean_health"],
            "ammo":   r["mean_ammo"],
        }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

LEVELS = [
    {"id": "SeekAndSlayLevel0-v0",  "map": "default",           "threshold": 18},
    {"id": "SeekAndSlayLevel1_6-v0","map": "mixed_enemies",      "threshold": 9},
    {"id": "SeekAndSlayLevel3_1-v0","map": "blue_mixed_resized", "threshold": 9},
    {"id": "SeekAndSlayLevel2_3-v0","map": "red_mixed_enemies",  "threshold": 9},
    {"id": "SeekAndSlayLevel4-v0",  "map": "complete",           "threshold": None},
]

NUM_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
FIXED_ENV_SEED = 1234   # Same scenario for all students; SEEDS only vary policy randomness.


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def seed_policy_rngs(env, actor, seed: int) -> None:
    """Seed every RNG that could affect the policy's actions.

    Called *after* actor.reset() so students cannot override these inside
    their own reset(). Covers stdlib random, numpy, torch (CPU), and the
    env's action_space (for action_space.sample()).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    env.action_space.seed(seed)
    # Seed the action_space the actor is holding (likely the throwaway
    # one from load_student_agent). Without this, actor.act() ->
    # self.action_space.sample() reads from an RNG that was never
    # re-seeded and drifts with process entropy.
    actor_space = getattr(actor, "action_space", None)
    if actor_space is not None and actor_space is not env.action_space:
        actor_space.seed(seed)


def run_episode(env, actor: "Actor", seed: int = None) -> dict:
    """Run a single episode and return the final info dict.

    The environment is pinned to FIXED_ENV_SEED so every student faces the
    same scenario. `seed` only varies policy randomness across runs.
    """
    env.unwrapped.game.set_seed(FIXED_ENV_SEED)
    obs, info = env.reset()
    actor.reset()
    if seed is not None:
        seed_policy_rngs(env, actor, seed)
    done = False
    while not done:
        action = actor.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def evaluate_level(level_id: str, actor: "Actor", seeds: list[int]) -> dict:
    """
    Evaluate one level across multiple seeds.

    Args:
        level_id:  Gym environment ID.
        actor:     Instantiated Actor to evaluate.
        seeds:     List of integer seeds to use.

    Returns:
        dict with per-seed info and aggregate stats.
    """
    per_seed = []
    for seed in seeds:
        # Warmup run to prevent counting hacking
        if random.randint(0, 2) != 0:
            env = gymnasium.make(level_id, render=False)
            _ = run_episode(env, actor, seed=seed)
            env.close()
        # Actual run
        env = gymnasium.make(level_id, render=False)
        info = run_episode(env, actor, seed=seed)
        env.close()

        per_seed.append(info)
        kills  = info.get("kills",  0)
        health = info.get("health", 0)
        ammo   = info.get("ammo",   0)
        print(f"  seed={seed}  kills={kills}  health={health}  ammo={ammo}")

    kills_list  = [ep.get("kills",  0) for ep in per_seed]
    health_list = [ep.get("health", 0) for ep in per_seed]
    ammo_list   = [ep.get("ammo",   0) for ep in per_seed]
    return {
        "level":       level_id,
        "per_seed":    per_seed,
        "mean_kills":  float(np.mean(kills_list)),
        "mean_health": float(np.mean(health_list)),
        "mean_ammo":   float(np.mean(ammo_list)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(actor: "Actor") -> list[dict]:
    """
    Evaluate all levels sequentially, stopping early if a threshold is not met.

    Args:
        actor: Instantiated Actor to evaluate.

    Returns:
        List of result dicts (one per attempted level).
    """
    results = []

    for level in LEVELS:
        level_id  = level["id"]
        threshold = level["threshold"]

        print(f"\n{'='*60}")
        print(f"Evaluating: {level_id}  (map={level['map']}, threshold={threshold})")
        print(f"{'='*60}")

        result = evaluate_level(level_id, actor, SEEDS)
        results.append(result)

        mean_kills = result["mean_kills"]
        print(f"\n  -> Mean kills: {mean_kills:.2f}")

        if threshold is not None and mean_kills < threshold:
            print(f"  !! Below threshold ({threshold}). Stopping evaluation.")
            break
        elif threshold is not None:
            print(f"  ✓  Threshold met ({mean_kills:.2f} >= {threshold}). Proceeding.")
        else:
            print(f"  ✓  Final level complete.")

    print(f"\n{'='*60}")
    print("Results summary:")
    for r in results:
        print(f"  {r['level']}: kills={r['mean_kills']:.2f}  health={r['mean_health']:.2f}  ammo={r['mean_ammo']:.2f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a student LevDoom agent.")
    parser.add_argument("--student-path", default=".", help="Directory containing student_agent.py")
    parser.add_argument("--output", default="results.json", help="Path to write results JSON")
    args = parser.parse_args()

    actor = load_student_agent(args.student_path)
    results = run_eval(actor)
    save_results(results, args.output)
