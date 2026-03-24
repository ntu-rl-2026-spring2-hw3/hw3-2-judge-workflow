# LevDoom Seek and Slay — Judge Workflow

This repository contains the automated evaluation system for the LevDoom Seek and Slay competition. When a student pushes to their submission repo, GitHub Actions runs this judge and submits the result to the leaderboard.


## Repository overview

```mermaid
flowchart LR
    subgraph student-repo["Student Repo (per student)"]
        SW[".github/workflows/main.yml"]
        SA["student_agent.py\nweights, etc."]
    end

    subgraph judge-repo["hw3-2-judge-workflow"]
        JW["evaluate.yml\n(reusable workflow)"]
        JP["judge.py"]
    end

    subgraph leaderboard-repo["hw3-2-leaderboard"]
        LW["update_leaderboard.yml\n(repository_dispatch)"]
        LP["GitHub Pages\nleaderboard site"]
    end

    SW -- "workflow_call" --> JW
    JW -- "checkout + run" --> JP
    JP -- "results.json" --> JW
    JW -- "POST /dispatches\nevent_type: submit_score" --> LW
    LW -- "updates leaderboard.json" --> LP
```


## How evaluation works

1. The student's repo is checked out to `./student/`.
2. Dependencies are installed from `student/requirements.txt`.
3. `judge.py` loads `student/student_agent.py` and instantiates `StudentAgent` **once**.
4. For each level, the agent is run on 5 fixed seeds. `reset()` is called before every episode.
5. If the agent's mean kills fall below a level's threshold, evaluation stops early.
6. Results are submitted to the leaderboard.

### Levels

| Level | Environment ID | Threshold (mean kills) |
|-------|---------------|------------------------|
| 0 | `SeekAndSlayLevel0-v0` | 22 |
| 1 | `SeekAndSlayLevel1_6-v0` | 15 |
| 2 | `SeekAndSlayLevel2_1-v0` | 9 |
| 3 | `SeekAndSlayLevel3_1-v0` | 7 |
| 4 | `SeekAndSlayLevel4-v0` | — (final level) |

### Scoring

Leaderboard ranks are determined first by **number of levels reached**, then by a weighted score as tiebreaker:

```
score = kills × 0.8 + health × 0.1 + ammo × 0.1
```

Each value is the mean across the 5 seeds for that level.


## Student submission guide

### Required files

Your repo must contain:

| File | Purpose |
|------|---------|
| `student_agent.py` | Your agent implementation (see below) |
| `requirements.txt` | Python dependencies your agent needs |
| `meta.xml` | Your student ID |
| `.github/workflows/main.yml` | Workflow that triggers evaluation on push |

### Setting up `.github/workflows/main.yml`

Create this file in your repo to automatically evaluate your agent whenever you push to `main`:

```yaml
name: Submit to Leaderboard

on:
  push:
    branches:
      - main

jobs:
  evaluate:
    uses: ntu-rl-2026-spring2-hw3/hw3-2-judge-workflow/.github/workflows/evaluate.yml@main
```

### Implementing `student_agent.py`

Define a class named `StudentAgent` with three methods:

| Method | When called | Purpose |
|--------|-------------|---------|
| `__init__(self, action_space)` | Once at startup | Load model weights, build network, etc. |
| `reset(self)` | Before every episode | Clear per-episode state (frame stack, hidden states, etc.) |
| `act(self, obs) -> int` | Every timestep | Return an integer action |

`reset()` is optional — if you have no per-episode state to clear, you can omit it.

```python
class StudentAgent:
    def __init__(self, action_space):
        # Called once. Load weights here.
        self.action_space = action_space

    def reset(self):
        # Called before each episode. Clear internal state here.
        pass

    def act(self, obs) -> int:
        # Called every timestep. Return an integer action.
        return self.action_space.sample()
```

### Loading model weights or other files

The judge does **not** run from your repo's directory, so bare relative paths will fail.
Always use `Path(__file__).parent` to reference files in your repo:

```python
from pathlib import Path
_DIR = Path(__file__).parent

class StudentAgent:
    def __init__(self, action_space):
        self.model = torch.load(_DIR / "weights.pth")  # correct
        # torch.load("weights.pth")                    # FileNotFoundError
```

### `meta.xml` format

Replace `your_student_id` with your actual student ID. This file is used by the judge to identify you when submitting results to the leaderboard.

```xml
<submission>
  <info>
    <name>r00000001</name>
  </info>
</submission>
```
