# LevDoom Seek and Slay — Judge Workflow

This repository contains the automated evaluation system for the LevDoom Seek and Slay competition. When a student pushes to their submission repo, GitHub Actions runs this judge and submits the result to the leaderboard.


## Repository overview

```mermaid
flowchart LR
    subgraph student-repo["Student Repo (per student)"]
        SA[student_agent.py\nweights, etc.]
        SW[.github/workflows\nmain.yml]
    end

    subgraph judge-repo["hw3-2-judge-workflow"]
        JW[evaluate.yml\nreusable workflow]
        JP[judge.py\nevaluation logic]
    end

    subgraph leaderboard-repo["hw3-2-leaderboard"]
        LW[.github/workflows\nsubmit.yml\n]
        LP[GitHub Pages\nleaderboard site]
    end

    SW -- "workflow_call" --> JW
    JW -- "checkout + run" --> JP
    JW -- "POST evaluation results" --> LW
    LW -- "updates" --> LP
```


## How evaluation works

1. The student's repo is checked out to `./student/`.
2. Dependencies are installed from `student/requirements.txt`.
3. `judge.py` loads `student/student_agent.py` and evaluates the agent across 5 levels.
4. For each level, the agent is run on 5 fixed seeds. A **fresh agent instance is created at the start of every episode**.
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
    secrets: inherit
```

`secrets: inherit` passes the necessary secrets (`LEADERBOARD_TOKEN`, `SUBMIT_SECRET`) from the judge workflow to your job. You do not need to configure any secrets in your own repo.

### Implementing `student_agent.py`

Define a class named `StudentAgent` with `__init__(self, action_space)` and `act(self, obs)`.

```python
class StudentAgent:
    def __init__(self, action_space):
        # Called once at the start of every episode.
        # action_space.n  → number of discrete actions
        # action_space.sample() → random action
        self.action_space = action_space

    def act(self, obs) -> int:
        # Called at every timestep. Return an integer action.
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
