# 🎮 STRANGER THINGS XODIA - PISB Credenz 2026

## Overview

Train 3 separate RL agents to navigate Stranger Things universes. Each universe requires different strategies and has its own agent with separate memory.

---

## 🌌 The Three Universes

### 👧 Universe 1: Find Will
**Mission**: Eleven must rescue Will from the Upside Down

**Environment**:
- Grid: 20x20
- Start: (2, 2) bottom-left
- Goal: Will at (17, 12)
- Enemies: 6 Demogorgons (wandering patrol)

**Observation** (6 values):
```
[agent_x, agent_y, will_x, will_y, nearest_demo_x, nearest_demo_y]
```

**Rewards**:
- ✅ +500: Found Will (distance < 1.5)
- ❌ -100: Demogorgon collision (distance < 1.8)
- ⚠️ -0.1 × distance_to_will: Encourages progress toward Will
- ⏱️ -0.5: Per step

**Max Steps**: 300

---

### 🔬 Universe 2: Escape Room
**Mission**: Escape Hawkins Lab while avoiding Russian guards

**Environment**:
- Grid: 20x20
- Start: (2, 18) top-left
- Exit: (18.5, 2.5) bottom-right
- Enemies: 5 Russian guards (patrol routes)

**Observation** (6 values):
```
[agent_x, agent_y, exit_x, exit_y, nearest_guard_x, nearest_guard_y]
```

**Rewards** (Complex):
- ✅ +600: Reached exit (distance < 1.5)
- ✅ Progressive shaping for approaching exit:
  - If distance < 2.0: +(2.0 - distance) × 50
  - If distance < 5.0: +(5.0 - distance) × 10
  - Otherwise: -distance × 0.3
- ❌ Guard detection (distance < 4.5):
  - Penalty = -100 × (4.5 - distance) / 4.5 (scales with proximity)
  - If distance < 1.5: Additional -150 (capture penalty)
- ⏱️ -0.8 per step (BUT +0.8 if within 12.0 units of exit, effectively 0 cost)

**Max Steps**: 300

**Note**: Guards have a "flashlight" detection range of 4.5 units with scaling penalties.

---

### ⚡ Universe 3: Fight Vecna
**Mission**: Rescue 2 children while avoiding Vecna and projectiles

**Environment**:
- Grid: 20x20
- Start: (10, 2) bottom-center
- Targets: 2 children at (3, 10) and (17, 10)
- Enemies: Vecna + 4 Demogorgons + projectiles

**Observation** (12 values):
```
[agent_x, agent_y,
 child1_x, child1_y,
 child2_x, child2_y,
 vecna_x, vecna_y,
 child1_rescued (0/1),
 child2_rescued (0/1),
 projectile_x, projectile_y]
```

**Rewards**:
- ✅ +200: Each child rescued
- ✅ +300: Both children rescued (WIN!)
- ❌ -200: Too close to Vecna (distance < 2.5, instant death)
- ❌ -150: Hit by projectile (distance < 0.8, instant death)
- ❌ -80: Demogorgon collision (distance < 1.5)
- ⚠️ Progress penalties for unrescued children:
  - If child 1 not rescued: -0.05 × distance_to_child1
  - If child 2 not rescued: -0.05 × distance_to_child2
- ⏱️ -1.0: Per step

**Max Steps**: 300

**Note**: Vecna fires projectiles periodically that track the agent.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy pillow opencv-python
```

### 2. Create Your Solution
Edit `submission_template.py`:
- Implement 3 discretization functions
- Implement 3 agent creation functions
- Tune hyperparameters

### 3. Test Your Solution
```bash
# Quick test (no visualization)
python competition_runner.py submission_template.py

# Full test with visualization
python competition_runner.py submission_template.py --visualize

# Custom episode count
python competition_runner.py submission_template.py --episodes 500 --visualize
```

---

## 📊 Scoring

**Total Score = Universe1_best + Universe2_best + Universe3_best**

Example:
```
Universe 1 (Find Will):    +450.2
Universe 2 (Escape Room):  +520.8
Universe 3 (Fight Vecna):  +650.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL SCORE:              1621.1  ← Your ranking!
```

---

## 📁 Required Files

Your repo should have:
```
xodia-competition/
├── realm_engine.py           # Game engine
├── competition_runner.py     # Testing tool
├── submission_template.py    # Template to edit
└── assets/                   # Game graphics
    ├── univ1/ (4 files)
    ├── univ2/ (3 files)
    └── univ3/ (8 files)
```

---

## 📤 Submission

Submit ONLY your completed Python file (e.g., `team_solution.py`)

---


## 🎓 Q-Learning Tips

**Discretization**

**Hyperparameters**

**Exploration**

---

Good luck! 🏆
