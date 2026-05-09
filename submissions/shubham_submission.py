"""
STRANGER THINGS XODIA - COMPLETED SUBMISSION
============================================
Team: [YOUR TEAM NAME]

Universe 1 (Find Will)   → submission_template_2.py  [Reactive Q-Learning]
Universe 2 (Escape Room) → my_agent-1.py             [Tabular Q-Learning, tiny state space]
Universe 3 (Fight Vecna) → submission_template_2.py  [Reactive Q-Learning]
"""

import numpy as np
import random
import math
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_info():
    return {
        "team_name": "Stranger Q-Learners",
        "algorithm": "Hybrid Q-Learning",
        "U1_source": "submission_template_2 — Reactive (8-dir angle, blocking check, anti-zigzag)",
        "U2_source": "my_agent-1 — Tabular convergence-first (324 states, optimistic init +10)",
        "U3_source": "submission_template_2 — Reactive (mission mode, Vecna flee)",
    }

def safe_obs(obs, n=6):
    """Guard against short/None observation vectors."""
    if obs is None:
        return [0.0] * n
    if len(obs) < n:
        obs = list(obs) + [0.0] * (n - len(obs))
    return obs

# Shared helpers used by my_agent-1 U2
def _sign3(delta, threshold=1.0):
    """3-way bucket: -1 | 0 | +1"""
    if delta >  threshold: return  1
    if delta < -threshold: return -1
    return 0

def _zone3(dist, near, far):
    """0 = danger, 1 = caution, 2 = safe"""
    if dist < near: return 0
    if dist < far:  return 1
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE 1 — FIND WILL
# Source: submission_template_2.py (Reactive Q-Learning)
#
# Rewards (README):
#   +500  found Will (dist < 1.5)
#   -100  Demogorgon hit (dist < 1.8)
#   -0.1 × dist_to_will  per step
#   -0.5  per step, -50 timeout
#
# Why reactive: 6 Demogorgons wander randomly — fixed grid states
# become stale instantly. Angle-based direction + blocking check
# handles moving threats far better.
# ─────────────────────────────────────────────────────────────────────────────

def discretize_universe1(observation):
    """
    Reactive state — focuses on relationships, not raw positions.
    State: (will_direction x8, will_proximity x3, demo_threat x3, demo_blocking x2)
    = 8 × 3 × 3 × 2 = 144 states
    """
    ax, ay = observation[0], observation[1]
    wx, wy = observation[2], observation[3]
    dx, dy = observation[4], observation[5]

    # 1. Direction to Will (8 compass octants)
    angle_to_will  = np.arctan2(wy - ay, wx - ax)
    will_direction = int((angle_to_will + np.pi) / (np.pi / 4)) % 8

    # 2. Distance to Will (3 coarse bins)
    dist_to_will = math.hypot(wx - ax, wy - ay)
    if   dist_to_will < 5.0:  will_proximity = 0
    elif dist_to_will < 12.0: will_proximity = 1
    else:                      will_proximity = 2

    # 3. Demogorgon threat (penalty triggers at <1.8, warn from 5.0)
    demo_dist = math.hypot(dx - ax, dy - ay)
    if   demo_dist < 2.5: demo_threat = 2
    elif demo_dist < 5.0: demo_threat = 1
    else:                  demo_threat = 0

    # 4. Is Demogorgon blocking path to Will? (cosine corridor check)
    agent = np.array([ax, ay])
    will  = np.array([wx, wy])
    demo  = np.array([dx, dy])
    vec_goal   = will - agent
    vec_enemy  = demo - agent
    dist_enemy = np.linalg.norm(vec_enemy)
    dist_goal  = np.linalg.norm(vec_goal)

    demo_blocking = 0
    if dist_enemy > 0 and dist_goal > 0 and dist_enemy < 4.0:
        cos_angle = np.dot(vec_goal, vec_enemy) / (dist_goal * dist_enemy)
        if cos_angle > 0.6 and dist_enemy < 3.5:
            demo_blocking = 1

    return (will_direction, will_proximity, demo_threat, demo_blocking)


def create_universe1_agent():
    """Reactive Q-learning agent for Universe 1 (from submission_template_2)."""
    Q1 = {}

    alpha     = 0.40
    gamma     = 0.90
    epsilon   = 0.85
    eps_decay = 0.997
    eps_min   = 0.10
    n_actions = 4

    ACTIONS = {
        0: np.array([1,  0]),
        1: np.array([0,  1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }

    action_history   = deque(maxlen=5)
    position_history = deque(maxlen=10)
    stuck_counter    = [0]

    def direct_action(agent, will):
        dx = will[0] - agent[0]
        dy = will[1] - agent[1]
        return (0 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 3)

    def is_path_blocked(agent, will, demo):
        vec_goal  = will - agent
        vec_enemy = demo - agent
        dist_e = np.linalg.norm(vec_enemy)
        dist_g = np.linalg.norm(vec_goal)
        if dist_e > 4.0 or dist_g == 0 or dist_e == 0:
            return False
        cos_angle = np.dot(vec_goal, vec_enemy) / (dist_g * dist_e)
        return cos_angle > 0.6 and dist_e < 3.5

    def safe_dodge(agent, will, demo):
        best_action, best_score = None, -1e9
        for a in range(4):
            next_pos  = agent + ACTIONS[a]
            score = np.linalg.norm(next_pos - demo) * 2.5 - np.linalg.norm(next_pos - will) * 1.2
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def anti_zigzag(action, agent, will):
        if not action_history:
            return action
        opposite = (action_history[-1] + 2) % 4
        if action == opposite:
            dx_g = will[0] - agent[0]
            dy_g = will[1] - agent[1]
            action = (1 if dy_g > 0 else 3) if abs(dx_g) > abs(dy_g) else (0 if dx_g > 0 else 2)
        return action

    def select_action(observation):
        nonlocal epsilon
        if observation is None or len(observation) < 6:
            return 0

        state = discretize_universe1(observation)
        if state not in Q1:
            Q1[state] = np.zeros(n_actions)

        ax, ay = observation[0], observation[1]
        agent  = np.array([ax, ay])
        will   = np.array([observation[2], observation[3]])
        demo   = np.array([observation[4], observation[5]])

        # Stuck detection — if visiting ≤3 unique cells in 8 steps, boost epsilon
        position_history.append((round(ax, 1), round(ay, 1)))
        if len(position_history) >= 8:
            if len(set(list(position_history)[-8:])) <= 3:
                stuck_counter[0] += 1
            else:
                stuck_counter[0] = max(0, stuck_counter[0] - 1)
        if stuck_counter[0] > 3:
            epsilon = min(0.9, epsilon + 0.15)
            stuck_counter[0] = 0
            action_history.clear()

        if random.random() < epsilon:
            dist_demo = np.linalg.norm(agent - demo)
            if dist_demo < 2.0:
                action = safe_dodge(agent, will, demo)
            elif not is_path_blocked(agent, will, demo):
                action = direct_action(agent, will)
            else:
                action = safe_dodge(agent, will, demo)
            action = anti_zigzag(action, agent, will)
        else:
            q_vals = Q1[state].copy()
            if action_history:
                q_vals[(action_history[-1] + 2) % 4] -= 0.2
            action = int(np.argmax(q_vals))
            action = anti_zigzag(action, agent, will)

        action_history.append(action)
        return action

    def update(state, action, reward, next_state):
        nonlocal epsilon
        if state not in Q1:      Q1[state]      = np.zeros(n_actions)
        if next_state not in Q1: Q1[next_state] = np.zeros(n_actions)

        max_next = np.max(Q1[next_state])
        # Asymmetric TD: amplify wins (+500), slightly dampen big losses
        if reward > 50:   td_target = reward + gamma * max_next * 1.3
        elif reward < -50: td_target = reward + gamma * max_next * 0.8
        else:              td_target = reward + gamma * max_next

        Q1[state][action] += alpha * (td_target - Q1[state][action])
        epsilon = max(eps_min, epsilon * eps_decay)

    return select_action, update, Q1


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE 2 — ESCAPE ROOM
# Source: my_agent-1.py (Tabular Q-Learning, convergence-first)
#
# Rewards (README):
#   +600  reached exit (dist < 1.5)
#   Progressive shaping: dist<2 → +(2-d)*50 | dist<5 → +(5-d)*10 | else → -d*0.3
#   Guard flashlight (dist < 4.5): -100*(4.5-d)/4.5
#   Guard capture   (dist < 1.5): -150 additional
#   -0.8 per step (+0.8 if within 12 of exit, so net ~0 once close)
#   -100 timeout
#
# Why tabular: tiny 324-state space guarantees ~185 visits per
# (state, action) pair in 200 episodes → reliable Q convergence.
# Optimistic init (+10) forces exploration of all actions before settling.
#
# State = (rx, ry, guard_zone, gdx, gdy) → max 3×3×4×3×3 = 324 states
#   rx, ry      : sign of (exit - agent) on each axis  {-1, 0, +1}
#   guard_zone  : 0=caught(<1.5), 1=run(<3.0), 2=flashlight(<4.5), 3=safe
#   gdx, gdy    : sign of guard direction (0,0 when safe — saves states)
# ─────────────────────────────────────────────────────────────────────────────

def discretize_universe2(obs):
    ax, ay = obs[0], obs[1]
    ex, ey = obs[2], obs[3]
    gx, gy = obs[4], obs[5]

    rx = _sign3(ex - ax, 1.0)
    ry = _sign3(ey - ay, 1.0)

    d_g = math.hypot(gx - ax, gy - ay)
    if   d_g < 1.5: guard_zone = 0
    elif d_g < 3.0: guard_zone = 1
    elif d_g < 4.5: guard_zone = 2
    else:           guard_zone = 3

    if guard_zone < 3:
        gdx = _sign3(gx - ax, 0.5)
        gdy = _sign3(gy - ay, 0.5)
    else:
        gdx, gdy = 0, 0

    return (rx, ry, guard_zone, gdx, gdy)


def create_universe2_agent():
    """Tabular Q-learning agent for Universe 2 (from my_agent-1)."""
    Q2        = {}
    epsilon   = [1.0]
    alpha     = 0.30
    gamma     = 0.96    # Exit reward (+600) is distant — high gamma propagates it back
    eps_min   = 0.02
    eps_decay = 0.996
    n_actions = 4
    OPT_INIT  = 10.0    # Optimistic init: forces agent to try every action first

    def _get(s):
        if s not in Q2:
            Q2[s] = np.full(n_actions, OPT_INIT, dtype=np.float64)
        return Q2[s]

    def select_action(obs):
        s = discretize_universe2(obs)
        if random.random() < epsilon[0]:
            return random.randrange(n_actions)
        q  = _get(s)
        mx = float(np.max(q))
        # Random tie-breaking so agent doesn't always default to action 0
        return random.choice([a for a in range(n_actions) if q[a] == mx])

    def update(s, a, r, ns):
        # Standard Q-learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') - Q(s,a)]
        q_s  = _get(s)
        q_ns = _get(ns)
        q_s[a] += alpha * (r + gamma * float(np.max(q_ns)) - q_s[a])
        epsilon[0] = max(eps_min, epsilon[0] * eps_decay)

    return select_action, update, Q2


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE 3 — FIGHT VECNA
# Source: submission_template_2.py (Reactive Q-Learning)
#
# Rewards (README):
#   +200  each child rescued (dist < 1.5)
#   +300  both rescued (WIN)
#   -200  Vecna touch (dist < 2.5) — instant death
#   -150  projectile hit (dist < 0.8) — instant death
#   -80   Demogorgon collision (dist < 1.5)
#   -0.05 × dist per unrescued child per step
#   -1.0  per step, -100 timeout
#
# Why reactive: Vecna fires projectiles that track the agent — fixed
# grid states don't capture the dynamic threat. Mission-mode encoding
# (which child to fetch) ensures agent always knows its current goal.
#
# State: (target_direction x8, target_id x3, vecna_threat x3)
# = 8 × 3 × 3 = 72 states
# ─────────────────────────────────────────────────────────────────────────────

def discretize_universe3(observation):
    """
    Reactive state for Universe 3 (from submission_template_2).
    target_id encodes rescue status so agent knows when goals change.
    """
    ax, ay   = observation[0], observation[1]
    c1x, c1y = observation[2], observation[3]
    c2x, c2y = observation[4], observation[5]
    vx, vy   = observation[6], observation[7]
    c1r      = int(observation[8])
    c2r      = int(observation[9])

    # Mission mode: which child to target?
    if   not c1r: target_x, target_y, target_id = c1x, c1y, 0
    elif not c2r: target_x, target_y, target_id = c2x, c2y, 1
    else:         target_x, target_y, target_id = ax,  ay,  2

    # 8-octant direction to current target
    if target_id < 2:
        angle            = np.arctan2(target_y - ay, target_x - ax)
        target_direction = int((angle + np.pi) / (np.pi / 4)) % 8
    else:
        target_direction = 0

    # Vecna threat (README kills at <2.5, warn from 3.5 and 5.5)
    vecna_dist = math.hypot(vx - ax, vy - ay)
    if   vecna_dist < 3.5: vecna_threat = 2
    elif vecna_dist < 5.5: vecna_threat = 1
    else:                   vecna_threat = 0

    return (target_direction, target_id, vecna_threat)


def create_universe3_agent():
    """Reactive Q-learning agent for Universe 3 (from submission_template_2)."""
    Q3 = {}

    alpha     = 0.35
    gamma     = 0.92
    epsilon   = 0.85
    eps_decay = 0.996
    eps_min   = 0.10
    n_actions = 4

    action_history = deque(maxlen=5)

    def _get(s):
        if s not in Q3:
            Q3[s] = np.zeros(n_actions)
        return Q3[s]

    def select_action(observation):
        nonlocal epsilon
        state = discretize_universe3(observation)

        ax, ay   = observation[0], observation[1]
        c1x, c1y = observation[2], observation[3]
        c2x, c2y = observation[4], observation[5]
        vx, vy   = observation[6], observation[7]
        c1r      = int(observation[8])
        c2r      = int(observation[9])

        if random.random() < epsilon:
            # Determine current child target
            if   not c1r: tx, ty = c1x, c1y
            elif not c2r: tx, ty = c2x, c2y
            else:
                # Both rescued — just survive until win triggers
                action = random.randint(0, 3)
                action_history.append(action)
                epsilon = max(eps_min, epsilon * eps_decay)
                return action

            vecna_dist = math.hypot(vx - ax, vy - ay)

            # Flee Vecna if within danger zone (kills at <2.5, flee from 4.0)
            if vecna_dist < 4.0:
                dx_away = ax - vx
                dy_away = ay - vy
                action = (0 if dx_away > 0 else 2) if abs(dx_away) > abs(dy_away) \
                         else (1 if dy_away > 0 else 3)
            else:
                # Move toward current child
                dx = tx - ax
                dy = ty - ay
                action = (0 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 3)

        else:
            action = int(np.argmax(_get(state)))

        action_history.append(action)
        epsilon = max(eps_min, epsilon * eps_decay)
        return action

    def update(state, action, reward, next_state):
        nonlocal epsilon
        q_now  = _get(state)
        q_next = _get(next_state)

        # Boost learning signal for big child rescues (+200/+300)
        if reward > 100:
            td_target = reward + gamma * np.max(q_next) * 1.4
        else:
            td_target = reward + gamma * np.max(q_next)

        q_now[action] += alpha * (td_target - q_now[action])
        epsilon = max(eps_min, epsilon * eps_decay)

    return select_action, update, Q3


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ROUTER FUNCTION (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_for_universe(universe_name):
    """Router function provided by the competition template."""
    name = universe_name.upper()
    if "FIND WILL" in name:
        return create_universe1_agent(), discretize_universe1
    elif "ESCAPE" in name:
        return create_universe2_agent(), discretize_universe2
    elif "VECNA" in name or "FIGHT" in name:
        return create_universe3_agent(), discretize_universe3
    else:
        raise ValueError(f"Unknown universe: '{universe_name}'")