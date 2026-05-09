import numpy as np
import random

ACTIONS = 4


############################################
# HELPER FUNCTIONS
############################################

def direction(ax, ay, tx, ty, tol=2):
    dx = 0 if abs(ax - tx) < tol else (1 if ax < tx else 2)
    dy = 0 if abs(ay - ty) < tol else (1 if ay < ty else 2)
    return dx, dy


def distance_bin(dist, bins):
    for i, b in enumerate(bins):
        if dist < b:
            return i
    return len(bins)


def best_action(q_values):
    max_q = q_values.max()
    actions = np.flatnonzero(q_values == max_q)
    return random.choice(actions)


############################################
# UNIVERSE 1 — FIND WILL
############################################

def discretize_universe1(obs):
    ax, ay, wx, wy, dx, dy = obs

    will_dist = np.hypot(ax-wx, ay-wy)
    demo_dist = np.hypot(ax-dx, ay-dy)

    dirx, diry = direction(ax, ay, wx, wy)

    return (
        distance_bin(will_dist, [3,6,10]),
        distance_bin(demo_dist, [2,4,6]),
        dirx,
        diry
    )


def create_universe1_agent():

    Q = {}

    alpha = 0.18
    gamma = 0.96

    epsilon = 1.0
    decay = 0.995
    min_epsilon = 0.05


    def select_action(obs):
        nonlocal epsilon

        state = discretize_universe1(obs)

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = best_action(Q[state])

        epsilon = max(min_epsilon, epsilon * decay)

        return action


    def update(state, action, reward, next_state):

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if next_state not in Q:
            Q[next_state] = np.zeros(ACTIONS, dtype=np.float32)

        td_target = reward + gamma * Q[next_state].max()
        Q[state][action] += alpha * (td_target - Q[state][action])


    return select_action, update, Q


############################################
# UNIVERSE 2 — ESCAPE LAB
############################################

def discretize_universe2(obs):
    ax, ay, ex, ey, gx, gy = obs

    exit_dist = np.hypot(ax-ex, ay-ey)
    guard_dist = np.hypot(ax-gx, ay-gy)

    dirx, diry = direction(ax, ay, ex, ey)

    return (
        distance_bin(exit_dist, [2,5,9,14]),
        distance_bin(guard_dist, [2,4.5,7]),
        dirx,
        diry
    )


def create_universe2_agent():

    Q = {}

    alpha = 0.15
    gamma = 0.97

    epsilon = 1.0
    decay = 0.994
    min_epsilon = 0.05


    def select_action(obs):
        nonlocal epsilon

        state = discretize_universe2(obs)

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = best_action(Q[state])

        epsilon = max(min_epsilon, epsilon * decay)

        return action


    def update(state, action, reward, next_state):

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if next_state not in Q:
            Q[next_state] = np.zeros(ACTIONS, dtype=np.float32)

        td_target = reward + gamma * Q[next_state].max()
        Q[state][action] += alpha * (td_target - Q[state][action])


    return select_action, update, Q


############################################
# UNIVERSE 3 — FIGHT VECNA
############################################

def discretize_universe3(obs):

    ax, ay = obs[0], obs[1]
    c1x, c1y = obs[2], obs[3]
    c2x, c2y = obs[4], obs[5]
    vx, vy = obs[6], obs[7]
    r1, r2 = int(obs[8]), int(obs[9])
    px, py = obs[10], obs[11]

    if not r1:
        tx, ty = c1x, c1y
    elif not r2:
        tx, ty = c2x, c2y
    else:
        tx, ty = 10,10

    target_dist = np.hypot(ax-tx, ay-ty)
    vecna_dist = np.hypot(ax-vx, ay-vy)
    proj_dist = np.hypot(ax-px, ay-py)

    dirx, diry = direction(ax, ay, tx, ty)

    mission = r1*2 + r2

    return (
        mission,
        distance_bin(target_dist, [3,6,10]),
        distance_bin(vecna_dist, [2.5,5,8]),
        distance_bin(proj_dist, [1,3]),
        dirx,
        diry
    )


def create_universe3_agent():

    Q = {}

    alpha = 0.16
    gamma = 0.98

    epsilon = 1.0
    decay = 0.993
    min_epsilon = 0.07


    def select_action(obs):
        nonlocal epsilon

        state = discretize_universe3(obs)

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = best_action(Q[state])

        epsilon = max(min_epsilon, epsilon * decay)

        return action


    def update(state, action, reward, next_state):

        if state not in Q:
            Q[state] = np.zeros(ACTIONS, dtype=np.float32)

        if next_state not in Q:
            Q[next_state] = np.zeros(ACTIONS, dtype=np.float32)

        td_target = reward + gamma * Q[next_state].max()
        Q[state][action] += alpha * (td_target - Q[state][action])


    return select_action, update, Q


############################################
# REQUIRED ROUTER
############################################

def get_agent_for_universe(universe_name):

    if "FIND WILL" in universe_name:
        return create_universe1_agent(), discretize_universe1

    elif "ESCAPE" in universe_name:
        return create_universe2_agent(), discretize_universe2

    elif "VECNA" in universe_name or "FIGHT" in universe_name:
        return create_universe3_agent(), discretize_universe3

    else:
        raise ValueError(f"Unknown universe: {universe_name}")


def get_agent_info():
    return {
        "team_name": "Data_Rangers",
        "algorithm": "Advanced Q-Learning with Smart Discretization",
        "universes": 3,
        "description": "Competition-optimized RL agents with epsilon decay and threat-aware states"
    }
