"""
STRANGER THINGS XODIA - SUBMISSION TEMPLATE
============================================
Team: CODE MONK
"""

import numpy as np
import random
def bucket_distance(d):
    d = abs(d)
    if d == 0:
        return 0
    elif d <= 2:
        return 1
    else:
        return 2


# =========================================================
# UNIVERSE 1: FIND WILL
# =========================================================

def discretize_universe1(observation):
    agent_x, agent_y = observation[0], observation[1]
    will_x, will_y = observation[2], observation[3]

    dx = np.sign(will_x - agent_x)
    dy = np.sign(will_y - agent_y)

    state = (dx, dy)
    return state


def create_universe1_agent():
    Q1 = {}

    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    def select_action(observation):
        nonlocal epsilon
        state = discretize_universe1(observation)

        if state not in Q1:
            Q1[state] = np.zeros(4)

        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q1[state]))

    def update(state, action, reward, next_state):
        nonlocal epsilon

        if state not in Q1:
            Q1[state] = np.zeros(4)

        if next_state not in Q1:
            Q1[next_state] = np.zeros(4)

        best_next = np.max(Q1[next_state])

        Q1[state][action] += alpha * (
            reward + gamma * best_next - Q1[state][action]
        )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return select_action, update, Q1


# =========================================================
# UNIVERSE 2: ESCAPE ROOM
# =========================================================

def discretize_universe2(observation):
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]

    dx = np.sign(exit_x - agent_x)
    dy = np.sign(exit_y - agent_y)

    state = (dx, dy)
    return state


def create_universe2_agent():
    Q2 = {}

    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    def select_action(observation):
        nonlocal epsilon
        state = discretize_universe2(observation)

        if state not in Q2:
            Q2[state] = np.zeros(4)

        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q2[state]))

    def update(state, action, reward, next_state):
        nonlocal epsilon

        if state not in Q2:
            Q2[state] = np.zeros(4)

        if next_state not in Q2:
            Q2[next_state] = np.zeros(4)

        best_next = np.max(Q2[next_state])

        Q2[state][action] += alpha * (
            reward + gamma * best_next - Q2[state][action]
        )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return select_action, update, Q2


# =========================================================
# UNIVERSE 3: FIGHT VECNA
# =========================================================

def discretize_universe3(observation):
    agent_x, agent_y = observation[0], observation[1]

    child1_x, child1_y = observation[2], observation[3]
    child2_x, child2_y = observation[4], observation[5]

    vecna_x, vecna_y = observation[6], observation[7]

    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])

    # If children not rescued → target nearest child
    if child1_rescued == 0:
        target_x, target_y = child1_x, child1_y
        phase = 0
    elif child2_rescued == 0:
        target_x, target_y = child2_x, child2_y
        phase = 0
    else:
        target_x, target_y = vecna_x, vecna_y
        phase = 1  # Combat phase

    dx = bucket_distance(target_x - agent_x)
    dy = bucket_distance(target_y - agent_y)

    state = (dx, dy, phase)
    return state



def create_universe3_agent():
    Q3 = {}

    alpha = 0.3
    gamma = 0.98
    epsilon = 1.0
    epsilon_decay = 0.990
    epsilon_min = 0.05

    def select_action(observation):
        nonlocal epsilon
        state = discretize_universe3(observation)

        if state not in Q3:
            Q3[state] = np.zeros(4)

        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q3[state]))

    def update(state, action, reward, next_state):
        nonlocal epsilon

        if state not in Q3:
            Q3[state] = np.zeros(4)

        if next_state not in Q3:
            Q3[next_state] = np.zeros(4)

        best_next = np.max(Q3[next_state])

        Q3[state][action] += alpha * (
            reward + gamma * best_next - Q3[state][action]
        )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return select_action, update, Q3


# =========================================================
# REQUIRED ROUTER FUNCTION (DO NOT MODIFY)
# =========================================================

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
        "team_name": "CODE MONK",
        "team_members": ["Yash Pakhale", "Adesh Singh"],

        "strategy": "Q-Learning with distance bucketing and epsilon decay"
    }
