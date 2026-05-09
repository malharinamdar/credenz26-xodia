"""
STRANGER THINGS XODIA - SUBMISSION TEMPLATE
============================================
Team: [YOUR TEAM NAME]

TASK: Create 3 separate RL agents (one per universe)
- Universe 1: Find Will (6 Demogorgons)
- Universe 2: Escape Room (5 Guards)
- Universe 3: Fight Vecna (6 Demogorgons + Vecna)
"""

import numpy as np
import random


# UNIVERSE 1: FIND WILL


def discretize_universe1(observation):
    """
    Discretize state for Universe 1

    observation[0:2]: Agent position (x, y)
    observation[2:4]: Will position (x, y)
    observation[4:6]: Nearest Demogorgon (x, y)
    """

    bin_size = 3
    agent_x, agent_y = observation[0], observation[1]

    def rel(a, base, clip=5):
        return int(np.clip((a - base) / bin_size, -clip, clip))

    demo_dist = np.sqrt((observation[4] - agent_x) ** 2 + (observation[5] - agent_y) ** 2)
    if demo_dist < 2.5:
        danger = 3
    elif demo_dist < 4.0:
        danger = 2
    elif demo_dist < 6.0:
        danger = 1
    else:
        danger = 0

    return (
        rel(observation[2], agent_x),  # will dx
        rel(observation[3], agent_y),  # will dy
        danger,  # demogorgon danger level
        rel(observation[4], agent_x),  # demo dx (direction to avoid)
        rel(observation[5], agent_y),  # demo dy
    )


def create_universe1_agent():
    """Create Q-learning agent for Universe 1"""
    Q1 = {}

    # Hyperparameters
    alpha = 0.4  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon_start = 1.0  # Initial exploration rate
    epsilon_min = 0.15  # Minimum exploration rate
    decay_rate = 0.9995  # Step-based decay rate

    step_count = [0]  # Mutable counter for step-based decay

    def select_action(observation):
        # Discretize the observation to get state
        state = discretize_universe1(observation)

        # Initialize Q-values for new state
        if state not in Q1:
            Q1[state] = [0.0, 0.0, 0.0, 0.0]

        # Step-based epsilon decay
        step_count[0] += 1
        epsilon = max(epsilon_min, epsilon_start * (decay_rate ** step_count[0]))

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(Q1[state])  # Exploit

    def update(state, action, reward, next_state):
        # Initialize Q-values for new states
        if state not in Q1:
            Q1[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q1:
            Q1[next_state] = [0.0, 0.0, 0.0, 0.0]

        # Q-learning update rule
        best_next = max(Q1[next_state])
        Q1[state][action] += alpha * (reward + gamma * best_next - Q1[state][action])

    return select_action, update, Q1


# UNIVERSE 2: ESCAPE ROOM


def discretize_universe2(observation):
    """
    Discretize state for Universe 2

    observation[0:2]: Agent position (x, y)
    observation[2:4]: Exit position (x, y)
    observation[4:6]: Nearest guard (x, y)
    """
    # Use relative positions to reduce state space and improve generalization
    bin_size = 2
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]
    guard_x, guard_y = observation[4], observation[5]

    state = (
        int((exit_x - agent_x) / bin_size),  # Relative to exit (goal)
        int((exit_y - agent_y) / bin_size),
        int((guard_x - agent_x) / bin_size),  # Relative to nearest guard (enemy)
        int((guard_y - agent_y) / bin_size),
    )

    return state


def create_universe2_agent():
    """Create Q-learning agent for Universe 2"""
    Q2 = {}

    # Hyperparameters
    alpha = 0.1
    gamma = 0.95
    epsilon_start = 1.0  # Initial exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    decay_rate = 0.9995  # Step-based decay rate

    step_count = [0]  # Mutable counter for step-based decay

    def select_action(observation):
        # Discretize the observation to get state
        state = discretize_universe2(observation)

        # Initialize Q-values for new state
        if state not in Q2:
            Q2[state] = [0.0, 0.0, 0.0, 0.0]

        # Step-based epsilon decay
        step_count[0] += 1
        epsilon = max(epsilon_min, epsilon_start * (decay_rate ** step_count[0]))

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(Q2[state])  # Exploit

    def update(state, action, reward, next_state):
        # Initialize Q-values for new states
        if state not in Q2:
            Q2[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q2:
            Q2[next_state] = [0.0, 0.0, 0.0, 0.0]

        # Q-learning update rule
        best_next = max(Q2[next_state])
        Q2[state][action] += alpha * (reward + gamma * best_next - Q2[state][action])

    return select_action, update, Q2


# UNIVERSE 3: FIGHT VECNA


def discretize_universe3(observation):
    """
    Discretize state for Universe 3

    observation[0:2]:   Agent position (x, y)
    observation[2:4]:   Child 1 position (x, y)
    observation[4:6]:   Child 2 position (x, y)
    observation[6:8]:   Vecna position (x, y)
    observation[8]:     Child 1 rescued (1.0 or 0.0)
    observation[9]:     Child 2 rescued (1.0 or 0.0)
    observation[10:12]: Projectile position (x, y)
    """
    agent_x, agent_y = observation[0], observation[1]
    c1_rescued = int(observation[8])
    c2_rescued = int(observation[9])
    b = 2  # Coarser bins

    def rel(a, base, clip=8):
        return int(np.clip((a - base) / b, -clip, clip))

    # Only track the UNRESCUED child — don't waste state space on rescued ones
    if not c1_rescued:
        target_dx = rel(observation[2], agent_x)
        target_dy = rel(observation[3], agent_y)
    else:
        target_dx = rel(observation[4], agent_x)
        target_dy = rel(observation[5], agent_y)

    # Vecna danger level instead of raw position
    vecna_dist = np.sqrt((observation[6] - agent_x) ** 2 + (observation[7] - agent_y) ** 2)
    if vecna_dist < 2.5:
        vecna_danger = 3
    elif vecna_dist < 4.0:
        vecna_danger = 2
    elif vecna_dist < 7.0:
        vecna_danger = 1
    else:
        vecna_danger = 0

    # Projectile danger level instead of raw position
    proj_dist = np.sqrt((observation[10] - agent_x) ** 2 + (observation[11] - agent_y) ** 2)
    if proj_dist < 1.0:
        proj_danger = 3
    elif proj_dist < 2.5:
        proj_danger = 2
    elif proj_dist < 5.0:
        proj_danger = 1
    else:
        proj_danger = 0

    return (
        target_dx,
        target_dy,  # Direction to current target child
        vecna_danger,  # How dangerous is Vecna right now
        proj_danger,  # How dangerous is projectile
        rel(observation[6], agent_x),
        rel(observation[7], agent_y),  # Vecna direction
        c1_rescued,
        c2_rescued,  # Progress flags
    )


def create_universe3_agent():
    """Create Q-learning agent for Universe 3"""
    Q3 = {}

    # Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon_start = 1.0  # Initial exploration rate
    epsilon_min = 0.1  # Minimum exploration rate
    decay_rate = 0.9995  # Step-based decay rate

    step_count = [0]  # Mutable counter for step-based decay

    def select_action(observation):
        # Discretize the observation to get state
        state = discretize_universe3(observation)

        # Initialize Q-values for new state
        if state not in Q3:
            Q3[state] = [0.0, 0.0, 0.0, 0.0]

        # Step-based epsilon decay
        step_count[0] += 1
        epsilon = max(epsilon_min, epsilon_start * (decay_rate ** step_count[0]))

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(Q3[state])  # Exploit

    def update(state, action, reward, next_state):
        # Initialize Q-values for new states
        if state not in Q3:
            Q3[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q3:
            Q3[next_state] = [0.0, 0.0, 0.0, 0.0]

        # Q-learning update rule
        best_next = max(Q3[next_state])
        Q3[state][action] += alpha * (reward + gamma * best_next - Q3[state][action])

    return select_action, update, Q3


# REQUIRED FUNCTIONS


def get_agent_for_universe(universe_name):
    """Router function - DO NOT MODIFY"""
    if "FIND WILL" in universe_name:
        return create_universe1_agent(), discretize_universe1
    elif "ESCAPE" in universe_name:
        return create_universe2_agent(), discretize_universe2
    elif "VECNA" in universe_name or "FIGHT" in universe_name:
        return create_universe3_agent(), discretize_universe3
    else:
        raise ValueError(f"Unknown universe: {universe_name}")


def get_agent_info():
    """Return agent configuration info"""
    return {
        "agent_type": "Tabular Q-Learning",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.1,
        "state_discretization": "position binning (bin_size=2)",
    }
