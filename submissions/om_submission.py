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
    agent_x, agent_y = observation[0], observation[1]
    will_x, will_y = observation[2], observation[3]
    demo_x, demo_y = observation[4], observation[5]
    
    # Discretize into buckets (5x5 units per bucket = 4x4 grid)
    agent_bucket_x = min(int(agent_x / 5), 3)
    agent_bucket_y = min(int(agent_y / 5), 3)
    will_bucket_x = min(int(will_x / 5), 3)
    will_bucket_y = min(int(will_y / 5), 3)
    
    # Relative position to Will (coarse)
    dx = 1 if agent_x < will_x - 3 else (2 if agent_x > will_x + 3 else 0)
    dy = 1 if agent_y < will_y - 3 else (2 if agent_y > will_y + 3 else 0)
    
    # Demogorgon danger level
    demo_dist = np.sqrt((agent_x - demo_x)**2 + (agent_y - demo_y)**2)
    demo_danger = 0 if demo_dist > 5 else (1 if demo_dist > 3 else 2)
    
    state = (agent_bucket_x, agent_bucket_y, dx, dy, demo_danger)
    
    return state


def create_universe1_agent():
    """Create Q-learning agent for Universe 1"""
    Q1 = {}
    
    # Hyperparameters
    alpha = 0.20      # Learning rate
    gamma = 0.97      # Discount factor
    epsilon = 0.18    # Exploration rate
    
    def select_action(observation):
        state = discretize_universe1(observation)
        
        # Initialize Q-values for new state
        if state not in Q1:
            Q1[state] = [0.0, 0.0, 0.0, 0.0]
        
        # Epsilon-greedy: explore vs exploit
        if random.random() < epsilon:
            return random.randint(0, 3)  # Random action (0=Right, 1=Up, 2=Left, 3=Down)
        else:
            return int(np.argmax(Q1[state]))  # Best action
    
    def update(state, action, reward, next_state):
        # Initialize Q-values if needed
        if state not in Q1:
            Q1[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q1:
            Q1[next_state] = [0.0, 0.0, 0.0, 0.0]
        
        # Q-learning update formula
        best_next_q = max(Q1[next_state])
        Q1[state][action] = Q1[state][action] + alpha * (reward + gamma * best_next_q - Q1[state][action])
    
    return select_action, update, Q1


# UNIVERSE 2: ESCAPE ROOM

def discretize_universe2(observation):
    """
    Discretize state for Universe 2
    
    observation[0:2]: Agent position (x, y)
    observation[2:4]: Exit position (x, y)
    observation[4:6]: Nearest guard (x, y)
    """
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]
    guard_x, guard_y = observation[4], observation[5]
    
    # Discretize agent position (finer grid for precision)
    agent_bucket_x = min(int(agent_x / 4), 4)
    agent_bucket_y = min(int(agent_y / 4), 4)
    
    # Relative direction to exit
    dx = 1 if agent_x < exit_x - 4 else (2 if agent_x > exit_x + 4 else 0)
    dy = 1 if agent_y < exit_y - 4 else (2 if agent_y > exit_y + 4 else 0)
    
    # Guard proximity (critical for avoiding detection)
    guard_dist = np.sqrt((agent_x - guard_x)**2 + (agent_y - guard_y)**2)
    if guard_dist < 2.5:
        guard_danger = 3  # Critical danger
    elif guard_dist < 4.5:
        guard_danger = 2  # High danger (detection range)
    elif guard_dist < 7:
        guard_danger = 1  # Medium danger
    else:
        guard_danger = 0  # Safe
    
    # Distance to exit (for progress tracking)
    dist_to_exit = np.sqrt((agent_x - exit_x)**2 + (agent_y - exit_y)**2)
    exit_proximity = 0 if dist_to_exit > 10 else (1 if dist_to_exit > 5 else 2)
    
    state = (agent_bucket_x, agent_bucket_y, dx, dy, guard_danger, exit_proximity)
    
    return state


def create_universe2_agent():
    """Create Q-learning agent for Universe 2"""
    Q2 = {}
    
    # Hyperparameters (more exploration needed for complex pathing)
    alpha = 0.22
    gamma = 0.95
    epsilon = 0.10
    
    def select_action(observation):
        state = discretize_universe2(observation)
        
        if state not in Q2:
            Q2[state] = [0.0, 0.0, 0.0, 0.0]
        
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q2[state]))
    
    def update(state, action, reward, next_state):
        if state not in Q2:
            Q2[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q2:
            Q2[next_state] = [0.0, 0.0, 0.0, 0.0]
        
        best_next_q = max(Q2[next_state])
        Q2[state][action] = Q2[state][action] + alpha * (reward + gamma * best_next_q - Q2[state][action])
    
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
    child1_x, child1_y = observation[2], observation[3]
    child2_x, child2_y = observation[4], observation[5]
    vecna_x, vecna_y = observation[6], observation[7]
    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])
    proj_x, proj_y = observation[10], observation[11]
    
    # Agent position (coarse buckets)
    agent_bucket_x = min(int(agent_x / 5), 3)
    agent_bucket_y = min(int(agent_y / 5), 3)
    
    # Mission state (which children are rescued)
    mission_state = child1_rescued * 2 + child2_rescued  # 0, 1, 2, or 3
    
    # Next target direction
    if not child1_rescued:
        target_x, target_y = child1_x, child1_y
    elif not child2_rescued:
        target_x, target_y = child2_x, child2_y
    else:
        target_x, target_y = 10, 10  # Center if both rescued
    
    dx = 1 if agent_x < target_x - 3 else (2 if agent_x > target_x + 3 else 0)
    dy = 1 if agent_y < target_y - 3 else (2 if agent_y > target_y + 3 else 0)
    
    # Vecna danger
    vecna_dist = np.sqrt((agent_x - vecna_x)**2 + (agent_y - vecna_y)**2)
    vecna_danger = 0 if vecna_dist > 6 else (1 if vecna_dist > 4 else 2)
    
    # Projectile danger
    proj_dist = np.sqrt((agent_x - proj_x)**2 + (agent_y - proj_y)**2)
    proj_danger = 1 if proj_dist < 3 else 0
    
    state = (agent_bucket_x, agent_bucket_y, mission_state, dx, dy, vecna_danger, proj_danger)
    
    return state


def create_universe3_agent():
    """Create Q-learning agent for Universe 3"""
    Q3 = {}
    
    # Hyperparameters (highest exploration for most complex universe)
    alpha = 0.23
    gamma = 0.95
    epsilon = 0.11
    
    def select_action(observation):
        state = discretize_universe3(observation)
        
        if state not in Q3:
            Q3[state] = [0.0, 0.0, 0.0, 0.0]
        
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q3[state]))
    
    def update(state, action, reward, next_state):
        if state not in Q3:
            Q3[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in Q3:
            Q3[next_state] = [0.0, 0.0, 0.0, 0.0]
        
        best_next_q = max(Q3[next_state])
        Q3[state][action] = Q3[state][action] + alpha * (reward + gamma * best_next_q - Q3[state][action])
    
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
    """Return agent metadata - REQUIRED by competition runner"""
    return {
        "team_name": "YOUR_TEAM_NAME_HERE",
        "algorithm": "Q-Learning",
        "universes": 3,
        "description": "Multi-agent Q-learning with state discretization"
    }