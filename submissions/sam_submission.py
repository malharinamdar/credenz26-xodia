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
    
    rel_will_x = int((will_x - agent_x) / 2)
    rel_will_y = int((will_y - agent_y) / 2)
    
    # Only care about the demo if it's within 3 units
    dist_demo = abs(demo_x - agent_x) + abs(demo_y - agent_y)
    if dist_demo > 2:
        dist_demo = 3 # "Far away" constant
        
    # TODO: Implement your discretization-->done
    return (rel_will_x, rel_will_y, dist_demo) # Replace this!


def create_universe1_agent():
    """Create Q-learning agent for Universe 1"""
    Q1 = {}
    
    # TODO: Set hyperparameters -->done
    alpha = 0.2     # Learning rate
    gamma = 0.95     # Discount factor
    epsilon = 0.1   # Exploration rate
    params = {"epsilon": 0.1}

    def select_action(observation):
        # TODO: Implement epsilon-greedy-->done
        if np.random.random() < epsilon:
            return random.randint(0, 3)  # 0=Right, 1=Up, 2=Left, 3=Down
        else:
            state = discretize_universe1(observation)
            if state not in Q1:
                return random.randint(0, 3)
            return np.argmax(Q1[state])
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        if next_state not in Q1:
            Q1[next_state] = np.zeros(4)
        if state not in Q1:
            Q1[state] = np.zeros(4)
        best_next_q = np.max(Q1[next_state])
        Q1[state][action] += alpha * (reward - abs(state[0]) - abs(state[1])+abs(state[2]) + gamma * best_next_q - Q1[state][action])
    
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
    relative_exit_x = int((exit_x - agent_x) / 2)
    relative_exit_y = int((exit_y - agent_y) / 2)
    relative_guard_x = int((guard_x - agent_x) / 2)
    relative_guard_y = int((guard_y - agent_y) / 2)
    
    # TODO: Implement your discretization
    state = (relative_exit_x, relative_exit_y, relative_guard_x, relative_guard_y)  # Replace this!
    
    return state


def create_universe2_agent():
    """Create Q-learning agent for Universe 2"""
    Q2 = {}
    
    # TODO: Set hyperparameters
    alpha = 0.4     # Learning rate
    gamma = 0.95     # Discount factor
    epsilon = 0.1   # Exploration rate
    params = {"epsilon": 0.1}

    def select_action(observation):
        params["epsilon"] = max(0.01, params["epsilon"] * 0.9999)  # Decay epsilon
        # TODO: Implement action selection
        if np.random.random() < params["epsilon"]:
            return random.randint(0, 3)  # 0=Right, 1=Up, 2=Left, 3=Down
        else:
            state = discretize_universe2(observation)
            if state not in Q2:
                return random.randint(0, 3)
            return np.argmax(Q2[state])
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        
        if next_state not in Q2:
            Q2[next_state] = np.zeros(4)
        if state not in Q2:
            Q2[state] = np.zeros(4)
        best_next_q = np.max(Q2[next_state])
        Q2[state][action] += alpha * (reward - abs(state[0]) - abs(state[1])+abs(state[2])+abs(state[3]) + gamma * best_next_q - Q2[state][action])
    
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
    child1_pos_x, child1_pos_y = observation[2], observation[3]
    child2_pos_x, child2_pos_y = observation[4], observation[5]
    vecna_x, vecna_y = observation[6], observation[7]
    proj_x, proj_y = observation[10], observation[11]
    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])

    venca_pos = 0  # Default "close" constant
    venca_dist = np.sqrt(np.square(vecna_x - agent_x) + np.square(vecna_y - agent_y))
    if venca_dist < 3:
        venca_pos = 1  # "Far away" constant
    proj_pos = 0  # Default "far" constant
    proj_dist = np.sqrt(np.square(proj_x - agent_x) + np.square(proj_y - agent_y))
    if proj_dist < 3:
        proj_pos = 1  # "Close" constant
        
    if child1_rescued:
        rel_child_x, rel_child_y = int((child2_pos_x - agent_x) / 4), int((child2_pos_y - agent_y) / 2)
    else:
        rel_child_x, rel_child_y = int((child1_pos_x - agent_x) / 4), int((child1_pos_y - agent_y) / 2)

    # TODO: Implement your discretization
    state = (int(rel_child_x), int(rel_child_y), child1_rescued, child2_rescued,proj_pos, venca_pos)  # Replace this!
    
    return state


def create_universe3_agent():
    """Create Q-learning agent for Universe 3"""
    Q3 = {}
    
    # TODO: Set hyperparameters
    alpha = 0.4
    gamma = 0.95
    epsilon = 0.2
    params = {"epsilon": epsilon}
    
    def select_action(observation):
        # TODO: Implement action selection
        params["epsilon"] = max(0.01, params["epsilon"] * 0.999)  # Decay epsilon
        if np.random.random() < params["epsilon"]:
            return random.randint(0, 3)  # 0=Right, 1=Up, 2=Left, 3=Down
        else:
            state = discretize_universe3(observation)
            if state not in Q3:
                return random.randint(0, 3)
            return np.argmax(Q3[state])
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        if next_state not in Q3:
            Q3[next_state] = np.zeros(4)
        if state not in Q3:
            Q3[state] = np.zeros(4)
        best_next_q = np.max(Q3[next_state])
        Q3[state][action] += alpha * (reward + gamma * best_next_q - Q3[state][action])
    
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
    """
    Returns a dictionary with team information.
    """
    return {
        "team_name": "Spartans",  # Replace with your team name
        "participants": ["Sameer Mangulkar" , "Vinit Vibahdik"], # Replace with your name
        "description": "A specialized Q-Learning suite utilizing dynamic epsilon decay and "
            "conditional state-space reduction. Universe 3 employs target-switching. Was unable to solve universe 3 within 200 episodes"
            "logic and binary proximity flags for instant-death threats like Vecna "
            "and projectiles to ensure high survivability within 200 episodes."
    }
