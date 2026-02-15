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
    
    # TODO: Implement your discretization
    state = (int(agent_x), int(agent_y))  # Replace this!
    
    return state


def create_universe1_agent():
    """Create Q-learning agent for Universe 1"""
    Q1 = {}
    
    # TODO: Set hyperparameters
    alpha = NA      # Learning rate
    gamma = NA      # Discount factor
    epsilon = NA    # Exploration rate
    
    def select_action(observation):
        # TODO: Implement epsilon-greedy
        return random.randint(0, 3)  # 0=Right, 1=Up, 2=Left, 3=Down
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        pass
    
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
    
    # TODO: Implement your discretization
    state = (int(agent_x), int(agent_y))  # Replace this!
    
    return state


def create_universe2_agent():
    """Create Q-learning agent for Universe 2"""
    Q2 = {}
    
    # TODO: Set hyperparameters
    alpha = NA
    gamma = NA
    epsilon = NA
    
    def select_action(observation):
        # TODO: Implement action selection
        return random.randint(0, 3)
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        pass
    
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
    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])
    
    # TODO: Implement your discretization
    state = (int(agent_x), int(agent_y), child1_rescued, child2_rescued)  # Replace this!
    
    return state


def create_universe3_agent():
    """Create Q-learning agent for Universe 3"""
    Q3 = {}
    
    # TODO: Set hyperparameters
    alpha = NA
    gamma = NA
    epsilon = NA
    
    def select_action(observation):
        # TODO: Implement action selection
        return random.randint(0, 3)
    
    def update(state, action, reward, next_state):
        # TODO: Q-learning update
        pass
    
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
