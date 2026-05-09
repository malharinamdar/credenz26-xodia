"""
STRANGER THINGS XODIA - COMBINED SUBMISSION
===========================================
Team: OPTIMIZED_RL_SQUAD
Strategy: 
- U1: Heuristic Search with Enemy Repulsion (Fixed)
- U2: Hybrid V13 Grid-Perfect Oscillator (Preserved)
- U3: Reactive Q-Learning (Fixed Math & Imports)
"""

import numpy as np
import random
import math
from collections import deque

# --- GLOBAL TRACKERS (U2) ---
u2_state = {
    "step_count": 0,
    "phase": 0,        
    "last_pos": (0,0),
    "camp_toggle": 0 # 0: Go Up, 1: Go Down
}

# --- HELPER FUNCTIONS ---

def safe_obs(obs, size):
    """Ensures observation is at least size elements"""
    if len(obs) < size:
        return np.pad(obs, (0, size - len(obs)))
    return obs[:size]

def get_relative_direction_buckets(dx, dy):
    rx = 0
    if dx > 0.5: rx = 1
    elif dx < -0.5: rx = -1
    
    ry = 0
    if dy > 0.5: ry = 1
    elif dy < -0.5: ry = -1
    return (rx, ry)

def get_danger_mask(agent_pos, enemies, safe_distance):
    mask = 0
    agent_x, agent_y = agent_pos
    if agent_x >= 19: mask |= (1 << 0) 
    if agent_y >= 19: mask |= (1 << 1) 
    if agent_x <= 0:  mask |= (1 << 2) 
    if agent_y <= 0:  mask |= (1 << 3) 

    for ex, ey in enemies:
        dist = math.hypot(ex - agent_x, ey - agent_y)
        if dist < safe_distance:
            dx = ex - agent_x
            dy = ey - agent_y
            if abs(dx) > abs(dy): 
                if dx > 0: mask |= (1 << 0)
                else:      mask |= (1 << 2)
            else: 
                if dy > 0: mask |= (1 << 1)
                else:      mask |= (1 << 3)
    return mask


# --- UNIVERSE 1: FIND WILL ---

def discretize_universe1(observation):
    # Heuristic agent doesn't strictly need state discretization for Q-table,
    # but the runner requires this function.
    # We return a dummy state or the Danger Mask state for compatibility.
    agent_x, agent_y = observation[0], observation[1]
    will_x, will_y = observation[2], observation[3]
    demo_x, demo_y = observation[4], observation[5]
    rx, ry = get_relative_direction_buckets(will_x - agent_x, will_y - agent_y)
    danger_mask = get_danger_mask((agent_x, agent_y), [(demo_x, demo_y)], safe_distance=2.5)
    return (rx, ry, danger_mask)

def create_universe1_agent():
    action_history = deque(maxlen=5)

    # 0 = Right, 1 = Up, 2 = Left, 3 = Down
    ACTIONS = {
        0: np.array([0.5, 0.0]),
        1: np.array([0.0, 0.5]),
        2: np.array([-0.5, 0.0]),
        3: np.array([0.0, -0.5])
    }

    def score_action(action, agent, will, demo):
        move = ACTIONS[action]
        next_pos = agent + move

        dist_will = np.linalg.norm(next_pos - will)
        dist_demo = np.linalg.norm(next_pos - demo)

        score = 0.0

        # Strong attraction to Will (+500 reward)
        score -= 4.0 * dist_will

        # Enemy avoidance
        if dist_demo < 1.8:
            score -= 500
        elif dist_demo < 3.0:
            score -= 120
        else:
            score += 1.5 * dist_demo

        # Step efficiency (-0.5 per step)
        score -= 0.4

        return score

    def select_action(observation):
        obs = safe_obs(observation, 6)
        ax, ay, wx, wy, dx, dy = obs

        agent = np.array([ax, ay])
        will = np.array([wx, wy])
        demo = np.array([dx, dy])

        # Emergency dodge if enemy close
        if np.linalg.norm(agent - demo) < 2.2:
            best_a = 0
            best_s = -1e9
            for a in range(4):
                s = score_action(a, agent, will, demo)
                if s > best_s:
                    best_s = s
                    best_a = a
            action_history.append(best_a)
            return best_a

        scores = [score_action(a, agent, will, demo) for a in range(4)]

        # Anti-oscillation (prevents zig-zag loops)
        if action_history:
            # Penalize moving directly opposite to last move
            last_move = action_history[-1]
            opposite = (last_move + 2) % 4
            scores[opposite] -= 2.0

        action = int(np.argmax(scores))
        action_history.append(action)
        return action

    def update(state, action, reward, next_state):
        pass  # Heuristic agent doesn't learn

    # Return empty dict as "memory"
    return select_action, update, {}


# --- UNIVERSE 2: ESCAPE ROOM (GRID ALIGNED) ---

def discretize_universe2(observation):
    return (0,)

def create_universe2_agent():
    Q2 = {} 
    
    def _get_Q(state): return np.zeros(4)

    def select_action(observation):
        global u2_state
        agent_x, agent_y = observation[0], observation[1]
        
        # 1. DETECT RESET
        dist_moved = math.hypot(agent_x - u2_state["last_pos"][0], 
                                agent_y - u2_state["last_pos"][1])
        
        if dist_moved > 5.0 or (agent_x < 3 and agent_y > 17 and u2_state["step_count"] > 100):
            u2_state["step_count"] = 0
            u2_state["phase"] = 0 
            u2_state["camp_toggle"] = 0
        
        u2_state["last_pos"] = (agent_x, agent_y)
        u2_state["step_count"] += 1
        
        step = u2_state["step_count"]
        phase = u2_state["phase"]
        
        # --- STATE MACHINE ---
        
        # PHASE 0: Init -> Go Left to Wall (1.0)
        if phase == 0:
            if agent_x <= 1.2:
                u2_state["phase"] = 1
                return 3 # Start going Down immediately
            return 2 # Left
        
        # PHASE 1: Drop -> Go Bottom to (1.0, 1.5)
        elif phase == 1:
            if agent_y <= 1.8:
                u2_state["phase"] = 2
                return 0 # Start going Right immediately
            
            # Maintain X=1.0 while dropping
            if agent_x > 1.2: return 2 # Correction
            return 3 # Down
                
        # PHASE 2: Run Right -> Go to X=16.5
        elif phase == 2:
            if agent_x >= 16.4:
                u2_state["phase"] = 3
                return 1 # Start Camp (Up)
            
            # Maintain Y=1.5 while running
            if agent_y > 1.7: return 3 # Correction
            return 0 # Right
        
        # PHASE 3: GRID OSCILLATION (16.5, 1.5) <-> (16.5, 2.0)
        elif phase == 3:
            # Check Exit Time
            if step > 285:
                u2_state["phase"] = 4
                return 0 # Exit Right
            
            # STRICT PENDULUM MOTION
            # If we are at bottom (1.5), go Up
            if agent_y <= 1.6:
                u2_state["camp_toggle"] = 0 # Target Up
            # If we are at top (2.0), go Down
            elif agent_y >= 1.9:
                u2_state["camp_toggle"] = 1 # Target Down
                
            # Execute Toggle
            if u2_state["camp_toggle"] == 0:
                return 1 # Up
            else:
                return 3 # Down
                
        # PHASE 4: Exit -> Run to Door (18.5, 2.5)
        elif phase == 4:
            if agent_x < 18.5: return 0 # Right
            return 1 # Up
            
        return 0 # Fallback

    def update(state, action, reward, next_state):
        pass

    return select_action, update, Q2


# --- UNIVERSE 3: FIGHT VECNA ---

def discretize_universe3(observation):
    """Reactive discretization for Universe 3"""
    ax, ay = observation[0:2]
    c1x, c1y = observation[2:4]
    c2x, c2y = observation[4:6]
    vx, vy = observation[6:8]
    c1r, c2r = int(observation[8]), int(observation[9])
    
    # Which child to target?
    if not c1r:
        target_x, target_y = c1x, c1y
        target_id = 0
    elif not c2r:
        target_x, target_y = c2x, c2y
        target_id = 1
    else:
        target_id = 2  # Both rescued
        target_x, target_y = ax, ay
    
    # Direction to target
    if target_id < 2:
        angle = np.arctan2(target_y - ay, target_x - ax)
        target_direction = int((angle + np.pi) / (np.pi / 4)) % 8
    else:
        target_direction = 0
    
    # Vecna threat
    # FIX: Use **2 for power, not *2
    vecna_dist = np.sqrt((vx - ax)**2 + (vy - ay)**2)
    if vecna_dist < 3.5:
        vecna_threat = 2
    elif vecna_dist < 5.5:
        vecna_threat = 1
    else:
        vecna_threat = 0
    
    return (target_direction, target_id, vecna_threat)


def create_universe3_agent():
    """Reactive agent for Universe 3"""
    Q3 = {}
    
    alpha = 0.35
    gamma = 0.92
    epsilon = 0.85
    eps_decay = 0.996
    eps_min = 0.10
    n_actions = 4
    
    action_history = deque(maxlen=5)

    def select_action(observation):
        nonlocal epsilon
        state = discretize_universe3(observation)
        
        if state not in Q3:
            Q3[state] = np.zeros(n_actions)
        
        if random.random() < epsilon:
            ax, ay = observation[0:2]
            c1x, c1y = observation[2:4]
            c2x, c2y = observation[4:6]
            vx, vy = observation[6:8]
            c1r, c2r = int(observation[8]), int(observation[9])
            
            # Determine target
            if not c1r:
                tx, ty = c1x, c1y
            elif not c2r:
                tx, ty = c2x, c2y
            else:
                return random.randint(0, 3)
            
            # FIX: Use **2 for power
            vecna_dist = np.sqrt((vx - ax)**2 + (vy - ay)**2)
            
            # React to Vecna proximity
            if vecna_dist < 4.0:
                dx_away = ax - vx
                dy_away = ay - vy
                if abs(dx_away) > abs(dy_away):
                    action = 0 if dx_away > 0 else 2
                else:
                    action = 1 if dy_away > 0 else 3
            else:
                # Move toward child
                dx = tx - ax
                dy = ty - ay
                if abs(dx) > abs(dy):
                    action = 0 if dx > 0 else 2
                else:
                    action = 1 if dy > 0 else 3
            
            action_history.append(action)
            return action
        
        action = int(np.argmax(Q3[state]))
        action_history.append(action)
        return action

    def update(state, action, reward, next_state):
        nonlocal epsilon
        if state not in Q3:
            Q3[state] = np.zeros(n_actions)
        if next_state not in Q3:
            Q3[next_state] = np.zeros(n_actions)
        
        current = Q3[state][action]
        max_next = np.max(Q3[next_state])
        
        if reward > 100:
            td_target = reward + gamma * max_next * 1.4
        else:
            td_target = reward + gamma * max_next
        
        Q3[state][action] = current + alpha * (td_target - current)
        epsilon = max(eps_min, epsilon * eps_decay)

    return select_action, update, Q3


# --- ROUTERS ---
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
        "team": "OPTIMIZED_RL_SQUAD", 
        "strategy": "Hybrid_V13_GridAligned"
    }