import numpy as np
import random
import math

# ===============================
# COMMON UTILITIES
# ===============================

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def bucket(d):
    if d < 3: return 0
    elif d < 6: return 1
    elif d < 10: return 2
    else: return 3

# ===============================
# UNIVERSE 1: FIND WILL
# ===============================

def bucket_will(dist):
    if dist < 2: return 0
    elif dist < 6: return 1
    else: return 2

def bucket_enemy(dist):
    if dist < 2: return 0
    elif dist < 5: return 1
    else: return 2
def extract_vector_features(observation):
    # Observation: [ax, ay, wx, wy, nx, ny]
    ax, ay, wx, wy, nx, ny = observation
    
    # 1. ATTRACTIVE FORCE (Will)
    dx_w, dy_w = wx - ax, wy - ay
    dist_w = math.sqrt(dx_w**2 + dy_w**2)
    
    # 2. REPULSIVE FORCE (Nearest Demogorgon)
    dx_n, dy_n = nx - ax, ny - ay
    dist_n = math.sqrt(dx_n**2 + dy_n**2)
    
    # CALCULATE RESULTANT VECTOR
    # We weight the 'Push' of the enemy based on how close they are.
    # If dist_n > 3.0, the push is effectively 0.
    push_strength = 0
    if dist_n < 3.0:
        # Inverse square law: force increases drastically as you get closer
        push_strength = 1.0 / (dist_n**2 + 0.1)
    
    # Resultant Direction
    # Pull of Will (Normalized) minus Push of Enemy (Weighted)
    rx = (dx_w / dist_w) - (push_strength * (dx_n / dist_n if dist_n > 0 else 0))
    ry = (dy_w / dist_w) - (push_strength * (dy_n / dist_n if dist_n > 0 else 0))

    # 3. ELITE DISCRETIZATION
    # We discretize the ANGLE of the resultant vector
    angle = math.atan2(ry, rx)
    angle_bucket = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8 # 8 cardinal directions

    # State: [Resultant Angle, Threat Intensity, Diagonal Balance]
    threat = 2 if dist_n < 1.9 else (1 if dist_n < 2.8 else 0)
    balance = 1 if abs(dx_w) > abs(dy_w) else -1
    
    return (angle_bucket, threat, balance)

def create_universe1_agent():
    Q1 = {}
    # HYPER-SPEEDRUN PARAMETERS
    params = {
        "alpha": 0.1,        # Precise updates
        "gamma": 0.999,      # Absolute focus on the +500 win
        "eps": 1.0,
        "eps_decay": 0.985,  # Faster decay to solidify the fast path
        "eps_min": 0.01
    }

    def select_action(obs):
        state = extract_vector_features(obs)
        if state not in Q1: Q1[state] = np.zeros(4)
        
        if random.random() < params["eps"]:
            return random.randint(0, 3)
        
        # Tie-breaking with tiny noise
        return int(np.argmax(Q1[state] + np.random.randn(4) * 1e-7))

    def update(state, action, reward, next_state):
        if state not in Q1: Q1[state] = np.zeros(4)
        if next_state not in Q1: Q1[next_state] = np.zeros(4)

        # Standard Bellman with high Gamma
        best_next = np.max(Q1[next_state])
        td_target = reward + params["gamma"] * best_next
        Q1[state][action] += params["alpha"] * (td_target - Q1[state][action])
        
        # Anneal epsilon
        if params["eps"] > params["eps_min"]:
            params["eps"] *= params["eps_decay"]

    return select_action, update, Q1


def discretize_universe1(observation):
    return extract_vector_features(observation)

# UNIVERSE 2: ESCAPE ROOM
# ===============================
# ===============================
# UNIVERSE 2: ESCAPE ROOM
# ===============================

def discretize_universe2(observation):
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]
    guard_x, guard_y = observation[4], observation[5]

    dx_exit = exit_x - agent_x
    dy_exit = exit_y - agent_y

    dx_guard = guard_x - agent_x
    dy_guard = guard_y - agent_y

    dist_exit = distance(agent_x, agent_y, exit_x, exit_y)
    dist_guard = distance(agent_x, agent_y, guard_x, guard_y)

    # Exit buckets aligned to reward shaping
    if dist_exit < 1.5:
        exit_bucket = 0
    elif dist_exit < 2.0:
        exit_bucket = 1
    elif dist_exit < 5.0:
        exit_bucket = 2
    elif dist_exit < 12.0:
        exit_bucket = 3
    else:
        exit_bucket = 4

    # Guard buckets aligned to flashlight behavior
    if dist_guard < 1.5:
        guard_bucket = 0      # capture zone
    elif dist_guard < 3.0:
        guard_bucket = 1      # high danger
    elif dist_guard < 4.5:
        guard_bucket = 2      # flashlight danger
    elif dist_guard < 7.0:
        guard_bucket = 3      # near safe
    else:
        guard_bucket = 4      # safe

    in_flashlight = 1 if dist_guard < 4.5 else 0

    state = (
        sign(dx_exit),
        sign(dy_exit),
        exit_bucket,
        sign(dx_guard),
        sign(dy_guard),
        guard_bucket,
        in_flashlight
    )

    return state


def create_universe2_agent():
    Q2 = {}

    alpha = 0.08
    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.07
    epsilon_decay = 0.997

    def select_action(observation):
        nonlocal epsilon

        state = discretize_universe2(observation)

        if state not in Q2:
            Q2[state] = [0.5] * 4

        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(Q2[state]))

    def update(state, action, reward, next_state):
        nonlocal epsilon

        if state not in Q2:
            Q2[state] = [0.5] * 4
        if next_state not in Q2:
            Q2[next_state] = [0.5] * 4

        best_next = max(Q2[next_state])

        Q2[state][action] += alpha * (
            reward + gamma * best_next - Q2[state][action]
        )

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return select_action, update, Q2

import numpy as np
import math
import random

def extract_features_universe3(observation):
    # Observation: [ax, ay, c1x, c1y, c2x, c2y, vx, vy, c1res, c2res, px, py]
    a = observation[0:2]    # Agent
    c1 = observation[2:4]   # Child 1
    c2 = observation[4:6]   # Child 2
    v = observation[6:8]    # Vecna
    c1_res, c2_res = observation[8], observation[9]
    p = observation[10:12]  # Closest Projectile

    # 1. Goal Selection
    # If child 1 isn't rescued, it's the target. Otherwise, child 2.
    target = c1 if not c1_res else c2
    dist_target = np.linalg.norm(a - target)
    dist_v = np.linalg.norm(a - v)
    dist_p = np.linalg.norm(a - p)

    # 2. Advanced Feature Vector (12 features)
    f = np.zeros(12)
    f[0] = 1.0 # Bias
    
    # NAVIGATION: Direction to current target (Unit Vector)
    if dist_target > 0:
        f[1] = (target[0] - a[0]) / dist_target
        f[2] = (target[1] - a[1]) / dist_target
    
    # DANGER AVOIDANCE: Exponential "Heatmap"
    # These values spike to 1.0 when in the death zone, and drop to 0 quickly
    f[3] = math.exp(-max(0, dist_v - 2.5) / 2.0) # Vecna Danger
    f[4] = math.exp(-max(0, dist_p - 0.8) / 1.0) # Projectile Danger
    
    # DIRECTIONAL EVASION: Which way is the danger?
    # Tells the agent to move in the opposite direction of the threat
    f[5] = np.sign(a[0] - v[0]) if dist_v < 5.0 else 0
    f[6] = np.sign(a[1] - v[1]) if dist_v < 5.0 else 0
    f[7] = np.sign(a[0] - p[0]) if dist_p < 3.0 else 0
    f[8] = np.sign(a[1] - p[1]) if dist_p < 3.0 else 0
    
    # PROGRESS STATE:
    f[9] = float(c1_res)
    f[10] = float(c2_res)
    
    # STALL PREVENTION: Penalty for being far from target
    f[11] = dist_target / 20.0 

    return f
def discretize_universe3(observation):
    return extract_features_universe3(observation)
    
def create_universe3_agent():
    num_features = 12
    num_actions = 4
    weights = np.zeros((num_actions, num_features))
    
    # Optimized for a 200-episode sprint
    lr = 0.05
    gamma = 0.98 
    eps = {"val": 1.0, "decay": 0.99, "min": 0.02}

    def select_action(obs):
        f = extract_features_universe3(obs)
        if random.random() < eps["val"]:
            return random.randint(0, 3)
        return int(np.argmax(np.dot(weights, f)))

    def update(f, action, reward, next_f):
        # Q-Learning with Reward Normalization
        # We clip the reward to prevent the -200 from destroying the weights
        clipped_reward = np.clip(reward, -50, 200)
        
        q_curr = np.dot(weights[action], f)
        q_next = np.max(np.dot(weights, next_f))
        
        td_error = (clipped_reward + gamma * q_next) - q_curr
        
        # Weight Update
        weights[action] += lr * td_error * f
        
        # Decay exploration
        if eps["val"] > eps["min"]:
            eps["val"] *= eps["decay"]

    return select_action, update, weights

    
# ===============================
# ROUTER
# ===============================

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
        "agent_type": "Q-Learning",
        "multi_agent": True
    }