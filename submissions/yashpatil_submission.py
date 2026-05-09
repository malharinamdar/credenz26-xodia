"""
STRANGER THINGS XODIA - SUBMISSION TEMPLATE
============================================
Team: CODENSTEIN

TASK: Create 3 separate RL agents (one per universe)
- Universe 1: Find Will (6 Demogorgons)
- Universe 2: Escape Room (5 Guards)
- Universe 3: Fight Vecna (6 Demogorgons + Vecna)
"""

#UNIVERSE1:

import pygame
import numpy as np
import random
from collections import defaultdict

# --- CONFIGURATION ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 500
GRID_SIZE = 20
CELL_W = SCREEN_WIDTH / GRID_SIZE
CELL_H = SCREEN_HEIGHT / GRID_SIZE

class Universe1Env:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Universe 1: Eleven Rescues Will")
        self.clock = pygame.time.Clock()
        
        # Load and scale assets
        self.bg = pygame.transform.scale(pygame.image.load('uni1_bg.jpg'), (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.agent_img = pygame.transform.scale(pygame.image.load('uni1_agent.png'), (50, 50))
        self.enemy_img = pygame.transform.scale(pygame.image.load('uni1_enemy.png'), (60, 60))
        self.target_img = pygame.transform.scale(pygame.image.load('uni1_target.jpg'), (70, 70))
        
        self.reset()

    def reset(self):
        # Start: (2, 2) bottom-left. In Pygame Y=0 is top, so we invert Y.
        self.agent_pos = np.array([2.0, 17.0]) 
        self.will_pos = np.array([17.0, 7.0]) # Goal at (17, 12) from bottom
        
        # 6 Demogorgons with wandering patrol
        self.enemies = [np.array([random.uniform(5, 18), random.uniform(2, 15)]) for _ in range(6)]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # Find nearest demogorgon
        dists = [np.linalg.norm(self.agent_pos - e) for e in self.enemies]
        nearest_idx = np.argmin(dists)
        # Observation (6 values): [agent_x, agent_y, will_x, will_y, nearest_demo_x, nearest_demo_y]
        return np.array([*self.agent_pos, *self.will_pos, *self.enemies[nearest_idx]])

    def step(self, action):
        # 0: Right, 1: Up, 2: Left, 3: Down
        moves = {0: [0.5, 0], 1: [0, -0.5], 2: [-0.5, 0], 3: [0, 0.5]}
        self.agent_pos += np.array(moves[action])
        self.agent_pos = np.clip(self.agent_pos, 0, GRID_SIZE - 1)

        # Move enemies (wandering patrol)
        for i in range(len(self.enemies)):
            self.enemies[i] += np.random.uniform(-0.3, 0.3, size=2)
            self.enemies[i] = np.clip(self.enemies[i], 2, GRID_SIZE - 2)

        self.steps += 1
        obs = self._get_obs()
        dist_will = np.linalg.norm(self.agent_pos - self.will_pos)
        dist_enemy = np.linalg.norm(self.agent_pos - obs[4:6])

        # Rewards Logic
        reward = -0.5  # Per step penalty
        reward -= 0.1 * dist_will # Progress penalty
        done = False

        if dist_will < 1.5:
            reward += 500
            done = True
        elif dist_enemy < 1.8:
            reward -= 100
            done = True
        elif self.steps >= 300:
            done = True

        return obs, reward, done

    def render(self, ep_info=""):
        self.screen.blit(self.bg, (0, 0))
        
        # Draw Will (Goal)
        wx, wy = self.will_pos * [CELL_W, CELL_H]
        self.screen.blit(self.target_img, (wx - 35, wy - 35))
        
        # Draw Demogorgons
        for en in self.enemies:
            ex, ey = en * [CELL_W, CELL_H]
            self.screen.blit(self.enemy_img, (ex - 30, ey - 30))
            
        # Draw Eleven (Agent)
        ax, ay = self.agent_pos * [CELL_W, CELL_H]
        self.screen.blit(self.agent_img, (ax - 25, ay - 25))

        pygame.display.flip()

# --- RL BOT LOGIC ---
def discretize(obs):
    # Simplify state for Q-table: relative direction to Will and Danger proximity
    ax, ay, wx, wy, ex, ey = obs
    rel_x = 1 if wx > ax + 0.5 else -1 if wx < ax - 0.5 else 0
    rel_y = 1 if wy > ay + 0.5 else -1 if wy < ay - 0.5 else 0
    
    dist_en = np.linalg.norm([ax-ex, ay-ey])
    danger = 1 if dist_en < 3.0 else 0
    
    return (rel_x, rel_y, danger)

def run_bot():
    env = Universe1Env()
    Q_table = defaultdict(lambda: np.zeros(4))
    alpha, gamma, epsilon = 0.2, 0.95, 1.0
    
    for ep in range(500):
        obs = env.reset()
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); return

            state = discretize(obs)
            # Epsilon-greedy selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q_table[state])

            next_obs, reward, done = env.step(action)
            next_state = discretize(next_obs)

            # Q-Learning Update
            best_next = np.max(Q_table[next_state])
            Q_table[state][action] += alpha * (reward + gamma * best_next - Q_table[state][action])
            
            obs = next_obs
            
            # Slow down rendering for the final trained episodes
            if ep > 480:
                env.render()
                env.clock.tick(20)
            elif ep % 50 == 0:
                env.render() # Visual check every 50 episodes

        epsilon = max(0.01, epsilon * 0.99)
        if ep % 50 == 0:
            print(f"Episode {ep} | States Learned: {len(Q_table)}")

    # Final Evaluation
    print("Training Finished. Watching final escape...")
    while True:
        obs = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
            action = np.argmax(Q_table[discretize(obs)])
            obs, _, done = env.step(action)
            env.render()
            env.clock.tick(15)

if __name__ == "__main__":
    run_bot()


UNIVERSE2:

import pygame
import numpy as np
import random
import os
from collections import defaultdict

# --- ASSET LOADING ---
# Use absolute paths to prevent "File Not Found" errors
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_img(name, size):
    path = os.path.join(BASE_PATH, name)
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, size)

# --- Q-LEARNING AGENT ---
class HawkinsAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = 1.0
        self.alpha = 0.08
        self.gamma = 0.97

    def discretize(self, obs):
        ax, ay, ex, ey, gx, gy = obs
        dist_exit = np.sqrt((ex-ax)**2 + (ey-ay)**2)
        dist_guard = np.sqrt((gx-ax)**2 + (gy-ay)**2)
        dx = 1 if ex > ax + 0.5 else (-1 if ex < ax - 0.5 else 0)
        dy = 1 if ey > ay + 0.5 else (-1 if ey < ay - 0.5 else 0)
        danger = 0 if dist_guard > 4.5 else (1 if dist_guard >= 2.0 else 2)
        return (int(dist_exit/1.5), int(dist_guard/1.0), dx, dy, danger)

    def select_action(self, obs):
        state = self.discretize(obs)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

# --- THE GAME ---
def run_game():
    pygame.init()
    screen = pygame.display.set_mode((1000, 650))
    pygame.display.set_caption("Universe 2: Escape Hawkins Lab")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    # Load your assets
    bg = load_img("univ2_bg.png", (1000, 600))
    agent_img = load_img("uni2_agent.png", (60, 60))
    enemy_img = load_img("uni2_enemy.png", (50, 50))

    agent = HawkinsAgent()
    
    # Environment Setup
    agent_pos = np.array([2.0, 18.0])
    exit_pos = np.array([18.5, 2.5])
    guards = [np.array([10.0, 10.0]), np.array([5.0, 5.0]), np.array([15.0, 15.0])]
    
    episode = 1
    running = True

    while running:
        # Reset Episode
        agent_pos = np.array([2.0, 18.0])
        total_reward = 0
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            # Get Observation
            nearest_g = guards[0] # Simplified for demo
            obs = [*agent_pos, *exit_pos, *nearest_g]
            
            # Action
            action = agent.select_action(obs)
            move = {0:[0.5,0], 1:[0,-0.5], 2:[-0.5,0], 3:[0,0.5]}[action]
            agent_pos = np.clip(agent_pos + move, 0, 19)

            # Check Collisions / Goals
            dist_exit = np.linalg.norm(agent_pos - exit_pos)
            dist_guard = np.linalg.norm(agent_pos - nearest_g)
            
            reward = -0.5
            if dist_exit < 1.5: reward = 600; done = True
            if dist_guard < 1.5: reward = -200; done = True

            # Update Q-Table (Internal Logic)
            state = agent.discretize(obs)
            agent.q_table[state][action] += agent.alpha * (reward - agent.q_table[state][action])
            total_reward += reward

            # --- DRAW ---
            screen.blit(bg, (0, 0))
            # Draw Guard Zones (Visualizing the Danger state)
            for g in guards:
                pygame.draw.circle(screen, (255, 0, 0, 30), (int(g[0]*50), int(g[1]*30)), 70)
                screen.blit(enemy_img, (g[0]*50-25, g[1]*30-25))
            
            screen.blit(agent_img, (agent_pos[0]*50-30, agent_pos[1]*30-30))
            
            # Dashboard
            pygame.draw.rect(screen, (30, 30, 40), (0, 600, 1000, 50))
            status = f"EPISODE: {episode} | EPSILON: {agent.epsilon:.2f} | REWARD: {int(total_reward)}"
            screen.blit(font.render(status, True, (255, 255, 255)), (20, 615))

            pygame.display.flip()
            clock.tick(60)

        agent.epsilon = max(0.05, agent.epsilon * 0.993)
        episode += 1

    pygame.quit()

if __name__ == "__main__":
    run_game()


UNIVERSE3:

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class VecnaRescueEnv(gym.Env):
    def __init__(self):
        super(VecnaRescueEnv, self).__init__()
        
        # Grid Setup
        self.grid_size = 20
        self.max_steps = 300
        
        # Observation space (12 values as defined)
        # Low/High bounds for [x, y, c1x, c1y, c2x, c2y, vx, vy, r1, r2, px, py]
        self.observation_space = spaces.Box(low=0, high=20, shape=(12,), dtype=np.float32)
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([10.0, 2.0])
        self.child1_pos = np.array([3.0, 10.0])
        self.child2_pos = np.array([17.0, 10.0])
        self.vecna_pos = np.array([10.0, 15.0]) # Starting position for Vecna
        self.projectile_pos = np.array([10.0, 15.0]) # Projectile starts at Vecna
        
        self.child1_rescued = 0
        self.child2_rescued = 0
        self.steps = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            *self.agent_pos, *self.child1_pos, *self.child2_pos,
            *self.vecna_pos, self.child1_rescued, self.child2_rescued,
            *self.projectile_pos
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -1.0 # Time penalty
        terminated = False
        truncated = False

        # 1. Move Agent
        move_map = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0], 4: [0, 0]}
        move = np.array(move_map[action])
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.grid_size)

        # 2. Update Projectile (Tracks Agent)
        proj_dir = self.agent_pos - self.projectile_pos
        dist_proj = np.linalg.norm(proj_dir)
        if dist_proj > 0:
            self.projectile_pos += (proj_dir / dist_proj) * 0.8 # Projectile speed

        # 3. Check Rescues
        if not self.child1_rescued and np.linalg.norm(self.agent_pos - self.child1_pos) < 1.0:
            self.child1_rescued = 1
            reward += 200
        if not self.child2_rescued and np.linalg.norm(self.agent_pos - self.child2_pos) < 1.0:
            self.child2_rescued = 1
            reward += 200

        # 4. Check Death Conditions
        dist_vecna = np.linalg.norm(self.agent_pos - self.vecna_pos)
        dist_hit_proj = np.linalg.norm(self.agent_pos - self.projectile_pos)

        if dist_vecna < 2.5:
            reward -= 200
            terminated = True
        elif dist_hit_proj < 0.8:
            reward -= 150
            terminated = True

        # 5. Progress Penalties
        if not self.child1_rescued:
            reward -= 0.05 * np.linalg.norm(self.agent_pos - self.child1_pos)
        if not self.child2_rescued:
            reward -= 0.05 * np.linalg.norm(self.agent_pos - self.child2_pos)

        # 6. Win Condition
        if self.child1_rescued and self.child2_rescued:
            reward += 300
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}