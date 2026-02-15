"""
STRANGER THINGS - XODIA 3-UNIVERSE CHALLENGE
=============================================
Universe 1: Find Will in the Upside Down
Universe 2: Escape Room from Hawkins Lab
Universe 3: Fight Vecna and Save Children

"""

import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ====================================================================
# BASE UNIVERSE CLASS
# ====================================================================

class StrangerThingsUniverse:
    """Base class for all universes with HIGH-QUALITY rendering"""
    
    def __init__(self, name, description, screen_size=(1800, 1200), grid_size=(20, 20)):
        self.name = name
        self.description = description
        self.screen_size = screen_size  
        self.grid_size = grid_size
        self.cell_size = (screen_size[0] // grid_size[0], screen_size[1] // grid_size[1])
        
        self.agent_pos = np.array([1.0, 1.0])
        self.target_pos = np.array([18.0, 18.0])
        self.steps = 0
        self.max_steps = 300
        self.trajectory = []
        self.total_reward = 0.0
        self.last_action = 0
        
        self.enemies = []
        self.obstacles = []
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.script_dir, 'assets')
        
        self.backgrounds = self._load_backgrounds()
        self.sprites = self._load_sprites()
        
    def _load_backgrounds(self):
        """Override in subclass"""
        return {}
    
    def _load_sprites(self):
        """Override in subclass"""
        return {}
    
    def grid_to_screen(self, grid_pos):
        """Convert grid coordinates to screen pixels (centered)"""
        x = int(grid_pos[0] * self.cell_size[0])
        y = int((self.grid_size[1] - grid_pos[1]) * self.cell_size[1])
        return (x, y)
    
    def reset(self):
        """Reset universe to initial state"""
        self.agent_pos = np.array([1.0, 1.0])
        self.target_pos = np.array([18.0, 17.0])
        self.steps = 0
        self.trajectory = []
        self.total_reward = 0.0
        self.last_action = 0
        self._reset_enemies()
        return self.observe()
    
    def _reset_enemies(self):
        """Reset enemy positions"""
        pass
    
    def observe(self):
        """Return observation vector"""
        obs = [self.agent_pos[0], self.agent_pos[1], 
               self.target_pos[0], self.target_pos[1]]
        
        if self.enemies:
            distances = [np.linalg.norm(self.agent_pos - e['pos']) for e in self.enemies]
            nearest_idx = np.argmin(distances)
            obs.extend([self.enemies[nearest_idx]['pos'][0], 
                       self.enemies[nearest_idx]['pos'][1]])
        
        return np.array(obs)
    
    def _draw_sprite(self, img, sprite, position, size=80, rotation=0, alpha=255):
        """HIGH-QUALITY sprite drawing with NO quality loss"""
        if sprite is None:
            return
        
        screen_pos = self.grid_to_screen(position)
        
        # HIGH-QUALITY resize using LANCZOS resampling (best quality)
        resized = sprite.resize((size, size), Image.Resampling.LANCZOS)
        
        # Rotate if needed (high quality)
        if rotation != 0:
            resized = resized.rotate(-rotation, expand=False, resample=Image.Resampling.BICUBIC)
        
        # Alpha adjustment
        if alpha < 255:
            resized = resized.copy()
            if resized.mode != 'RGBA':
                resized = resized.convert('RGBA')
            resized.putalpha(alpha)
        
        # Center the sprite
        paste_x = screen_pos[0] - size // 2
        paste_y = screen_pos[1] - size // 2
        
        # Ensure within bounds
        paste_x = max(0, min(paste_x, self.screen_size[0] - size))
        paste_y = max(0, min(paste_y, self.screen_size[1] - size))
        
        # Paste with alpha blending for smooth edges
        if resized.mode == 'RGBA':
            img.paste(resized, (paste_x, paste_y), resized)
        else:
            img.paste(resized, (paste_x, paste_y))


# ====================================================================
# UNIVERSE 1: FIND WILL IN THE UPSIDE DOWN
# ====================================================================

class Universe1_FindWill(StrangerThingsUniverse):
    """
    STORY: Eleven must find and rescue Will trapped in the Upside Down
    ASSETS: 4 files in univ1/ folder
    """
    
    def __init__(self):
        super().__init__(
            name="👧 FIND WILL",
            description="Eleven searches for Will in the Upside Down"
        )
        
        self.agent_pos = np.array([2.0, 2.0])  # Eleven starts bottom-left
        self.target_pos = np.array([17.0, 12.0])  # Will WAY DOWN for FULL VISIBILITY
        self._reset_enemies()
        
    def _load_backgrounds(self):
        """Load Universe 1 background"""
        bg_path = os.path.join(self.assets_dir, 'univ1', 'uni1_bg.png')
        if os.path.exists(bg_path):
            return {'bg': Image.open(bg_path).convert('RGBA')}
        else:
            print(f"⚠️  Missing: {bg_path}")
            return {}
    
    def _load_sprites(self):
        """Load Universe 1 sprites (4 assets)"""
        sprites = {}
        sprite_files = {
            'agent': 'uni1_agent.png',      # Eleven
            'target': 'uni1_target.png',    # Will (trapped)
            'enemy': 'uni1_enemy.png'       # Demogorgon
        }
        
        for key, filename in sprite_files.items():
            path = os.path.join(self.assets_dir, 'univ1', filename)
            if os.path.exists(path):
                sprites[key] = Image.open(path).convert('RGBA')
                img=sprites[key]
                print(f"✅ Loaded {key}: {filename} | Size: {img.size} | Mode: {img.mode}")
            else:
                print(f"⚠️  Missing: {path}")
                sprites[key] = None
        
        return sprites
    
    def _reset_enemies(self):
        """6 Demogorgons wandering the Upside Down"""
        self.enemies = [
            {'pos': np.array([10.0, 10.0]), 'speed': 0.15, 'patrol_center': np.array([10.0, 10.0])},
            {'pos': np.array([15.0, 5.0]), 'speed': 0.12, 'patrol_center': np.array([15.0, 5.0])},
            {'pos': np.array([5.0, 15.0]), 'speed': 0.13, 'patrol_center': np.array([5.0, 15.0])},
            {'pos': np.array([8.0, 8.0]), 'speed': 0.14, 'patrol_center': np.array([8.0, 8.0])},
            {'pos': np.array([12.0, 16.0]), 'speed': 0.13, 'patrol_center': np.array([12.0, 16.0])},
            {'pos': np.array([17.0, 10.0]), 'speed': 0.14, 'patrol_center': np.array([17.0, 10.0])},
        ]
    
    def step(self, action):
        """Execute one step in Universe 1"""
        self.last_action = action
        new_pos = self.agent_pos + action
        new_pos[0] = np.clip(new_pos[0], 0.5, self.grid_size[0] - 0.5)
        new_pos[1] = np.clip(new_pos[1], 0.5, self.grid_size[1] - 0.5)
        
        self.agent_pos = new_pos
        self.steps += 1
        self.trajectory.append(self.agent_pos.copy())
        
        reward = 0.0
        done = False
        
        # Check if found Will
        distance_to_will = np.linalg.norm(self.agent_pos - self.target_pos)
        if distance_to_will < 1.5:
            reward += 500  # FOUND WILL!
            done = True
            return self.observe(), reward, done
        
        # Distance penalty (encourage moving toward Will)
        reward -= distance_to_will * 0.1
        
        # Move Demogorgons (wander in patrol areas)
        for enemy in self.enemies:
            if random.random() < 0.3:
                wander = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                enemy['pos'] += wander
                diff = enemy['patrol_center'] - enemy['pos']
                if np.linalg.norm(diff) > 3.0:
                    enemy['pos'] += diff * 0.1
        
        # Demogorgon collision penalty
        for enemy in self.enemies:
            if np.linalg.norm(self.agent_pos - enemy['pos']) < 1.8:
                reward -= 100
        
        reward -= 0.5  # Step penalty
        
        if self.steps >= self.max_steps:
            done = True
            reward -= 50
        
        self.total_reward += reward
        return self.observe(), reward, done
    
    def render_pil(self):
        """HIGH-QUALITY rendering for Universe 1"""
        # Background
        if self.backgrounds.get('bg'):
            img = self.backgrounds['bg'].resize(self.screen_size, Image.Resampling.LANCZOS).convert('RGBA')
            # Darken for Upside Down effect
            darkener = Image.new('RGBA', self.screen_size, (10, 5, 20, 120))
            img = Image.alpha_composite(img, darkener)
        else:
            img = Image.new('RGBA', self.screen_size, (15, 10, 25, 255))
        
        # Draw Will FIRST with glow (so he's visible)
        will_screen = self.grid_to_screen(self.target_pos)
        glow = Image.new('RGBA', self.screen_size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        for r in range(120, 40, -15):
            alpha = max(30, 255 - r * 2)
            glow_draw.ellipse([
                will_screen[0] - r, will_screen[1] - r,
                will_screen[0] + r, will_screen[1] + r
            ], fill=(255, 220, 100, alpha))
        img = Image.alpha_composite(img, glow)
        
        # Draw Will sprite (LARGE)
        self._draw_sprite(img, self.sprites.get('target'), self.target_pos, size=140)
        
        # Draw Demogorgons (LARGE)
        for enemy in self.enemies:
            self._draw_sprite(img, self.sprites.get('enemy'), enemy['pos'], size=220)
        
        # Draw Eleven on bike (LARGE, with rotation)
        rotation_map = {0: 0, 1: 90, 2: 180, 3: 270}
        if isinstance(self.last_action, np.ndarray):
            if self.last_action[0] > 0: action_idx = 0
            elif self.last_action[1] > 0: action_idx = 1
            elif self.last_action[0] < 0: action_idx = 2
            else: action_idx = 3
        else:
            action_idx = self.last_action
        
        rotation = rotation_map.get(action_idx, 0)
        self._draw_sprite(img, self.sprites.get('agent'), self.agent_pos, size=200, rotation=rotation)
        
        # HUD
        img_final = img.convert('RGB')
        draw = ImageDraw.Draw(img_final)
        
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
            font_stats = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            font_title = ImageFont.load_default()
            font_stats = ImageFont.load_default()
        
        draw.rectangle([0, 0, self.screen_size[0], 120], fill=(0, 0, 0, 220))
        draw.text((30, 30), "👧 FIND WILL IN THE UPSIDE DOWN", fill=(255, 100, 100), font=font_title)
        draw.text((30, 80), f"Steps: {self.steps}/{self.max_steps} | Reward: {self.total_reward:.1f}", 
                 fill=(200, 200, 200), font=font_stats)
        
        return img_final


# ====================================================================
# UNIVERSE 2: ESCAPE ROOM FROM HAWKINS LAB
# ====================================================================

class Universe2_EscapeRoom(StrangerThingsUniverse):
    """
    STORY: Steve, Robin & Dustin must escape the Russian-controlled lab
    ASSETS: 4 files in univ2/ folder (includes collision mask)
    """
    
    def __init__(self):
        super().__init__(
            name="🔬 ESCAPE ROOM",
            description="Escape the Russian-controlled Hawkins Lab"
        )
        
        # NO COLLISION MASK - simple open room gameplay
        self.collision_mask = None
        
        # CORRECT POSITIONS: Start TOP-LEFT, Exit BOTTOM-RIGHT (at gate door)
        self.agent_pos = np.array([2.0, 18.0])  # TOP-LEFT start
        self.target_pos = np.array([18.5, 2.5])  # EXIT at actual gate door (right+higher)
        self._reset_enemies()
    
    def _load_backgrounds(self):
        """Load Universe 2 background"""
        bg_path = os.path.join(self.assets_dir, 'univ2', 'univ2_bg.png')
        if os.path.exists(bg_path):
            return {'bg': Image.open(bg_path).convert('RGBA')}
        else:
            print(f"⚠️  Missing: {bg_path}")
            return {}
    
    def _load_sprites(self):
        """Load Universe 2 sprites (3 assets - no need to load mask again)"""
        sprites = {}
        sprite_files = {
            'agent': 'uni2_agent.png',      # Steve/Robin/Dustin group
            'enemy': 'uni2_enemy.png'       # Russian guard
        }
        
        for key, filename in sprite_files.items():
            path = os.path.join(self.assets_dir, 'univ2', filename)
            if os.path.exists(path):
                sprites[key] = Image.open(path).convert('RGBA')
            else:
                print(f"⚠️  Missing: {path}")
                sprites[key] = None
        
        return sprites
    
    def _reset_enemies(self):
        """7 Russian guards with varied patterns - bottom path blocked"""
        self.enemies = [
            # Top guards (facing DOWN - toward agent spawn)
            {
                'pos': np.array([5.0, 16.0]),
                'patrol_path': [np.array([5.0, 16.0]), np.array([10.0, 16.0])],
                'patrol_idx': 0,
                'speed': 0.22,
                'facing': 'down'
            },
            {
                'pos': np.array([17.5, 14.0]),
                'patrol_path': [np.array([17.5, 14.0]), np.array([17.5, 18.0])],
                'patrol_idx': 0,
                'speed': 0.24,
                'facing': 'down'
            },
            # Middle guards (horizontal patrol - blocking path)
            {
                'pos': np.array([6.0, 10.0]),
                'patrol_path': [np.array([6.0, 10.0]), np.array([14.0, 10.0])],
                'patrol_idx': 0,
                'speed': 0.26,
                'facing': 'down'
            },
        
            {
                'pos': np.array([8.0, 5.0]),
                'patrol_path': [np.array([8.0, 5.0]), np.array([13.0, 5.0])],
                'patrol_idx': 0,
                'speed': 0.25,
                'facing': 'up'
            },
            {
                'pos': np.array([15.0, 6.0]),
                'patrol_path': [np.array([15.0, 6.0]), np.array([15.0, 10.0])],
                'patrol_idx': 0,
                'speed': 0.23,
                'facing': 'up'
            },
        ]
    
    def reset(self):
        """Override to maintain correct start/exit positions"""
        self.agent_pos = np.array([2.0, 18.0])  # TOP-LEFT start (MUST STAY HERE!)
        self.target_pos = np.array([18.5, 2.5])  # EXIT at gate door (right+higher)
        self.steps = 0
        self.trajectory = []
        self.total_reward = 0.0
        self.last_action = 0
        self._reset_enemies()
        return self.observe()
    
    def step(self, action):
        """Execute one step in Universe 2"""
        self.last_action = action
        new_pos = self.agent_pos + action
        
        # No collision checking - simple movement
        reward = 0
        
        new_pos[0] = np.clip(new_pos[0], 0.5, self.grid_size[0] - 0.5)
        new_pos[1] = np.clip(new_pos[1], 0.5, self.grid_size[1] - 0.5)
        
        self.agent_pos = new_pos
        self.steps += 1
        self.trajectory.append(self.agent_pos.copy())
        
        done = False
        
        # Check if reached exit
        distance_to_exit = np.linalg.norm(self.agent_pos - self.target_pos)
        if distance_to_exit < 1.5:
            reward += 600  # ESCAPED!
            done = True
            return self.observe(), reward, done
        
        # Progressive reward shaping
        if distance_to_exit < 2.0:
            reward += (2.0 - distance_to_exit) * 50
        elif distance_to_exit < 5.0:
            reward += (5.0 - distance_to_exit) * 10
        else:
            reward -= distance_to_exit * 0.3
        
        # Move guards
        for enemy in self.enemies:
            target_waypoint = enemy['patrol_path'][enemy['patrol_idx']]
            direction = target_waypoint - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist < 0.5:
                enemy['patrol_idx'] = (enemy['patrol_idx'] + 1) % len(enemy['patrol_path'])
            else:
                direction = direction / dist * enemy['speed']
                enemy['pos'] += direction
        
        # Guard detection - INCREASED DIFFICULTY
        for enemy in self.enemies:
            dist_to_guard = np.linalg.norm(self.agent_pos - enemy['pos'])
            if dist_to_guard < 4.5:  # Larger flashlight range (was 3.5)
                penalty = 100 * (4.5 - dist_to_guard) / 4.5  # Higher penalty (was 80)
                reward -= penalty
                
            # Immediate capture penalty if too close
            if dist_to_guard < 1.5:
                reward -= 150  # Severe penalty for getting caught
        
        reward -= 0.8
        
        if distance_to_exit < 12.0:
            reward += 0.8
        
        if self.steps >= self.max_steps:
            done = True
            reward -= 100
        
        self.total_reward += reward
        return self.observe(), reward, done
    
    def render_pil(self):
        """HIGH-QUALITY rendering for Universe 2"""
        # Background
        if self.backgrounds.get('bg'):
            img = self.backgrounds['bg'].resize(self.screen_size, Image.Resampling.LANCZOS).convert('RGBA')
        else:
            img = Image.new('RGBA', self.screen_size, (40, 50, 70, 255))
        
        # EXIT marker with glow
        exit_screen = self.grid_to_screen(self.target_pos)
        glow = Image.new('RGBA', self.screen_size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        for r in range(100, 40, -15):
            alpha = max(40, 255 - r * 2)
            glow_draw.ellipse([
                exit_screen[0] - r, exit_screen[1] - r,
                exit_screen[0] + r, exit_screen[1] + r
            ], fill=(50, 255, 100, alpha))
        img = Image.alpha_composite(img, glow)
        
        # Flashlights for guards
        flashlight = Image.new('RGBA', self.screen_size, (0, 0, 0, 0))
        flash_draw = ImageDraw.Draw(flashlight)
        for enemy in self.enemies:
            enemy_screen = self.grid_to_screen(enemy['pos'])
            cone_length = 180
            cone_width = 100
            points = [
                enemy_screen,
                (enemy_screen[0] - cone_width, enemy_screen[1] + cone_length),
                (enemy_screen[0] + cone_width, enemy_screen[1] + cone_length)
            ]
            flash_draw.polygon(points, fill=(255, 255, 150, 60))
        img = Image.alpha_composite(img, flashlight)
        
        # Draw guards with rotation based on facing direction
        for enemy in self.enemies:
            rotation = 180 if enemy['facing'] == 'up' else 0  # Up=180, Down=0
            self._draw_sprite(img, self.sprites.get('enemy'), enemy['pos'], size=150, rotation=rotation)
        
        # Draw agent group (LARGE, with rotation)
        rotation_map = {0: 0, 1: 90, 2: 180, 3: 270}
        if isinstance(self.last_action, np.ndarray):
            if self.last_action[0] > 0: action_idx = 0
            elif self.last_action[1] > 0: action_idx = 1
            elif self.last_action[0] < 0: action_idx = 2
            else: action_idx = 3
        else:
            action_idx = self.last_action
        
        rotation = rotation_map.get(action_idx, 0)
        self._draw_sprite(img, self.sprites.get('agent'), self.agent_pos, size=190, rotation=rotation)
        
        # EXIT text
        img_final = img.convert('RGB')
        draw = ImageDraw.Draw(img_final)
        
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
            font_stats = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            font_exit = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        except:
            font_title = ImageFont.load_default()
            font_stats = ImageFont.load_default()
            font_exit = ImageFont.load_default()
        
        draw.text((exit_screen[0] - 60, exit_screen[1] - 30), "EXIT", 
                 fill=(0, 255, 0), font=font_exit, stroke_width=4, stroke_fill=(0, 100, 0))
        
        draw.rectangle([0, 0, self.screen_size[0], 120], fill=(0, 0, 0, 220))
        draw.text((30, 30), "🔬 ESCAPE ROOM - HAWKINS LAB", fill=(100, 200, 255), font=font_title)
        draw.text((30, 80), f"Steps: {self.steps}/{self.max_steps} | Reward: {self.total_reward:.1f}", 
                 fill=(200, 200, 200), font=font_stats)
        
        return img_final


# ====================================================================
# UNIVERSE 3: FIGHT VECNA
# ====================================================================

class Universe3_FightVecna(StrangerThingsUniverse):
    """
    STORY: Will must fight Vecna and save 2 trapped children while dodging projectiles
    ASSETS: 8 files in univ3/ folder
    """
    
    def __init__(self):
        super().__init__(
            name="⚡ FIGHT VECNA",
            description="Save the children from Vecna's realm"
        )
        
        self.agent_pos = np.array([10.0, 2.0])  # Will starts bottom center
        self.child1_pos = np.array([3.0, 10.0])  # Left child
        self.child2_pos = np.array([17.0, 10.0])  # Right child
        self.vecna_pos = np.array([10.0, 10.0])  # Vecna at center
        
        self.child1_rescued = False
        self.child2_rescued = False
        
        self.projectiles = []
        self.projectile_timer = 0
        self.projectile_interval = 60
        
        self._reset_enemies()
    
    def _load_backgrounds(self):
        """Load Universe 3 background"""
        bg_path = os.path.join(self.assets_dir, 'univ3', 'univ3_bg.png')
        if os.path.exists(bg_path):
            return {'bg': Image.open(bg_path).convert('RGBA')}
        else:
            print(f"⚠️  Missing: {bg_path}")
            return {}
    
    def _load_sprites(self):
        """Load Universe 3 sprites (7 assets)"""
        sprites = {}
        sprite_files = {
            'agent': 'univ3_agent.png',     # Will
            'child': 'kid1.png',            # Child 1 (use same for both)
            'child2': 'kid2.png',           # Child 2
            'vecna': 'Vecna.png',           # Vecna boss
            'demo': 'univ3_demo.png',       # Demogorgon
            'proj1': 'univ3_proj1.png',     # Projectile type 1
            'proj2': 'univ3_proj2.png'      # Projectile type 2
        }
        
        for key, filename in sprite_files.items():
            path = os.path.join(self.assets_dir, 'univ3', filename)
            if os.path.exists(path):
                sprites[key] = Image.open(path).convert('RGBA')
            else:
                print(f"⚠️  Missing: {path}")
                sprites[key] = None
        
        return sprites
    
    def _reset_enemies(self):
        """6 Demogorgons with MOVEMENT patterns around Vecna"""
        self.enemies = [
            # Original 4 in corners
            {
                'pos': np.array([7.0, 7.0]),
                'speed': 0.12,
                'patrol_center': np.array([7.0, 7.0]),
                'wander_radius': 4.5
            },
            {
                'pos': np.array([13.0, 7.0]),
                'speed': 0.10,
                'patrol_center': np.array([13.0, 7.0]),
                'wander_radius': 4.5
            },
            {
                'pos': np.array([7.0, 13.0]),
                'speed': 0.11,
                'patrol_center': np.array([7.0, 13.0]),
                'wander_radius': 4.5
            },
            {
                'pos': np.array([13.0, 13.0]),
                'speed': 0.13,
                'patrol_center': np.array([13.0, 13.0]),
                'wander_radius': 4.5
            },
        ]
    
    def reset(self):
        """Reset Universe 3"""
        self.agent_pos = np.array([10.0, 2.0])
        self.child1_rescued = False
        self.child2_rescued = False
        self.projectiles = []
        self.projectile_timer = 0
        self.steps = 0
        self.trajectory = []
        self.total_reward = 0.0
        self._reset_enemies()
        return self.observe()
    
    def observe(self):
        """Universe 3 has 12 observation values"""
        obs = [
            self.agent_pos[0], self.agent_pos[1],
            self.child1_pos[0], self.child1_pos[1],
            self.child2_pos[0], self.child2_pos[1],
            self.vecna_pos[0], self.vecna_pos[1],
            1.0 if self.child1_rescued else 0.0,
            1.0 if self.child2_rescued else 0.0
        ]
        
        if self.projectiles:
            obs.extend([self.projectiles[0]['pos'][0], self.projectiles[0]['pos'][1]])
        else:
            obs.extend([0.0, 0.0])
        
        return np.array(obs)
    
    def step(self, action):
        """Execute one step in Universe 3"""
        self.last_action = action
        new_pos = self.agent_pos + action
        new_pos[0] = np.clip(new_pos[0], 0.5, self.grid_size[0] - 0.5)
        new_pos[1] = np.clip(new_pos[1], 0.5, self.grid_size[1] - 0.5)
        
        self.agent_pos = new_pos
        self.steps += 1
        self.trajectory.append(self.agent_pos.copy())
        
        reward = 0.0
        done = False
        
        # Check child rescues
        if not self.child1_rescued:
            if np.linalg.norm(self.agent_pos - self.child1_pos) < 1.5:
                self.child1_rescued = True
                reward += 200
        
        if not self.child2_rescued:
            if np.linalg.norm(self.agent_pos - self.child2_pos) < 1.5:
                self.child2_rescued = True
                reward += 200
        
        # Win condition
        if self.child1_rescued and self.child2_rescued:
            reward += 300
            done = True
            return self.observe(), reward, done
        
        # Vecna death zone
        if np.linalg.norm(self.agent_pos - self.vecna_pos) < 2.5:
            reward -= 200
            done = True
            return self.observe(), reward, done
        
        # Move Demogorgons (wander around patrol centers)
        for enemy in self.enemies:
            # Random wander movement
            if random.random() < 0.4:  # 40% chance to move each step
                wander = np.array([random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)])
                enemy['pos'] += wander
                
                # Pull back toward patrol center if too far
                diff = enemy['patrol_center'] - enemy['pos']
                distance = np.linalg.norm(diff)
                if distance > enemy['wander_radius']:
                    enemy['pos'] += diff * 0.15  # Pull back
                
                # Keep in bounds
                enemy['pos'][0] = np.clip(enemy['pos'][0], 2.0, 18.0)
                enemy['pos'][1] = np.clip(enemy['pos'][1], 2.0, 18.0)
        
        # Demogorgon zones
        for enemy in self.enemies:
            if np.linalg.norm(self.agent_pos - enemy['pos']) < 1.5:
                reward -= 80
        
        # Projectile system
        self.projectile_timer += 1
        if self.projectile_timer >= self.projectile_interval:
            self._fire_projectile()
            self.projectile_timer = 0
        
        for proj in self.projectiles[:]:
            proj['pos'] += proj['velocity'] * proj['speed']
            
            if np.linalg.norm(self.agent_pos - proj['pos']) < 0.8:
                reward -= 150
                done = True
                return self.observe(), reward, done
            
            if (proj['pos'][0] < 0 or proj['pos'][0] > 20 or 
                proj['pos'][1] < 0 or proj['pos'][1] > 20):
                self.projectiles.remove(proj)
        
        # Encourage progress
        if not self.child1_rescued:
            reward -= np.linalg.norm(self.agent_pos - self.child1_pos) * 0.05
        if not self.child2_rescued:
            reward -= np.linalg.norm(self.agent_pos - self.child2_pos) * 0.05
        
        reward -= 1.0
        
        if self.steps >= self.max_steps:
            done = True
            reward -= 100
        
        self.total_reward += reward
        return self.observe(), reward, done
    
    def _fire_projectile(self):
        """Fire projectile toward agent"""
        direction = self.agent_pos - self.vecna_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            velocity = direction / distance
        else:
            velocity = np.array([1.0, 0.0])
        
        # Alternate between projectile types
        proj_type = len(self.projectiles) % 2
        self.projectiles.append({
            'pos': self.vecna_pos.copy(),
            'velocity': velocity,
            'speed': 0.3,
            'type': proj_type
        })
    
    def render_pil(self):
        """HIGH-QUALITY rendering for Universe 3"""
        # Background
        if self.backgrounds.get('bg'):
            img = self.backgrounds['bg'].resize(self.screen_size, Image.Resampling.LANCZOS).convert('RGBA')
        else:
            img = Image.new('RGBA', self.screen_size, (80, 20, 20, 255))
        
        # Draw children with glow if not rescued
        for child_pos, rescued, child_sprite in [
            (self.child1_pos, self.child1_rescued, 'child'),
            (self.child2_pos, self.child2_rescued, 'child2')
        ]:
            if not rescued:
                child_screen = self.grid_to_screen(child_pos)
                glow = Image.new('RGBA', self.screen_size, (0, 0, 0, 0))
                glow_draw = ImageDraw.Draw(glow)
                for r in range(80, 30, -10):
                    alpha = max(30, 200 - r * 2)
                    glow_draw.ellipse([
                        child_screen[0] - r, child_screen[1] - r,
                        child_screen[0] + r, child_screen[1] + r
                    ], fill=(100, 200, 255, alpha))
                img = Image.alpha_composite(img, glow)
            
            self._draw_sprite(img, self.sprites.get(child_sprite), child_pos, size=150)
        
        # Draw Vecna (LARGE and menacing)
        self._draw_sprite(img, self.sprites.get('vecna'), self.vecna_pos, size=260)
        
        # Draw Demogorgons
        for enemy in self.enemies:
            self._draw_sprite(img, self.sprites.get('demo'), enemy['pos'], size=130)
        
        # Draw projectiles (alternate types)
        for proj in self.projectiles:
            sprite_key = 'proj1' if proj['type'] == 0 else 'proj2'
            self._draw_sprite(img, self.sprites.get(sprite_key), proj['pos'], size=60)
        
        # Draw agent (Will) with rotation
        rotation_map = {0: 0, 1: 90, 2: 180, 3: 270}
        if isinstance(self.last_action, np.ndarray):
            if self.last_action[0] > 0: action_idx = 0
            elif self.last_action[1] > 0: action_idx = 1
            elif self.last_action[0] < 0: action_idx = 2
            else: action_idx = 3
        else:
            action_idx = self.last_action
        
        rotation = rotation_map.get(action_idx, 0)
        self._draw_sprite(img, self.sprites.get('agent'), self.agent_pos, size=190, rotation=rotation)
        
        # HUD
        img_final = img.convert('RGB')
        draw = ImageDraw.Draw(img_final)
        
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
            font_stats = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            font_title = ImageFont.load_default()
            font_stats = ImageFont.load_default()
        
        draw.rectangle([0, 0, self.screen_size[0], 120], fill=(0, 0, 0, 220))
        draw.text((30, 30), "⚡ FIGHT VECNA - SAVE THE CHILDREN!", fill=(255, 200, 50), font=font_title)
        rescued_text = f"Rescued: {sum([self.child1_rescued, self.child2_rescued])}/2"
        draw.text((30, 80), f"Steps: {self.steps}/{self.max_steps} | {rescued_text} | Reward: {self.total_reward:.1f}", 
                 fill=(200, 200, 200), font=font_stats)
        
        return img_final


# ====================================================================
# ACTIONS & REGISTRY
# ====================================================================

ACTIONS = [
    np.array([0.5, 0.0]),   # 0: Right
    np.array([0.0, 0.5]),   # 1: Up
    np.array([-0.5, 0.0]),  # 2: Left
    np.array([0.0, -0.5])   # 3: Down
]

def get_all_universes():
    """Return all 3 universes"""
    return [
        Universe1_FindWill(),
        Universe2_EscapeRoom(),
        Universe3_FightVecna()
    ]

if __name__ == "__main__":
    print("🎮 STRANGER THINGS XODIA - 3 UNIVERSES")
    print("=" * 60)
    
    universes = get_all_universes()
    for i, u in enumerate(universes, 1):
        print(f"\n{i}. {u.name}")
        print(f"   {u.description}")
        print(f"   Start: {u.agent_pos}")
        print(f"   Observation Size: {len(u.observe())}")
