"""
STRANGER THINGS XODIA - COMPETITION RUNNER 
=================================================================
Handles submissions with SEPARATE agents for each universe.

Usage:
    python competition_runner.py <submission.py> --visualize
"""

import sys
import os
import importlib.util
import numpy as np
import argparse
import json
import glob
import time

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("  Warning: opencv-python not found. Install with: pip install opencv-python")

from realm_engine import get_all_universes, ACTIONS


# ---------------------------------------------------------------------
# Agent Loader
# ---------------------------------------------------------------------
def load_participant_agent(filepath):
    """Load participant's multi-agent submission"""
    spec = importlib.util.spec_from_file_location("participant_agent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# Dashboard (same as before)
# ---------------------------------------------------------------------
def create_dashboard(game_frame, universe_name, episode, max_episodes, reward, best_reward):
    if not OPENCV_AVAILABLE:
        return None
    
    GAME_WIDTH = 1400   # was 800 → wider game
    GAME_HEIGHT = 900

    PANEL_WIDTH = 320   # was 400 → thinner stats

    game_bgr = cv2.cvtColor(np.array(game_frame), cv2.COLOR_RGB2BGR)
    game_bgr = cv2.resize(game_bgr, (GAME_WIDTH, GAME_HEIGHT))

    panel = np.zeros((GAME_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    panel[:] = (20, 15, 30)
    
    def put_text(img, text, y, size=0.8, color=(200, 200, 200), thickness=2):
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 120), (40, 20, 50), -1)

    put_text(panel, "STRANGER THINGS", 45, 1.0, (255, 100, 100), 3)
    put_text(panel, "XODIA CHALLENGE", 85, 0.9, (200, 100, 150), 2)
    
    y = 180
    put_text(panel, "UNIVERSE:", y, 0.6, (150, 150, 150), 1)
    clean_name = universe_name.replace('👧', '').replace('🚲', '').replace('🔬', '').replace('⚡', '').strip()
    put_text(panel, clean_name[:25], y+35, 0.7, (255, 255, 255), 2)
    
    y = 300
    put_text(panel, "EPISODE:", y, 0.6, (150, 150, 150), 1)
    put_text(panel, f"{episode} / {max_episodes}", y+35, 1.1, (255, 200, 0), 3)
    
    bar_width = PANEL_WIDTH - 60

    progress = episode / max_episodes
    cv2.rectangle(panel, (30, y+55), (30+bar_width, y+65), (50, 40, 60), -1)
    cv2.rectangle(panel, (30, y+55), (30+int(bar_width*progress), y+65), (100, 255, 100), -1)
    
    y = 440
    put_text(panel, "CURRENT REWARD:", y, 0.6, (150, 150, 150), 1)
    color = (50, 50, 255) if reward < 0 else (50, 255, 50)
    put_text(panel, f"{reward:.1f}", y+35, 1.3, color, 3)
    
    y = 560
    put_text(panel, "BEST RECORD:", y, 0.6, (150, 150, 150), 1)
    put_text(panel, f"{best_reward:.1f}", y+35, 1.3, (0, 200, 255), 3)
    
    put_text(panel, "Press ESC to skip", 750, 0.5, (100, 100, 100), 1)
    
    dashboard = np.hstack((game_bgr, panel))
    return dashboard


# ---------------------------------------------------------------------
# Training (MULTI-AGENT VERSION)
# ---------------------------------------------------------------------
def train_universe_live(
    universe,
    select_action,
    update,
    discretize_func,
    episodes=200,
    visualize=False
):
    """Train agent on a single universe"""
    
    episode_rewards = []
    best_reward = -float("inf")
    best_trajectory = []
    
    print(f"\n{'=' * 70}")
    print(f"Training in {universe.name} ({episodes} episodes)")
    print(f"{'=' * 70}")

    window_name = "Stranger Things - XODIA Challenge"
    if visualize and OPENCV_AVAILABLE:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1720, 900)  # Fixed size

    
    def should_show(ep):
        # Show last 3 episodes in full
        if ep > episodes - 3:
            return True
        # Show every 10th episode
        elif ep % 10 == 0:
            return True
        else:
            return False

    for episode in range(1, episodes + 1):
        show_episode = visualize and OPENCV_AVAILABLE and should_show(episode)
        
        obs = universe.reset()
        total_reward = 0.0
        done = False
        step_counter = 0

        while not done:
            step_counter += 1
            
            # Agent selects action
            state = discretize_func(obs)
            action_idx = select_action(obs)
            action = ACTIONS[action_idx]
            next_obs, reward, done = universe.step(action)
            
            # Agent updates
            next_state = discretize_func(next_obs)
            update(state, action_idx, reward, next_state)
            
            obs = next_obs
            total_reward += reward

            # Visualization
            if show_episode:
                # Show every step for last 3 episodes, every 3rd step otherwise
                if episode > episodes - 3:
                    should_render = True
                else:
                    should_render = (step_counter % 3 == 0 or done)
                
                if should_render:
                    pil_img = universe.render_pil()
                    dashboard = create_dashboard(
                        pil_img, universe.name, episode, episodes, total_reward, best_reward
                    )
                    
                    if dashboard is not None:
                        cv2.imshow(window_name, dashboard)
                        # Longer delay for last episodes to see what's happening
                        if episode > episodes - 3:
                            delay = 15 if done else 5
                        else:
                            delay = 10 if done else 1
                        key = cv2.waitKey(delay)
                        if key == 27:
                            visualize = False
                            cv2.destroyAllWindows()
                            print("\n Visualization skipped by user.")

        episode_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = universe.trajectory.copy()

        if episode % max(1, episodes // 5) == 0:
            print(f"   Ep {episode:3d}/{episodes}: Reward={total_reward:6.1f} | Best={best_reward:6.1f}")

    return best_reward, episode_rewards


# ---------------------------------------------------------------------
# Main Competition Runner (MULTI-AGENT VERSION)
# ---------------------------------------------------------------------
def run_full_competition(participant_file, episodes_per_universe=200, visualize=False):
    print("\n" + "=" * 80)
    print("STRANGER THINGS XODIA - COMPETITION EVALUATION")
    print("=" * 80)
    print(f"Participant: {participant_file}")
    print("=" * 80)

    try:
        participant = load_participant_agent(participant_file)
        
        # Check if has multi-agent interface
        if not hasattr(participant, 'get_agent_for_universe'):
            print(" ERROR: Submission must have 'get_agent_for_universe()' function!")
            print("   Each universe needs its own agent.")
            return None
        
        agent_info = participant.get_agent_info()
        print("\n Agent Configuration:")
        for k, v in agent_info.items():
            print(f"   {k}: {v}")

    except Exception as e:
        print(f" Error loading participant agent: {e}")
        import traceback
        traceback.print_exc()
        return None

    universes = get_all_universes()
    total_score = 0.0
    universe_scores = []
    all_rewards = {}

    for universe in universes:
        try:
            print(f"\n{'=' * 70}")
            print(f"Loading agent for: {universe.name}")
            print(f"{'=' * 70}")
            
            # Get the SPECIFIC agent for this universe
            (select_action, update, memory), discretize_func = participant.get_agent_for_universe(universe.name)
            
            print(f"Agent loaded: {type(memory).__name__}")
            print(f"Discretize function: {discretize_func.__name__}")
            
            best_reward, episode_rewards = train_universe_live(
                universe,
                select_action,
                update,
                discretize_func,
                episodes=episodes_per_universe,
                visualize=visualize
            )

            universe_scores.append(best_reward)
            all_rewards[universe.name] = episode_rewards
            total_score += best_reward

            print(f"    {universe.name} Complete! Best Score: {best_reward:.2f}")
            
            if visualize and OPENCV_AVAILABLE:
                time.sleep(1.5)

        except Exception as e:
            print(f" Error in {universe.name}: {e}")
            import traceback
            traceback.print_exc()
            universe_scores.append(-1000.0)

    # Save results
    results = {
        "participant": participant_file,
        "total_score": total_score,
        "universe_scores": universe_scores,
        "universe_names": [u.name for u in universes],
        "agent_info": agent_info,
        "all_episode_rewards": all_rewards
    }

    if visualize and OPENCV_AVAILABLE:
        cv2.destroyAllWindows()
        
    # Final summary
    print("\n" + "=" * 80)
    print("📈 FINAL RESULTS")
    print("=" * 80)
    for universe, score in zip(universes, universe_scores):
        print(f"   {universe.name}: {score:.2f}")
    print(f"\n    TOTAL SCORE: {total_score:.2f}")
    print("=" * 80)
        
    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stranger Things XODIA Competition (Multi-Agent)")
    parser.add_argument("submission", nargs="?", help="Path to participant submission file")
    parser.add_argument("--all", help="Directory containing all submissions")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per universe")
    parser.add_argument("--visualize", action="store_true", help="Show live visualization")

    args = parser.parse_args()

    print("Usage:")
    print("  python competition_runner.py <submission.py> --visualize")
    print("  python competition_runner.py --all submissions/ --episodes 200")
