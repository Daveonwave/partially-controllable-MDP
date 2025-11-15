import gymnasium as gym
import numpy as np
import pickle
import time
import argparse
import os
import pygame
from pcmdp.utils import parameter_generator
from algo.utils import obs_to_key, build_state_index_map, get_state_size


def load_q_table(path):
    """Load a saved Q-table from a pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Q-table file not found: {path}")
    with open(path, "rb") as f:
        Q = pickle.load(f)
    return np.array(Q)


def choose_action(Q, state, epsilon=0.0):
    """Choose an action (greedy or epsilon-greedy)."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state, :])


def render_taxi(settings, Q, episodes=3, delay=0.5, epsilon=0.0, render_mode="human"):
    """Render the Taxi-v3 environment using the learned Q-table."""
    env = gym.make("pcmdp/taxi-v0", render_mode=render_mode, **{'settings': env_settings})
    
    keys, multipliers = build_state_index_map(env)
    get_indices = lambda obs, keys, multipliers, _no_batch: obs_to_key(obs, keys, multipliers, _no_batch)

    for ep in range(episodes):
        obs, _ = env.reset()
        obs_key = get_indices(obs, keys, multipliers, _no_batch=True)

        done = False
        cumulated_reward = 0
        steps = 0

        while not done:
            action = choose_action(Q, obs_key, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            new_obs_key = get_indices(next_obs, keys, multipliers, _no_batch=True)
            cumulated_reward += reward
            

            if render_mode == "human":
                # Render one frame
                frame = env.render()

                # Add overlay info via pygame
                pygame.display.set_caption(f"Taxi-v0 | Ep {ep+1} | Step {steps} | Reward {reward}")
                screen = pygame.display.get_surface()
                if screen is not None:
                    font = pygame.font.Font(None, 28)
                    text_surface = font.render(
                        f"Episode: {ep+1}   Step: {steps}   Reward: {reward}",
                        True,
                        (255, 255, 255)
                    )
                    screen.blit(text_surface, (10, 10))
                    pygame.display.flip()

                time.sleep(delay)

            steps += 1
            obs_key = new_obs_key
            done = terminated or truncated

        pygame.quit()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained Q-learning Taxi-v0 agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved Q-table pickle file.")
    parser.add_argument('--world', type=str, default='pcmdp/taxi/world.yaml', help="World configuration file.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (in seconds) between environment steps when rendering.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate for epsilon-greedy action selection during rendering.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "ansi"], help="Render mode: 'human' (pygame window) or 'ansi' (text-based).")

    args = parser.parse_args()

    # Load Q-table
    Q = load_q_table(args.model_path)
    print(f"Loaded Q-table with shape {Q.shape}")
    
    env_settings = parameter_generator(world_file=args.world)

    # Run rendering
    render_taxi(
        settings=env_settings,
        Q=Q,
        episodes=args.episodes,
        delay=args.delay,
        epsilon=args.epsilon,
        render_mode=args.render_mode,
    )
