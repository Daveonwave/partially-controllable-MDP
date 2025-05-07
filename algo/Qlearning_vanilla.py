from pcmdp.simulator.passenger import generate_arrival_distribution
from pcmdp.env import ElevatorEnv
from rich.pretty import pprint
from gymnasium import spaces
from collections import defaultdict
import numpy as np
import random

    
def q_learning(env):
    # Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.05
    n_episodes = 100
        
    actions_size = env.action_space.n    
    Q = defaultdict(lambda: np.zeros(actions_size))
        
    # Helper to convert an obs dict to a hashable flattened tuple
    def obs_to_key(obs):
        return tuple(spaces.flatten(env.observation_space, obs))
        
    for episode in range(n_episodes):
        obs, _ = env.reset()
        obs_key = obs_to_key(obs)
        
        cumulated_reward = 0        
        
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[obs_key])     # Exploit learned values
            
            if action != 0 and action != 1 and action != 2:
                print(f"Invalid action {action}")
            
            new_obs, reward, terminated, truncated, info = env.step(action) 
            new_obs_key = obs_to_key(new_obs)
            
            cumulated_reward += reward
            done = terminated or truncated
            
            # Q-update
            best_next_action = np.argmax(Q[new_obs_key])
            td_target = reward + gamma * Q[new_obs_key][best_next_action]
            Q[obs_key][action] += alpha * (td_target - Q[obs_key][action])

            obs, obs_key = new_obs, new_obs_key
            
            # Print the current state and action
            #pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, "
            #       f"Reward: {reward}, Cumulated Reward: {cumulated_reward}, "
            #       f"Current Pos: {obs['current_position']}, Passengers: {obs['n_passengers']}, "
            #       f"Queues: {obs['floor_queues']}, Arrivals: {obs['arrivals']}")
            
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode} finished with cumulated reward: {cumulated_reward}")  
        
    print("✅ Training complete!")
    return Q


            


