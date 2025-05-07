from pcmdp.simulator.passenger import generate_arrival_distribution
from pcmdp.env import ElevatorEnv
from rich.pretty import pprint
from gymnasium import spaces
from collections import defaultdict
from copy import deepcopy
import numpy as np
import itertools
import random


    
def augmented_q_learning(env):
    # Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    n_episodes = 1
        
    # Helper to convert an obs dict to a hashable flattened tuple
    def obs_to_key(obs):
        return tuple(spaces.flatten(env.observation_space, obs))
    
    def controllable_space():
        """
        Generates all the combinations of the controllable variables within the observation space

        Returns:
            list: A list of dicts 
        """
        keys_to_include = ['current_position', 'n_passengers', 'speed', 'floor_queues']
        spaces_to_include = {k: env.observation_space.spaces[k] for k in keys_to_include}

        # Get all possible values for each space
        values = []
        for key, space in spaces_to_include.items():
            if isinstance(space, spaces.Discrete):
                values.append(list(range(space.start, space.start + space.n)))
            elif isinstance(space, spaces.MultiDiscrete):
                dim_ranges = [
                    list(range(s, s + n)) for n, s in zip(space.nvec, space.start)
                ]
                # Cartesian product of all dimensions in the MultiDiscrete
                values.append(list(itertools.product(*dim_ranges)))
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")

        # Cartesian product of all value combinations
        combinations = list(itertools.product(*values))

        # Return as list of dictionaries for easier handling
        result = []
        for combo in combinations:
            obs = dict(zip(keys_to_include, combo))
            result.append(obs)

        return result
    
    actions_size = env.action_space.n    
    Q = defaultdict(lambda: np.zeros(actions_size))
    
    controllable_space = controllable_space()
        
    for episode in range(n_episodes):
        obs, _ = env.reset()
        uncontrollable_obs = obs['arrivals']

        cumulated_reward = 0        
        
        done = False
        while not done:
            action = env.action_space.sample()  
            print(f"Action: {action}")
            new_obs, _, terminated, truncated, _ = env.step(action) 
            uncontrollable_new_obs = new_obs['arrivals']
            
            # For each controllable factorization of the state, 
            # set the current uncontrollable observation
            for state in controllable_space:
                # Create a surrogate state with the uncontrollable observation
                surrog_state = state.copy()
                surrog_state['arrivals'] = uncontrollable_obs
                surrog_state_key = obs_to_key(surrog_state)
                
                # Create a surrogate environment with the current state
                surrogate_env = deepcopy(env)
                surrogate_env.unwrapped._elevator.set_status(surrog_state, current_time=env.unwrapped.current_time - 1)
                # Step through the surrogate environment
                surrog_new_state, reward, _, _, _ = surrogate_env.step(action)
                
                # Create a surrogate new state with the uncontrollable observation
                surrog_new_state['arrivals'] = uncontrollable_new_obs
                surrog_new_state_key = obs_to_key(surrog_new_state)

                # Q-update for the surrogate state
                best_next_action = np.argmax(Q[surrog_new_state_key])
                td_target = reward + gamma * Q[surrog_new_state_key][best_next_action]
                Q[surrog_state_key][action] += alpha * (td_target - Q[surrog_state_key][action])
                
                del surrogate_env    
            
            done = terminated or truncated
            obs = new_obs

        print(f"Episode {episode} finished with cumulated reward: {cumulated_reward}")  
        
    print("✅ Training complete!")
    return Q


            


