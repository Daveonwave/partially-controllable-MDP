from pcmdp.simulator.passenger import generate_arrival_distribution
from pcmdp.env import ElevatorEnv
from rich.pretty import pprint


def random_policy(env: ElevatorEnv):
    n_episodes = 10
    
    for episode in range(n_episodes):
        env.reset()
        done = False
        cumulated_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, Reward: {reward}, Cumulative Reward: {cumulated_reward}")


def longest_queue_first(env: ElevatorEnv):
    n_episodes = 1
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        cumulated_reward = 0
        
        while not done:
            if obs['n_passangers'] > 0:
                if obs['current_position'] == info['min_floor']:
                    action = 2  # stay still
                else:
                    action = 1  # move down
            else:
                # Check if there are any passengers waiting at the current floor
                len_max_queue, idx_max_queue = max((queue, idx + 1) for idx, queue in enumerate(obs['floor_queues']))
                # TODO: how to deal with ties?
                
                if len_max_queue == 0:
                    action = 2  # stay still
                elif obs['current_position'] == (idx_max_queue) * info['floor_height']:
                    action = 2  # stay still
                elif obs['current_position'] < (idx_max_queue) * info['floor_height']:
                    action = 0  # move up            
                elif obs['current_position'] > (idx_max_queue) * info['floor_height']:
                    action = 1
                else:
                    action = 2
                    
            obs, reward, terminated, truncated, info = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, "
                   f"Reward: {reward}, Cumulative Reward: {cumulated_reward}, "
                   f"Current Pos: {obs['current_position']}, Passengers: {obs['n_passangers']}, "
                   f"Queues: {obs['floor_queues']}")