import os
import gymnasium as gym
from collections import defaultdict
from tqdm.rich import tqdm, trange
import numpy as np
import pickle
import json
from algo.utils import *
from algo.evaluation import validation_step


def train(env, args, eval_params, seed=None, model_file=None, settings=None):
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)

    # Hyperparameters
    alpha = args['alpha']
    gamma = args['gamma']
    epsilon = args['epsilon']
    epsilon_decay = args['epsilon_decay']
    epsilon_min = args['epsilon_min']
    n_episodes = args['n_episodes']
    rng = np.random.default_rng(seed=seed)
    eval_seed = args['eval_seed']
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    eval_counter = 0
    learning_curve = {}
    keys, multipliers = build_state_index_map(env)
    print(f"Keys: {keys}, Multipliers: {multipliers}")

    #Q = defaultdict(default_q_values)  # Default Q-values for unseen states
    S = get_state_size(env)
    A = env.action_space.n
    Q = np.zeros((S, A))
    
    for episode in trange(n_episodes, desc="Training Q-Learning"):
        
        # Validation step every 1000 episodes during training
        if episode % args['eval_every'] == 0 and episode > 0:
            best_eval_reward, best_eval_episode, eval_counter = validation_step(
                env_name=args['env'],
                eval_params=eval_params,
                episode=episode, 
                eval_episodes=args['eval_episodes'],
                Q=Q, 
                keys=keys, 
                multipliers=multipliers,
                tol=args['tol'], 
                eval_counter=eval_counter,
                exp_name=out_path,
                learning_curve=learning_curve,
                best_eval_reward=best_eval_reward,
                best_eval_episode=best_eval_episode,
                writer=writer,
                train_seed=seed,
                dest_path=dest_path,
                eval_seed=eval_seed
                ) 
            
            if eval_counter == args['max_no_improvement']:
                break
        
        obs, _ = env.reset()
        obs_key = obs_to_key(obs, keys, multipliers)
        
        cumulated_reward = 0        
        done = False
        
        while not done:
            if rng.uniform() < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[obs_key, :])     # Exploit learned values
            
            next_obs, reward, terminated, truncated, _ = env.step(action) 
            next_obs_key = obs_to_key(next_obs, keys, multipliers)

            cumulated_reward += reward
            done = terminated or truncated
            
            # Q-update
            best_next_action = np.argmax(Q[next_obs_key, :])
            td_target = reward + gamma * Q[next_obs_key, best_next_action]
            Q[obs_key, action] += alpha * (td_target - Q[obs_key, action])

            obs, obs_key = next_obs, next_obs_key
        
        writer.add_scalar('Training/MeanEpisodeReward', cumulated_reward, episode)
        writer.add_scalar('Training/MedianEpisodeReward', np.median(cumulated_reward), episode)
        writer.add_scalar('Training/TD_Error', td_target - Q[obs_key][action], episode)
        
        writer.add_scalar('Q/Max', np.max(Q), episode)
        writer.add_scalar('Q/Min', np.min(Q), episode)
        writer.add_histogram('Q/Values', Q.flatten(), episode)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        writer.add_scalar('Exploration/Epsilon', epsilon, episode)

    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)
    
    print("Training complete!")
    writer.close()
