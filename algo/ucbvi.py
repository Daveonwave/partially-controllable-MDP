import os
import gymnasium as gym
from collections import defaultdict, Counter
from tqdm.rich import tqdm, trange
import numpy as np
import pickle
import json
from algo.utils import *
from algo.evaluation import validation_step


def train(env, args, eval_params, horizon, seed=None, model_file=None, settings=None):
    if args.get('dest_folder') is not None and os.path.exists(args['dest_folder']):
        dest_path = args['dest_folder']
    else:
        dest_path = './'
    writer, out_path = init_writer(f"{args['env']}/{args['exp_name']}/{args['algo']}", args, seed, dest_folder=dest_path)

    # Hyperparameters
    H = horizon
    n_episodes = args['n_episodes']
    delta = args['delta']
    c = args['c_bonus']
    rng = np.random.default_rng(seed=seed)
    eval_seed = args['eval_seed']
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    eval_counter = 0
    learning_curve = {}
    keys, multipliers = build_state_index_map(env)
    print(f"State keys: {keys} Multipliers: {multipliers}")
    
    S = get_state_size(env)  # total number of flattened states
    A = env.action_space.n
    print(f"State size: {S}, Action size: {A}, Horizon: {H}")
    
    # Visitation counters
    N_sa = np.zeros((S, A), dtype=np.int32)
    N_sas = defaultdict(lambda: defaultdict(Counter))
    
    # Estimates
    P_hat = defaultdict(lambda: defaultdict(dict))
    R_hat = np.zeros((S, A), dtype=np.float32)
    B_sa = np.zeros((S, A), dtype=np.float32)
    
    # Reward matrix
    R_hat = build_reward_matrix(env, keys, multipliers)
    
    # Value functions
    V = np.zeros((H+1, S), dtype=np.float32)
    Q = np.zeros((H, S, A), dtype=np.float32)
    
    # Precompute log term for bonus calculation
    log_term = np.log(2 * S * A * H / delta)
    
    def _update(s, a, r, s_next):
        """
        Update the empirical estimates of the MDP and the exploration bonus.

        Args:
            s (int): Current state.
            a (int): Action taken.
            r (float): Reward received.
            s_next (int): Next state.
        """
        s, s_next = int(s), int(s_next)
        
        N_sa[s, a] += 1
        N_sas[s][a][s_next] += 1
        n = N_sa[s, a]
        # R_hat[s, a] += (r - R_hat[s, a]) / n

        total = sum(N_sas[s][a].values())
        if total > 0:
            P_hat[s][a] = {sp: count / total for sp, count in N_sas[s][a].items()}
        else:
            P_hat[s][a] = {}
        B_sa[s, a] = c * np.sqrt(log_term / max(1, n))
    
    for episode in trange(n_episodes, desc="Training UCBVI"):
        # Validation step every 1000 episodes during training
        if episode % args['eval_every'] == 0 and episode > 0:
            best_eval_reward, best_eval_episode, eval_counter = validation_step(
                env_name=args['env'],
                eval_params=eval_params,
                episode=episode, 
                eval_episodes=args['eval_episodes'],
                Q=Q[0, :, :], 
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
        obs_key = obs_to_key(obs, keys, multipliers, no_batch=True)
        
        cumulated_reward = 0.
        h = 0
        done = False
        
        while not done:
            # randomized argmax on ties
            q_vals = Q[h, obs_key, :]
            action = int(rng.choice(np.flatnonzero(q_vals == q_vals.max())))  

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_key = obs_to_key(next_obs, keys, multipliers, no_batch=True)

            # Update empirical estimates
            _update(obs_key, action, reward, next_obs_key)
            cumulated_reward += reward
            done = terminated or truncated
            
            obs, obs_key = next_obs, next_obs_key
            h += 1
        
        writer.add_scalar('Training/MeanEpisodeReward', cumulated_reward, episode)
        writer.add_scalar('Training/MedianEpisodeReward', np.median(cumulated_reward), episode)
        
        # Backward Value Iteration with bonuses
        V[H, :]  = 0  # Terminal value function
        for h in reversed(range(H)):
            for s in range(S):
                # expected value per action (A,)
                exp_values = np.zeros(A, dtype=np.float32)
                for a, nexts in P_hat[s].items():
                    exp_values[a] = sum(prob * V[h+1, s_next] for s_next, prob in nexts.items())
                Q[h, s, :] = np.minimum(H, R_hat[s, :] + B_sa[s, :] + exp_values)
                V[h, s] = np.max(Q[h, s, :])
                
        writer.add_scalar('Q/Max', np.max(Q), episode)
        writer.add_scalar('Q/Min', np.min(Q), episode)
        writer.add_histogram('Q/Values', Q, episode)

    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)

    with open(f"{dest_path}/logs/results/{out_path}/{seed}/world_models.pkl", "wb") as model_output:
        pickle.dump({'V': V, 'N_sa': N_sa}, model_output)
    
    print("Training complete!")
    writer.close()

