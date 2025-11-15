import os
from tqdm.rich import trange
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
    rng = np.random.default_rng(seed=seed)
    eval_seed = args['eval_seed']
    
    best_eval_reward = -np.inf
    best_eval_episode = 0
    eval_counter = 0
    learning_curve = {}
    keys, multipliers = build_state_index_map(env)
    
    state_factorizability = False if env.spec.id == 'pcmdp/elevator-v0' else True
    
    S = get_state_size(env)  # total number of flattened states
    A = env.action_space.n
    R = build_reward_matrix(env, keys, multipliers)
    print(f"State size: {S}, Action size: {A}, Horizon: {H}")
    print(f"State keys: {keys} Multipliers: {multipliers}")
    
    ctrl_keys, ctrl_multipliers = build_state_index_map(env, env.unwrapped.get_controllables())
    if not state_factorizability:     # Special case: controllable dynamics depend on uncontrollable state
        P_ctrl = build_ctrl_transition_matrix(env, keys, multipliers, ctrl_keys, ctrl_multipliers)
    else:
        P_ctrl = build_ctrl_transition_matrix(env, ctrl_keys, ctrl_multipliers, ctrl_keys, ctrl_multipliers)
    
    unctrl_keys, unctrl_multipliers = build_state_index_map(env, env.unwrapped.get_uncontrollables())
    S_unctrl = get_composite_state_size(env, keys=unctrl_keys)
        
    # Visitation counters for uncontrollable dynamics
    N_ssp = np.zeros((S_unctrl, S_unctrl), dtype=np.int32)
    P_unctrl = np.zeros((S_unctrl, S_unctrl), dtype=np.float32)
    P_unctrl[:] = 1.0 / max(1, P_unctrl.shape[0])
    
    # Value functions
    V = np.zeros((H+1, S), dtype=np.float32)
    Q = np.zeros((H, S, A), dtype=np.float32)
    V[H, :] = 0.0
    
    # Precompute index mappings for all states
    s_ctrl_idx_map = np.zeros(S, dtype=int)
    s_unctrl_idx_map = np.zeros(S, dtype=int)
    for s in range(S):
        state = key_to_obs(s, env, keys, multipliers)
        s_ctrl_idx_map[s] = obs_to_key(state, ctrl_keys, ctrl_multipliers, no_batch=True)
        s_unctrl_idx_map[s] = obs_to_key(state, unctrl_keys, unctrl_multipliers, no_batch=True)        

    def _update_unctrl_dynamics(s_unctrl, s_unctrl_next):
        """
        Update the empirical estimates of the MDP and the exploration bonus.

        Args:
            s (int): Current state.
            s_next (int): Next state.
        """
        s_idx = obs_to_key(s_unctrl, unctrl_keys, unctrl_multipliers, no_batch=True)
        s_next_idx = obs_to_key(s_unctrl_next, unctrl_keys, unctrl_multipliers, no_batch=True)
        
        N_ssp[s_idx, s_next_idx] += 1
        #P_unctrl[s_idx, s_next_idx] = N_ssp[s_idx, s_next_idx] / np.sum(N_ssp[s_idx, :])  # MLE
        for s_idx in range(S_unctrl):
            total = np.sum(N_ssp[s_idx, :])
            if total > 0:
                P_unctrl[s_idx, :] = N_ssp[s_idx, :] / total

        #print(N_ssp)
        
    # Training procedure of Exa-VI
    for episode in trange(n_episodes, desc="Training ExA-VI"):
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
            next_obs_key = obs_to_key(next_obs, keys, multipliers)

            # Update empirical estimates of uncontrollable dynamics
            unctrl_obs = env.unwrapped.get_unctrl_obs(obs)
            unctrl_obs_next = env.unwrapped.get_unctrl_obs(next_obs)
            _update_unctrl_dynamics(unctrl_obs, unctrl_obs_next)
            
            cumulated_reward += reward
            done = terminated or truncated
            
            obs, obs_key = next_obs, next_obs_key
            h += 1

        writer.add_scalar('Training/MeanEpisodeReward', cumulated_reward, episode)
        writer.add_scalar('Training/MedianEpisodeReward', np.median(cumulated_reward), episode)
        
        # Assemble the P_hat from the controllable and uncontrollable parts
        P_hat = np.zeros((S, A, S), dtype=np.float32)
        for s in range(S):
            sc = s_ctrl_idx_map[s] if state_factorizability else s
            su = s_unctrl_idx_map[s]
            for a in range(A):
                for sp in range(S):
                    scp = s_ctrl_idx_map[sp]
                    sup = s_unctrl_idx_map[sp]
                    P_hat[s, a, sp] = P_ctrl[sc, a, scp] * P_unctrl[su, sup]
        
        # Now backward iteration
        for h in reversed(range(H)):
            Q[h] = R + np.einsum("sax,x->sa", P_hat, V[h + 1])
            V[h] = np.max(Q[h], axis=1)
        
        writer.add_scalar('Q/Max', np.max(Q), episode)
        writer.add_scalar('Q/Min', np.min(Q), episode)
        writer.add_histogram('Q/Values', Q, episode)

    os.makedirs(f"{dest_path}/logs/results/{out_path}/{seed}/", exist_ok=True)
    with open(f"{dest_path}/logs/results/{out_path}/{seed}/learning_curve.json", "w", encoding="utf8") as output_file:
        json.dump(learning_curve, output_file)

    with open(f"{dest_path}/logs/results/{out_path}/{seed}/world_models.pkl", "wb") as model_output:
        pickle.dump({'V': V, 'N_ssp': N_ssp}, model_output)
    
    print("Training complete!")
    writer.close()

