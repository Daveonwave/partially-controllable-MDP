import os
from tqdm.rich import tqdm, trange
import numpy as np
import json
import numpy as np
from pcmdp.elevator.elevator_vecEnv import FunctionalElevatorEnv
from pcmdp.taxi.taxi_vecEnv import FunctionalTaxiEnv
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
    
    S = get_state_size(env)
    A = env.action_space.n
    Q = np.zeros((S, A))
    
    controllable_space = get_controllable_space(env)
    unctrl_vars = env.unwrapped.get_uncontrollables()
    
    # Functional environment for vectorized operations
    if args['env'] == 'elevator':
        vec_env = FunctionalElevatorEnv(settings=settings)
    elif args['env'] == 'taxi':
        vec_env = FunctionalTaxiEnv(settings=settings)
    else:
        raise ValueError("Unsupported environment for exogenous Q-Learning.")
        
    for episode in trange(n_episodes):
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
                eval_seed=eval_seed
                )

            if eval_counter == args['max_no_improvement']:
                break
            
        # Sample the uncontrollable observation from the environment
        dataset = {key: [] for key in unctrl_vars}
        exog_count = 0
        
        obs, _ = env.reset()
        exog_obs = env.unwrapped.get_unctrl_obs(obs)
        for key in unctrl_vars:
            dataset[key].append(exog_obs[key])
        exog_count += 1

        done = False
        while not done:
            # Random sampling the action just to move the environment forward
            action = env.action_space.sample()  
            new_obs, _, terminated, truncated, _ = env.step(action) 
            exog_obs = env.unwrapped.get_unctrl_obs(new_obs)
            for key in unctrl_vars:
                dataset[key].append(exog_obs[key])
            exog_count += 1
            done = terminated or truncated
        
        # Now we have a dataset of uncontrollable observations
        vec_state = controllable_space.copy()
        batch_size = len(controllable_space[list(controllable_space.keys())[0]])
        params = {'batch_size': batch_size, **settings}
        cumulated_rewards = np.zeros(batch_size)
        
        # The initial state is not needed in this case, as we will use the uncontrollable observations directly
        #_ = vec_env.initial(rng=None, params=params)
        
        # NOTE: the indices are used to index the Q-table. Instead, vec_state and
        # vec_action use a subset of the entire state space, which is the controllable part.

        for i in range(exog_count - 1):
            # We take the current uncontrollable observation and the next one
            unctrl_obs = {key: dataset[key][i] for key in unctrl_vars}
            next_unctrl_obs = {key: dataset[key][i + 1] for key in unctrl_vars}
            
            # For each controllable state, we add the current uncontrollable observation
            vec_state = compose_vec_state(vec_state, unctrl_obs, batch_size)        
            indices = flatten_state(vec_state, keys, multipliers)

            # Choose action based on epsilon-greedy policy
            if rng.uniform() < epsilon:
                #vec_action = np.array([rng_action.integers(0, A)] * batch_size)  # Explore action space
                vec_action = np.array([rng.integers(0, A)] * batch_size)  # Explore action space
            else:
                vec_action = np.argmax(Q[indices, :], axis=1)
                
            # Get the next state from the environment
            next_vec_state = vec_env.transition(state=vec_state, action=vec_action, rng=None, params=params)
            next_vec_state = compose_vec_state(next_vec_state, next_unctrl_obs, batch_size)
            
            # Calculate rewards
            rewards = vec_env.reward(state=vec_state, action=vec_action, next_state=next_vec_state, rng=None, params=params)
            cumulated_rewards += rewards
            
            # Q-update
            next_indices = flatten_state(next_vec_state, keys, multipliers)
            best_next_actions = np.argmax(Q[next_indices, :], axis=1)
            td_targets = rewards + gamma * Q[next_indices, best_next_actions] 
            Q[indices, vec_action] += alpha * (td_targets - Q[indices, vec_action])
        
            # I don't have to move to the next state, as the next uncontrollable observation will be used in the next iteration            
            #vec_state = next_vec_state

        writer.add_scalar('Training/MeanEpisodeReward', np.mean(cumulated_rewards), episode)
        writer.add_scalar('Training/MedianEpisodeReward', np.median(cumulated_rewards), episode)
        writer.add_scalar('Training/TD_Error', np.mean(td_targets - Q[indices, vec_action]), episode)
        
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
