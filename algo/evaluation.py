import os
import pickle
import json
import gymnasium as gym
import numpy as np
from algo.utils import obs_to_key


def validation_step(env_name: str,
                    eval_params: dict,
                    episode: int, 
                    eval_episodes: int,
                    Q: np.ndarray, 
                    tol: float, 
                    eval_counter: int,
                    exp_name: str, 
                    learning_curve: dict,
                    best_eval_reward: float,
                    best_eval_episode: int,
                    keys: list = None, 
                    multipliers: np.ndarray = None,
                    writer=None,
                    train_seed=None,
                    dest_path='./',
                    eval_seed=None
                    ):
    eval_avg_reward, cum_rewards = evaluate(env_name, Q, eval_params, keys, multipliers, eval_episodes=eval_episodes, writer=writer, episode=episode, eval_seed=eval_seed)
    learning_curve[episode] = cum_rewards
    eval_counter += 1
    print(f"Seed {train_seed}: Evaluation finished with avg. cum. reward: {eval_avg_reward} at episode {episode}")
    
    if eval_avg_reward > best_eval_reward + tol:
        best_eval_reward = eval_avg_reward
        best_eval_episode = episode
        eval_counter = 0
        print(f"Seed {train_seed}: New best evaluation reward: {best_eval_reward} at episode {best_eval_episode}")

        os.makedirs(f"{dest_path}/logs/models/{exp_name}/{train_seed}", exist_ok=True)
        with open(f"{dest_path}/logs/models/{exp_name}/{train_seed}/best_q_table.pkl", "wb") as output_file:
            pickle.dump(Q, output_file)
    
    return best_eval_reward, best_eval_episode, eval_counter


def evaluate(env_name, Q, eval_env_params, keys=None, multipliers=None, eval_episodes=10, writer=None, episode=0, eval_seed=None):
    env = gym.make(f"pcmdp/{env_name}-v0", **{'settings': eval_env_params})
    avg_cumulated_reward = 0
    ep_cum_rewards = []
    
    # Select the function to get indices based on keys and multipliers
    get_indices = lambda obs, keys, multipliers, _no_batch: obs_to_key(obs, keys, multipliers, _no_batch)

    # Fixed seed for evaluation, to ensure reproducibility
    rng = np.random.default_rng(eval_seed)
    
    for _ in range(eval_episodes):
        seed = rng.integers(low=0, high=1e6, size=1)[0]
        obs, _ = env.reset(options={'seed': seed})
        obs_key = get_indices(obs, keys, multipliers, _no_batch=True)
        
        done = False
        cumulated_reward = 0
                
        while not done:
            action = np.argmax(Q[obs_key, :])  # Exploit learned values
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_key = get_indices(next_obs, keys, multipliers, _no_batch=True)

            cumulated_reward += reward
            done = terminated or truncated
            obs, obs_key = next_obs, next_obs_key
        
        ep_cum_rewards.append(cumulated_reward)
        #print(info, cumulated_reward)
        
    avg_cumulated_reward = np.mean(ep_cum_rewards) 
    std_cumulated_reward = np.std(ep_cum_rewards)
    ci95 = 1.96 * std_cumulated_reward / np.sqrt(len(ep_cum_rewards))
        
    writer.add_scalar('Evaluation/EpisodeReward', avg_cumulated_reward, episode)
    writer.add_scalar('Evaluation/StdReward', std_cumulated_reward, episode)
    writer.add_scalar("Evaluation/CI95", ci95, episode)
    return avg_cumulated_reward, ep_cum_rewards
