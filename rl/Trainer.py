"""
Rainbow Deep Q-Learning implementation from https://github.com/Kaixhin/Rainbow by Kai Arulkumaran et al.
https://github.com/Kaixhin/Rainbow/blob/master/main.py

Modified by Aleksanteri Sladek, 27.7.2022
    - Refactoring

"""

from __future__ import division
import bz2
import os
import pickle
from time import time
import sys
import json

import numpy as np
import torch
from tqdm import trange

from data.Constants import ERROR
from rl.DQNAgent import Agent
from rl.Env import UAVEnv
from rl.ReplayBuffer import ReplayMemory
from rl.test import test
from utils.Exceptions import SamplingError


def train_dqn(hparams):

    logger = hparams.logger
    results_dir = hparams.experiment_folder
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    # Set a seed
    np.random.seed(hparams.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    
    # Choose the device
    if torch.cuda.is_available():
        hparams.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = hparams.enable_cudnn
    else:
        hparams.device = torch.device('cpu')

    hparams.logger.log({"name": "Trainer",
                        "msg": f"Using device {hparams.device}"})

    # Environment initialization
    hparams.logger.log({"name": "Trainer",
                        "msg": "Initializing environment..."})
    env = UAVEnv(hparams)
    env.train()

    # Agent initialization (loadng of any pre-trained models is done here)
    dqn = Agent(hparams)

    # If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
    hparams.logger.log({"name": "Trainer", "msg": "Building training memory..."})
    mem = build_memory(hparams)

    priority_weight_increase = (1 - hparams.priority_weight) / (hparams.T_max - hparams.learn_start)

    """
    Training loop:
        First has the agent take one step in the environment, i.e, perform an action, saves it to the ReplayMemory.
        ReplayMemory is populated in this way for every training iteration. If learn_start iterations have passed,
        the DQN training begins.
    
    
    """
    hparams.logger.log({"name": "Trainer",
                        "msg": f"Training loop beginning, running for {hparams.T_max} iterations, "
                               f"warming up for {hparams.learn_start} iterations"})
    dqn.train()
    done = True
    state = None
    episode_times = []
    rewards = [0]
    s_time = time()
    learn_start = hparams.learn_start
    loop_count = 0
    eval_data = None
    objective = None
    obj_eval_file = os.path.join(hparams.experiment_folder, "obj_eval_stats.json")
    
    # Run training loop for T_max iterations
    for T in range(1, hparams.T_max + 1):
        # https://stackoverflow.com/questions/53198503/epsilon-and-learning-rate-decay-in-epsilon-greedy-q-learning
        epsilon = max(((hparams.T_max - T)/ hparams.T_max), 0) * (hparams.e_init - hparams.e_end) + hparams.e_end
        hparams.logger.log({"name": "Trainer",
                            "msg": f"Training iteration {T} beginning...",
                            "debug": True})

        # Iterate the agent in the environment by one step forwards
        if done:
            hparams.logger.log({"name": "Trainer",
                                "msg": f"T={T}, Next episode beginning (e={env.episode_count})."})
            e_time = time()
            ep_time = e_time - s_time
            episode_times.append(ep_time)
            
            s_time = time()
            
            rewards = []
            state = env.reset()
            objective = env.get_objective()
        
        if T % hparams.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights
        
        # Choose an action greedily (with e-greedy AND noisy weights)
        action = dqn.act_e_greedy(state, objective, epsilon=epsilon)
        hparams.logger.log({"name": "Trainer",
            "msg": f"Perform action {action.predecessor_action} (greedy={dqn.greedy}, e={epsilon:.2f})",
                            "debug": True})
        
        # Perform the action to take the step
        next_state, reward, done = env.step(action)
        
        if hparams.reward_clip > 0:
            reward = max(min(reward, hparams.reward_clip), -hparams.reward_clip)  # Clip rewards
            
        rewards.append(reward)

        hparams.logger.log({"name": "Trainer", "msg": f"Action reward {reward}."})
        
        # Append transition to memory
        mem.append(state.to_string(), action.to_string(), reward, objective, done)
        
        # If enough warmup iterations have passed, let the agent begin learning
        if T >= learn_start:
            
            # Anneal importance sampling weight β to 1
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
            # Learning is done every replay_frequency steps
            if T % hparams.replay_frequency == 0:
                hparams.logger.log({"name": "Trainer",
                                    "msg": f"T={T}, updating online network."})
                # Problem: Sometimes RMB sampling can't find a sample and gets stuck in a sampling loop
                try:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning
                    loop_count = 0
                except SamplingError as e:
                    # Prevent infinite looping
                    if loop_count > 10:
                        hparams.logger.log(
                            {"name": "Trainer",
                             "msg": f"Aborting experiment, max RMB sampling loop count ({loop_count}) exceeded!",
                             "msg_type": ERROR})
                        sys.exit(-1)
                    
                    hparams.logger.log({"name": "Trainer", "msg":
                                        "Skip learning iteration (infinite RMB sampling loop detected)\n"
                                        f"Error encountered: {e}"})
                    
                    # Let the agent accumulate more experience and try to learn again
                    learn_start = T + 100
                    loop_count += 1
                    
                except Exception as e:
                    raise e
                
            if T % hparams.evaluation_interval == 0:
                active_objectives = env.get_active_objectives()
                eval_data = evaluate(hparams, dqn, T, active_objectives)
                eval_data["episode"] = env.episode_count

                if not os.path.exists(obj_eval_file):
                    json.dump([eval_data], open(obj_eval_file, "w"))

                with open(obj_eval_file, "r") as f:
                    d = json.load(f)

                with open(obj_eval_file, "w") as f:
                    d.append(eval_data)
                    json.dump(d, f)
            else:
                eval_data = None
                
            # Update target network every target_update steps
            if T % hparams.target_update == 0:
                hparams.logger.log({"name": "Trainer", "msg": f"T={T}, updating target network."})
                dqn.update_target_net()

            # Checkpoint the network every checkpoint_interval steps
            if (hparams.checkpoint_interval != 0) and (T % hparams.checkpoint_interval == 0):
                checkpoint(hparams, dqn, results_dir, mem)
                
        state = next_state
        hparams.logger.log({"name": "Trainer",
                            "msg": f"Training iteration {T} ending...",
                            "debug": True})

        # Perform some logging of current parameters
        log_state(state, env, epsilon, reward, eval_data, hparams)
    
    if hparams.save_mem:
        hparams.logger.log({"name": "Trainer",
                            "msg": f"Saving final replay buffer."})
        save_memory(mem, f"{results_dir}/replay_memory.pkl", hparams.disable_bzip_memory)

    hparams.logger.log({"name": "Trainer",
                        "msg": f"Saving final model."})
    dqn.save(results_dir, 'model.pth')


def evaluate(hparams, dqn, T, active_objectives):

    hparams.logger.log({"name": "Trainer", "msg": f"\nEvaluating with objectives f{','.join(active_objectives)} "
                                              f"for {hparams.evaluation_episodes} episodes.\n"})

    dqn.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, obj_stats = test(hparams, dqn, active_objectives)
    dqn.train()  # Set DQN (online network) back to training mode

    hparams.logger.log({"name": "Trainer", "msg": f'T={T}/{hparams.T_max} | Avg. reward: {avg_reward} '})

    eval_data = {"eval_avg_reward": avg_reward}

    obj_data = dict()
    for obj in obj_stats.keys():
        for stat_key, value in obj_stats[obj].items():
            obj_data[f"eval_{obj}_{stat_key}"] = value
    
    eval_data.update(obj_data)
    
    return eval_data
    
    
def checkpoint(hparams, dqn, results_dir, mem):

    hparams.logger.log({"name": "Trainer",
                        "msg": f"Saving model checkpoint."})
    dqn.save(results_dir, 'model.pth')
    
    if hparams.save_mem:
        hparams.logger.log({"name": "Trainer",
                            "msg": f"Saving replay buffer."})
        save_memory(mem, f"{results_dir}/replay_memory.pkl", hparams.disable_bzip_memory)
        

def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)
            
            
def build_memory(hparams, mem_type=None):
    if mem_type is None:
        
        try:
            if not hparams.memory:
                hparams.logger.log({"name": "Trainer", "msg": "ReplayMemory buffer file not specified."})
                raise ValueError
            elif not os.path.exists(hparams.memory):
                hparams.logger.log({"name": "Trainer", "msg": f"ReplayMemory buffer file {hparams.memory} "
                                                              f"does not exist."})
                raise ValueError
            
            mem = load_memory(hparams.memory, hparams.disable_bzip_memory)
            
            # Don't need warmup iterations if the buffer is already populated
            hparams.learn_start = 0
            hparams.logger.log(
                {"name": "Trainer", "msg": f"Loaded existing ReplayMemory buffer from {hparams.memory}"})
            
            # In error case just create a new buffer and notify user
        except ValueError:
            hparams.logger.log({"name": "Trainer", "msg": f"Warning: Existing ReplayMemory not found!"})
            hparams.logger.log({"name": "Trainer", "msg": "Constructing new ReplayMemory buffer."})
            mem = ReplayMemory(hparams, hparams.memory_capacity)
            assert hparams.learn_start > 0, "Error, must have warmup iterations to populate buffer."
        
        hparams.memory = f"{hparams.experiment_folder}/replay_memory.pkl"
    
    elif mem_type == "validation":
        hparams.logger.log({"name": "Trainer", "msg": "Loading validation memory from file."})
        try:
            assert os.path.exists(hparams.val_memory)
            mem = load_memory(hparams.val_memory, hparams.disable_bzip_memory)
        except (AssertionError, Exception) as e:
            hparams.logger.log({"name": "Trainer", "msg": f"Loading validation memory from file failed: {e}"})
            hparams.logger.log({"name": "Trainer",
                                "msg": "Constructing new validation memory..."})
        
            # Construct validation memory, if it wasn't loaded
            mem = init_mem_buffer(hparams)
            hparams.val_memory = f"{hparams.experiment_folder}/val_memory.pkl"
        
            if hparams.save_mem:
                save_memory(mem, hparams.val_memory, hparams.disable_bzip_memory)
                
    else:
        raise Exception("Unknown mem_type argument.")
     
    return mem


def init_mem_buffer(hparams):
    mem = ReplayMemory(hparams, hparams.evaluation_size)
    env = UAVEnv(hparams)
    env.train()
    T, done = 0, True
    state = None
    for _ in trange(1, hparams.evaluation_size):
        if done:
            state = env.reset()
        
        symmetric = hparams.symmetric_actions if hparams.symmetric_actions is not None else False
        no_size = hparams.no_size if hparams.no_size is not None else False
        actions = state.get_successors(symmetric=symmetric, no_size=no_size)
        next_action_idx = np.random.randint(0, len(actions))
        next_action = actions[next_action_idx]
        objective = env.get_objective()
        
        next_state, _, done = env.step(next_action)
        mem.append(state.to_string(), next_action.to_string(), 0.0, objective, done)
        state = next_state
        
    return mem


def log_state(state, env, epsilon, reward, eval_data, hparams):
    # Log to text based loggers
    hparams.logger.log({"name": "Trainer",
                        "msg": f"Current state {state.__str__()}, "
                               f"metrics: {state.range},{state.cost},{state.velocity}, "
                               f"stable: {state.is_stable}",
    })
    
    # Log data to data loggers
    state_data = {
        "T": env.run_stats["n_iteration"],
        "episode": env.episode_count,
        "config": state.__str__(),
        "range": float(state.range),
        "cost": float(state.cost),
        "velocity": float(state.velocity),
        "result": state.result,
        "epsilon": epsilon,
        "n_stable": env.drone_stats["n_stable"],
        "n_successful": env.run_stats["n_successful_episode"],
        "train_reward": reward
    }
    
    # Log evaluation data if it was provided
    if eval_data is not None:
        state_data.update(eval_data)
    
    # Log objective-wise stats
    obj_data = dict()
    for obj in env.obj_stats.keys():
        for stat_key, value in env.obj_stats[obj].items():
            obj_data[f"{obj}_{stat_key}"] = value
            
    state_data.update(obj_data)
    
    hparams.logger.log({"name": "Trainer",
                        "data": state_data})
