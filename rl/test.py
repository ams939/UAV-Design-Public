# -*- coding: utf-8 -*-
from __future__ import division
import os
from copy import deepcopy

from tqdm import trange
import plotly
import numpy as np
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch

from rl.Env import UAVEnv
from rl.Curriculum import FixedCurriculum
from rl.DQNAgent import Agent
from utils.utils import ddict


def eval_dqn(hparams):
    results_dir = hparams.experiment_folder
    dqn = Agent(hparams)
    env = UAVEnv(hparams)

    done = True
    T_rewards = np.zeros(100)
    obj_complete = np.zeros(100)
    final_states = []
    for ep_idx in trange(100):
        episode_length = 0
        reward_sum = 0
        
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            
            obj = env.get_objective()
            action = dqn.act_e_greedy(state, obj)  # Choose an action ε-greedily
            state, reward, done = env.step(action)  # Step
            print(state, reward, done)
            reward_sum += reward
        
            # Terminate if we reach the maximum episode length
            if episode_length > hparams.max_episode_length:
                done = True
            episode_length += 1
        
            if done:
                T_rewards[ep_idx] = reward_sum
                final_states.append(f"{str(state)}, {state.get_metrics()}")
                if env.reward_class.objective_complete(state.get_metrics()):
                    print("Objective was completed!")
                    obj_complete[ep_idx] = 1
                break
    
    print("Final states:\n")
    print("\n".join(final_states))
    print(f"Completed {np.sum(obj_complete)}/{len(obj_complete)} objectives")
                
    env.close()
    
    
# Test DQN
def test(args, dqn, active_objectives):

    T_rewards = np.zeros((len(active_objectives), args.evaluation_episodes))

    # Evaluate model performance on a single active objective
    obj_stats = None
    for idx, obj in enumerate(active_objectives):
        env_args = deepcopy(args)
        env_args.curriculum_class = FixedCurriculum
        env_args.curriculum_hparams = ddict({"objective": obj})

        env = UAVEnv(env_args)
        env.eval()

        if obj_stats is None:
            obj_stats = deepcopy(env.obj_stats)

        done = True
        state = None
        # Test performance over several episodes
        for ep_idx in trange(args.evaluation_episodes):
            episode_length = 0
            reward_sum = 0
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False

                action = dqn.act_e_greedy(state, obj)  # Choose an action ε-greedily
                state, reward, done = env.step(action)  # Step
                reward_sum += reward

                # Terminate if we reach the maximum episode length
                if episode_length > args.max_episode_length:
                    done = True
                episode_length += 1

                if done:
                    T_rewards[idx][ep_idx] = reward_sum
                    break

        obj_stats[obj] = env.obj_stats[obj]
        obj_stats[obj]["n_attempts"] -= 1
    
    avg_reward = np.mean(T_rewards)
    
    # Return average reward and Q-value
    return avg_reward, obj_stats


if __name__ == "__main__":
    from train.Hyperparams import Hyperparams
    hparams_file = "experiments/DQN/040923105156/dqn_hparams_hyform.json"
    
    hparams = Hyperparams(hparams_file)
    hparams["device"] = torch.device("cpu")
    eval_dqn(hparams)
