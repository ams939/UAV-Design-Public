import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train.Hyperparams import Hyperparams


def viz_metrics_dic(exp_id: str, save=False):
	exp_path = f"experiments/DQN/{exp_id}"
	metrics_dic = torch.load(f"{exp_path}/metrics.pth")
	n_stable = metrics_dic["n_stable"]
	
	plt.plot(np.arange(len(n_stable))*1000, n_stable)
	plt.title("Cumulative stable drone count vs. episode")
	plt.xlabel("Episode")
	plt.ylabel("Cum. stable drone count")
	plt.tight_layout()
	
	if save:
		plt.savefig(f"{exp_path}/metrics.png")
	else:
		plt.show()
		
	plt.clf()
	

def viz_stable(data: pd.DataFrame, title="", save_path=None):
	n_episodes = np.amax(data["episode"].values)
	stable_ep = np.zeros(n_episodes)
	stable_ep_sm = np.zeros((n_episodes // 100) + 1)
	
	for row_idx in range(len(data)):
		ep_row = data.iloc[row_idx]
		
		stable = ep_row["result"] == 'Success'
		if stable:
			stable_ep[ep_row["episode"] - 1] += 1
			
			if stable_ep[ep_row["episode"] - 1] >= 25:
				stable_ep[ep_row["episode"] - 1] = 0
				
	for row_idx in range(100, n_episodes, 100):
		stable_ep_sm[row_idx // 100] = np.mean(stable_ep[row_idx - 100:row_idx])
	
	stable_ep_sm[-1] = np.mean(stable_ep[n_episodes*100:-1])
	
	plt.subplot(2, 1, 1)
	plt.plot(np.arange(len(stable_ep)), stable_ep)
	plt.title("Stable Drone Count per Episode")
	plt.xlabel("Episode")
	plt.ylabel("Count")
	
	plt.subplot(2, 1, 2)
	plt.plot(np.arange(0, len(stable_ep)+1, 100), stable_ep_sm, c='green')
	plt.title("Smoothed Stable Drone Count (Avg. per 100 episodes)")
	plt.xlabel("Episode")
	plt.ylabel("Avg. Count")
	
	plt.suptitle(title)
	plt.tight_layout()
	
	if save_path is not None:
		plt.savefig(f"{save_path}/stable.png")
	else:
		plt.show()
		
	plt.clf()
	
	
def viz_metric_trends(data: pd.DataFrame, thresholds=None, obj=None, save_path=None):
	episode_begin = np.amin(data["episode"].values)
	episode_end = np.amax(data["episode"].values)
	
	max_metrics = np.zeros((episode_end - episode_begin - 1, 3))
	for ep in range(episode_begin, episode_end - 1):
		ep_data = data.loc[data["episode"] == ep]
		ep_data = ep_data.loc[data["result"] == "Success"]
		
		metrics = ep_data[["range", "cost", "velocity"]].values
		metrics = np.asarray([np.amax(metrics[:, 0]), np.amin(metrics[:, 1]), np.amax(metrics[:, 2])])
		max_metrics[ep - episode_begin, :] = metrics
		
	eps = np.arange(episode_begin, episode_end - 1)
	plt.subplot(3, 1, 1)
	plt.plot(eps, max_metrics[:, 0], c="g", label="Max value")
	plt.xlabel("Episode")
	plt.ylabel("Range (mi)")
	
	if thresholds is not None:
		if "range" in thresholds:
			try:
				th = thresholds["range"]["lower"]
			except TypeError:
				try:
					th = thresholds["range"]
				except Exception:
					print("Error retrieving threshold")
					th = 0
			plt.plot([episode_begin, episode_end], [th, th], "r--", label=f"Obj. {obj}")
			
	plt.legend()
	
	plt.subplot(3, 1, 2)
	plt.plot(eps, max_metrics[:, 1], c="m", label="Min value")
	plt.xlabel("Episode")
	plt.ylabel("Cost (USD)")
	
	if thresholds is not None:
		if "cost" in thresholds:
			try:
				th = thresholds["cost"]["upper"]
			except TypeError:
				try:
					th = thresholds["cost"]
				except Exception:
					print("Error retrieving threshold")
					th = 0
			plt.plot([episode_begin, episode_end], [th, th], "r--", label=f"Obj. {obj}")
	plt.legend()
	plt.subplot(3, 1, 3)
	plt.plot(eps, max_metrics[:, 2], c="b", label="Max value")
	plt.xlabel("Episode")
	plt.ylabel("Velocity (mph)")
	
	if thresholds is not None:
		if "velocity" in thresholds:
			try:
				th = thresholds["velocity"]["lower"]
			except TypeError:
				try:
					th = thresholds["velocity"]
				except Exception:
					print("Error retrieving threshold")
					th = 0
			plt.plot([episode_begin, episode_end], [th, th], "r--", label=f"Obj. {obj}")
	plt.legend()
	plt.suptitle("Best metric value in each episode")
	plt.tight_layout()
	
	if save_path is not None:
		plt.savefig(f"{save_path}/metric_trends.png")
	else:
		plt.show()
		
	plt.clf()
	

def viz_epsilon(data: pd.DataFrame, save_path=None):
	
	eps = data["epsilon"].values[0:-1:100]
	episode = data["episode"].values[0:-1:100]
	
	plt.plot(episode, eps)
	plt.xlabel("Episode")
	plt.ylabel("Epsilon")
	
	plt.title("Epsilon vs. Episode")
	
	if save_path is not None:
		plt.savefig(f"{save_path}/epsilon.png")
	else:
		plt.show()
	

if __name__ == "__main__":
	import os
	
	exp_id = "103122224600"
	exp_ids = ['112822140104']
	
	for exp_id in exp_ids:
		exp_path = f"experiments/DQN/{exp_id}"
		hparams_file = "dqn_hparams_hyform.json"
		hparams = Hyperparams(f"{exp_path}/{hparams_file}")
		exp_viz_path = f"{exp_path}/viz"
		if not os.path.exists(exp_viz_path):
			os.mkdir(exp_viz_path)
			
		fname = f"{exp_path}/datafile_log.json"
		try:
			data = pd.DataFrame(json.load(open(fname)))
		except json.JSONDecodeError:
			data = pd.DataFrame(json.loads("[" + open(fname).read() + "]"))
		
		viz_metric_trends(data, thresholds=hparams.objective_definitions, obj=hparams.objective, save_path=exp_viz_path)
		
		viz_epsilon(data, save_path=exp_viz_path)
		viz_stable(data, title=f"Stable drone counts for {exp_id}\n(ep_len=25, LR=0.001, obj=2a)", save_path=exp_viz_path)
		
		viz_metrics_dic(exp_id, save=True)
		
	# max_ep = np.amax(data["episode"].values)
	# max_it = np.amax(data["T"].values)
	
	# data2 = pd.DataFrame(json.load(open("experiments/DQN/101022100707/datafile_log.json")))
	# data2["episode"] += max_ep
	# data2["T"] += max_it
	
	# os.mkdir("experiments/DQN/101022100707+100822101255")
	# data = pd.concat([data, data2])
	# data.to_csv("experiments/DQN/101022100707+100822101255/datafile_log.csv", index=False)
	