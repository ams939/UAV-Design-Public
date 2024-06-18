import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.DesignState import UAVDesign
import data.Constants as C
from data.Constants import INCREMENT_SYMBOL, DECREMENT_SYMBOL


def viz_size_dist(data: pd.DataFrame, plt_title="Drone Component size distributions"):
	stable_designs = data.loc[data["result"] == 'Success', 'config'].values
	
	# Parse drone design strings into drone design objects
	designs = [UAVDesign(uav_str) for uav_str in stable_designs]
	
	increments = []
	for design in designs:
		# Get the components for the drone
		components = design.get_components()
		for component in components:
			
			component = list(component)
			
			# If there's no size string, go to next component
			if len(component) == 4:
				increments.append(0)
				continue
			
			size_elems = component[4:]
			size = len(size_elems)
			
			if (INCREMENT_SYMBOL in size_elems) and (DECREMENT_SYMBOL not in size_elems):
				increments.append(size)
			elif (INCREMENT_SYMBOL not in size_elems) and (DECREMENT_SYMBOL in size_elems):
				increments.append(-size)
			else:
				print(f"Warning, mixture of size symbols in {''.join(component)}")
				n_inc = size_elems.count(INCREMENT_SYMBOL)
				n_dec = size_elems.count(DECREMENT_SYMBOL)
				
				if n_inc > n_dec:
					increments.append(size)
				elif n_inc < n_dec:
					increments.append(-size)
				else:
					if np.random.randn() < 0.5:
						increments.append(size)
					else:
						increments.append(-size)
	
	plt.subplot(2, 1, 1)
	plt.hist(increments, bins=np.arange(-30, 30, 1), histtype='bar', ec='black')
	plt.title(f"Distribution of sizes\n({INCREMENT_SYMBOL} and {DECREMENT_SYMBOL} symbol count)")
	plt.xlabel("Size")
	plt.ylabel("Frequency")
	plt.xticks(np.arange(-30, 30, 2) + 0.5, np.arange(-30, 30, 2))
	
	plt.subplot(2,1,2)
	plt.hist(increments, bins=np.arange(np.min(increments), np.max(increments), 1), log=True, color='g', histtype='bar', ec='black')
	plt.title(f"Distribution of sizes\n({INCREMENT_SYMBOL} and {DECREMENT_SYMBOL} symbol count)")
	plt.xlabel("Size")
	plt.ylabel("log(Frequency)")
	plt.suptitle(plt_title)
	plt.xticks(np.arange(-30, 30, 2) + 0.5, np.arange(-30, 30, 2))
	plt.tight_layout()
	plt.show()


def viz_metric_dists(data: pd.DataFrame, title, fname):

	metric_features = C.SIM_METRICS + [C.SIM_RESULT_COL]

	colors = ["#4287f5", "#f58442", "#109c3c", "#c43140"]

	plt.figure(figsize=(6, 12))
	plt.subplot(len(metric_features), 1, 1)
	plt.suptitle(f"Histograms of {fname} features\n(middle 90% of data, n={int(np.floor(0.9*len(data)))})")
	for idx, feat in enumerate(metric_features):
		try:
			feature_data = data[feat].values.squeeze()
		except KeyError:
			continue
		plt.subplot(len(metric_features), 1, idx + 1)
		if feat != C.SIM_RESULT_COL:
			feature_data = np.sort(feature_data)
			q1_idx = int(len(data) * 0.05)
			q3_idx = int(len(data) * 0.95)
			feature_data = feature_data[q1_idx:q3_idx]
			plt.title(f"mean={np.mean(feature_data):.2f}, sd={np.std(feature_data):.2f}")
		plt.hist(feature_data, color=colors[idx])
		plt.xlabel(feat)
	plt.tight_layout()
	plt.savefig(f"{os.path.dirname(fname)}/metric_histograms.png")
	
	
def viz_symmetric(designs: List[str]):
	from tqdm import trange
	n_symmetric = 0
	for idx in trange(len(designs)):
		uav = UAVDesign(designs[idx])
		
		# Remove the payload
		uav_tensor = uav.to_tensor()
		uav_tensor = uav_tensor[:6*7*7]
		uav_tensor = uav_tensor.reshape((7, 7, 6))
		uav_tensor = uav_tensor[:, :, :-1]
		
		uav_tensor = uav_tensor.numpy()
		
		uav_footprint = np.amax(uav_tensor, axis=-1)
		
		# The four perpendicular half-spaces
		l = np.zeros((7, 7))
		l[:, :4] = uav_footprint[:, :4]
		
		r = np.zeros((7, 7))
		r[:, 3:] = uav_footprint[:, 3:]
		
		u = np.zeros((7, 7))
		u[:4, :] = uav_footprint[:4, :]
		
		d = np.zeros((7, 7))
		d[3:, :] = uav_footprint[3:, :]
		
		# The four diagonal half-spaces
		ur_diag = np.triu(uav_footprint)
		ll_diag = np.tril(uav_footprint)
		ul_diag = np.flip(np.triu(np.flip(uav_footprint, axis=1)), axis=1)
		lr_diag = np.flip(np.tril(np.flip(uav_footprint, axis=1)), axis=1)
		
		if np.allclose(l, np.flip(r, axis=1)):
			n_symmetric += 1
		elif np.allclose(u, np.flip(d, axis=0)):
			n_symmetric += 1
		elif np.allclose(ur_diag, np.flip(np.flip(ll_diag, axis=0), axis=1)):
			n_symmetric += 1
		elif np.allclose(ul_diag, np.flip(np.flip(lr_diag, axis=0), axis=1)):
			n_symmetric += 1
			
	print(f"{n_symmetric}/{len(designs)} designs are symmetric")
	
	
def viz_obj_success(data, title=""):
	import json
	from scripts.preprocessing import filter_objective_success
	data = data.loc[data["result"] == "Success"]
	
	objectives = json.load(open("data/datafiles/objective_params.json", 'r'))
	
	results = np.zeros(len(objectives))
	obj_labels = []
	for idx, (obj, obj_th) in enumerate(objectives.items()):
		obj_labels.append(obj)
		obj_data = filter_objective_success(data, obj_th)
		results[idx] = len(obj_data)
		
	plt.bar(x=np.arange(len(results)), height=results, tick_label=obj_labels)
	plt.title(title)
	plt.xlabel("Objective Label")
	plt.ylabel("Count")
	plt.show()
	

if __name__ == "__main__":
	fname = "data/datafiles/preprocessed/design_database.csv"
	data = pd.read_csv(fname)
	viz_obj_success(data, title="Num. full dataset designs meeting objective thresholds")
	
	# data = data.loc[data["result"] == "Success"]
	
	# viz_metric_dists(data, "", fname)
	# viz_size_dist(data, "Original dataset component size distributions\nfor stable drones")
	
	# viz_symmetric(data["config"].values)
