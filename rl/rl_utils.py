import torch
import numpy as np

from rl.DesignState import UAVDesign
from inference.UAVSimulator import UAVSimulator
from data.Constants import SIM_SUCCESS, METRIC_NORM_FACTORS, RANGE_COL, VELOCITY_COL, COST_COL


def get_stable_successors(uav_str: str, simulator: UAVSimulator):
	results = sim_successors(uav_str, simulator)
	
	stable_successors = []
	for idx, result in enumerate(results):
		if result["result"] == SIM_SUCCESS:
			stable_successors.append(result)
			
	return stable_successors


def sim_successors(uav_str: str, simulator: UAVSimulator):
	uav_design = UAVDesign(uav_str)
	successors = uav_design.get_successors()
	
	successor_strings = [succ.to_string() for succ in successors]
	
	results = simulator.simulate_batch(successor_strings, return_df=False)
	
	return results


def normalize_metrics(metrics: dict):
	norm_metrics = {}
	
	for metric_col in metrics.keys():
		metric_val = metrics[metric_col]
		try:
			metric_min = METRIC_NORM_FACTORS[metric_col]["min"]
			metric_max = METRIC_NORM_FACTORS[metric_col]["max"]
		except KeyError:
			continue
		
		norm_metrics[metric_col] = min_max_norm(metric_val, metric_min, metric_max)
		
	return norm_metrics


def inverse_normalize_metrics(norm_metrics: dict):
	metrics = {}
	for metric_col in norm_metrics.keys():
		metric_val = norm_metrics[metric_col]
		metric_min = METRIC_NORM_FACTORS[metric_col]["min"]
		metric_max = METRIC_NORM_FACTORS[metric_col]["max"]
		metrics[metric_col] = inverse_min_max_norm(metric_val, metric_min, metric_max)
	
	return metrics


def min_max_norm(x, x_min, x_max):
	x_trunc = max(min(x, x_max), x_min)
	
	return (x_trunc - x_min) / (x_max - x_min)


def inverse_min_max_norm(x, x_min, x_max):
	return x*(x_max - x_min) + x_min


def obj2tensor(objective):
	t = torch.zeros(3)
	
	if objective is None:
		return t
	
	for idx, obj in enumerate([RANGE_COL, COST_COL, VELOCITY_COL]):
		if obj in objective:
			t[idx] = float(objective[obj])
			
	# assert not torch.all(t == 0.0).item()
	return t
