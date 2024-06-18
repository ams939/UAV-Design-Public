"""
Class implementing UAVSimulator using a GCN model

"""

from typing import List
import os

import torch

from train.Hyperparams import Hyperparams
from inference.NNSimulator import NNSimulator
from data.Constants import SIM_METRICS, SIM_OUTCOME, UAV_CONFIG_COL


""" This is an interface class for the GCNEncoder model, allowing it to function as a UAV Simulator Surrogate """
class GCNSimulator(NNSimulator):
	def __init__(self, hparams: Hyperparams):
		super(GCNSimulator, self).__init__(hparams)
		
	def simulate_batch(self, uav_str_list: List[str], return_df=True):
		raise NotImplementedError