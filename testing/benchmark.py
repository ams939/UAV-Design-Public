from inference.UAVSimulator import HyFormSimulator
from tqdm import trange
from inference.NNSimulator import NNSimulator
from train.Hyperparams import Hyperparams
from time import time
import pandas as pd
import numpy as np
import torch


def main():
	simulator = HyFormSimulator()
	# hparams = Hyperparams("trained_models/sim_nn/nn_hparams.json")
	# hparams.device = torch.device('cpu')
	# simulator = NNSimulator(hparams)
	data = pd.read_csv("testing/test.csv")
	
	times = []
	for idx in trange(1000):
		s_time = time()
		_ = simulator.simulate(str(data.values[idx, 0]))
		e_time = time()
		
		times.append(e_time - s_time)
	
	times = np.asarray(times)
	print(f"Mean: {np.mean(times)}, Std: {np.std(times)}, Total: {np.sum(times)}")
	

if __name__ == "__main__":
	main()
