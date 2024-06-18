import os
from typing import List

import numpy as np

from inference.UAVSimulator import UAVSimulator
from train.Hyperparams import Hyperparams


class RFSimulator(UAVSimulator):
	def __init__(self, hparams: Hyperparams, debug=False):
		self.hparams = hparams
		
		# Make sure a trained model exists!
		self.model_file = f"{self.hparams.experiment_folder}/{self.hparams.model_file}"
		assert os.path.exists(
			self.model_file), f"Error, couldn't load {self.hparams.experiment_folder}/{self.hparams.model_file}" \
						 "\nMake sure to specify the correct path and file in " \
						 "hparams.experiment_folder and hparams.model_file, s.t the model is found " \
						 "in hparams.experiment_folder/hparams.model_file"
		self.uav_parser = None
		self.tgt_cols = None
		super(RFSimulator, self).__init__(debug)
		
	def _init_simulator(self):
		self.model = self.hparams.model_class(self.hparams)
		
		# Load the model from the file
		self.model.load()
		
		self.dataset = self.hparams.dataset_class(self.hparams, load=False, verbose=False)
	
	def simulate(self, uav_str: str):
		return self.simulate_batch([uav_str])[0]
	
	def simulate_batch(self, uav_str_list: List[str]):
		"""
		Returns json object containing sim results for each string
		
		"""
		# Convert the UAV strings into a format suitable for the RF via the UAVDataset object
		if len(uav_str_list) == 1:
			uav_tensors = np.expand_dims(self.dataset.preprocessor.parse_design(uav_str_list[0]).numpy(), axis=0)
		else:
			uav_tensors = np.asarray([self.dataset.preprocessor.parse_design(uav_str).numpy() for uav_str in uav_str_list])
		
		y_preds = self.model.predict(uav_tensors)
		
		sim_results = []
		for i in range(len(y_preds)):
			result = dict()
			for idx, key in enumerate(self.model.targets):
				result["config"] = uav_str_list[i]
				result[key] = y_preds[i, idx]
			sim_results.append(result)
		return sim_results
		

if __name__ == "__main__":
	uavs = [
		"*aMM0++++*bNM1++*cMN2++*dLM1++*eML2++*fLN0++*gNL0++*hOM1+++*iMO2+++*jMK2+++*kKM1+++^ab^ac^ad^ae^df^eg^bh^ci^ej^dk^fc^gb,35,3",
		"*aMM0-*bNM1++*cMN1++*dLM2++*eML2++*fNL3*gLN3^ab^ac^ad^ae^ef^dg,20,3"
		]
	hparams = Hyperparams("hparams/rf_hparams.json")
	hparams.model_file = "rf_model.jb"
	hparams.experiment_folder = "experiments/SimRF"
	sim = RFSimulator(hparams)
	print(sim.simulate_batch(uavs))
