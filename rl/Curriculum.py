from abc import ABC, abstractmethod

import numpy as np
from functools import reduce


class UAVCurriculum(ABC):
	def __init__(self, hparams):
		self.hparams = hparams
		
	@abstractmethod
	def get_objective(self, episode):
		pass
	
	@abstractmethod
	def get_all_objectives(self):
		pass

	@abstractmethod
	def get_active_objectives(self):
		pass
	
	
class FixedCurriculum(UAVCurriculum):
	""" Curriculum returning just a single objective """
	def __init__(self, hparams):
		super(FixedCurriculum, self).__init__(hparams)
		self.curr_hparams = self.hparams.curriculum_hparams
		
	def get_objective(self, ep):
		return self.curr_hparams.objective
	
	def get_all_objectives(self):
		return [self.curr_hparams.objective]

	def get_active_objectives(self):
		return self.get_all_objectives()


class EpisodicCurriculum(UAVCurriculum):
	"""
	Returns one of n objectives within an objective pool with p = 1/n. Objective pool is updated to contain more objectives
	as episodes progress. Objectives and episode thresholds at which objectives are added are specified in the 'curriculum_
	hparams' dictionary as 'obj_name': 'episode' key-val pairs. E.g

	{
		'2a': 0,
		'3b': 1000
	}

	First objective threshold must be zero
	"""
	def __init__(self, hparams):
		super(EpisodicCurriculum, self).__init__(hparams)
		self.curr_hparams = hparams["curriculum_hparams"]
		self.obj_pool = []
		self.curriculum = []
		self._init()
		
	def _init(self):
		self.obj_pool = []
		self.curriculum = []
		
		s_times = list(self.curr_hparams.items())
		assert reduce((lambda x, y: type(y[1]) == int), s_times)
		
		s_times = sorted(s_times, key=(lambda x: x[1]))
		print(s_times)
		self.curriculum = s_times
		assert self.curriculum[0][1] == 0, "Must have an objective starting at episode 0"
			
	def _update(self, ep):
		while True and (not len(self.curriculum) == 0):
			if ep >= self.curriculum[0][1]:
				obj, s = self.curriculum.pop(0)
				self.obj_pool.append(obj)
				self.hparams.logger.log({"name": "UAVCurriculum", "msg": f"Added objective {obj} to objective pool"})
			else:
				break
				
	def reset(self):
		self._init()
	
	def get_objective(self, episode):
		self._update(episode)
		r_idx = np.random.randint(len(self.obj_pool))
		return self.obj_pool[r_idx]
	
	def get_all_objectives(self):
		return list(self.curr_hparams.keys())

	def get_active_objectives(self):
		return self.obj_pool.copy()


if __name__ == "__main__":
	from train.Hyperparams import DummyHyperparams
	hparams = {"logger_class": "train.Logging.ConsoleLogger",
			   "curriculum_hparams": {
				   "1a": 0,
				   "2a": 10,
				   "3a": 20
			   }}
	
	hparams = DummyHyperparams(hparams)
	hparams.logger = hparams.logger_class()
	
	curr = UAVCurriculum(hparams)
	eps = [0, 5, 10, 11, 15, 20, 30]
	
	for ep in eps:
		print(f"Epsiode {ep}")
		ob = curr.get_objective(ep)
		print(f"Got {ob}")
	