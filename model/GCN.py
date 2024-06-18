import sys
import os

import torch
from torchdrug.models import RGCN

from model.NN import MHSimNN, FCNet, HeadList


class GCNEncoder(RGCN):
	def __init__(self, input_dim, hidden_sizes):
		super(GCNEncoder, self).__init__(input_dim=input_dim, num_relation=1, hidden_dims=hidden_sizes)
	
	@property
	def __name__(self):
		return "GCNEncoder"
	
	def load(self):
		m_file = f"{self.model_dir}/{self.model_file}"
		if os.path.exists(m_file):
			self.logger.log({"name": self.__name__, "msg": f"Loading existing model from {m_file}"})
			state = torch.load(m_file, map_location=torch.device(self.device))
			self.load_state_dict(state)
		else:
			# Make sure that creating a new model was actually intended....
			if self.hparams.experiment_type in ['predict', 'eval', 'simulate', 'generate', 'run_dqn']:
				self.logger.log({"name": self.__name__, "msg": f"ERROR: Trying to perform experiment requiring a "
															   f"trained model, but no model found at {m_file}"})
				sys.exit(-1)
			
			self.logger.log({"name": self.__name__, "msg": f"WARNING: No existing model found, new model created."})
	
	def save(self, mname=""):
		m_file = f"{self.model_dir}/{mname}{self.model_file}"
		try:
			torch.save(self.state_dict(), m_file)
			self.logger.log({"name": self.__name__, "msg": f"Model saved to {m_file}"})
		except Exception:
			self.logger.log({"name": self.__name__, "msg": f"Error saving model."})
			
			
class SimGCN(MHSimNN):
	""" GCN Regressor and classifier """
	def __init__(self, hparams):
		super(SimGCN, self).__init__(hparams)
	
	def forward(self, x):
		out = self.hidden(x, x.node_feature.float())["graph_feature"]
		
		if self.is_regressor:
			reg_pred = self.reg_out(out)
		else:
			reg_pred = None
		
		if self.is_classifier:
			clf_pred = self.clf_out(out)
		else:
			clf_pred = None
		
		return reg_pred, clf_pred
	
	def predict(self, x):
		with torch.no_grad():
			# self.set_device(torch.device('cpu'))  # Inference done on CPU
			reg_out, clf_out = self.forward(x)
		
		if reg_out is not None:
			reg_out = reg_out.detach().cpu()
		
		if clf_out is not None:
			clf_out = torch.round(torch.sigmoid(clf_out.detach())).type(torch.long).cpu()
		
		return reg_out, clf_out
		
	def build(self):
		self.hidden = GCNEncoder(input_dim=self.in_features, hidden_sizes=self.hidden_sizes)
		hidden_out = self.hidden_sizes[-1]
		
		if self.is_regressor:
			heads = [FCNet(hidden_out, self.reg_sizes, 1, act_out=False) for _ in range(self.num_metrics)]
			
			self.reg_out = HeadList(heads)
		
		if self.is_classifier:
			self.clf_out = FCNet(hidden_out, self.clf_sizes, self.num_outcomes, act_out=False)
			
		self.load()
