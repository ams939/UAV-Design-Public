"""
This class defines the interface for NN models that operate on the UAV-Design problem.


"""

from abc import abstractmethod
import os
import sys

import torch
import torch.nn as nn

from train.Hyperparams import Hyperparams


class UAVModel(nn.Module):
    def __init__(self, hparams: Hyperparams):
        super(UAVModel, self).__init__()
        self.hparams = hparams
        self.logger = hparams.logger
        self.model_dir = f"{hparams.experiment_folder}"
        self.model_file = hparams.model_file

        if self.model_file is None:
            self.model_file = f"model.pt"

        self.device = hparams.device

    @property
    def __name__(self):
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args):
        pass

    def predict(self, *args):
        return self.forward(*args)

    def sample(self, n_samples):
        pass

    @abstractmethod
    def build(self):
        pass

    def set_device(self, device):
        self.device = device

    def load(self):
        m_file = f"{self.model_dir}/{self.model_file}"
        if os.path.exists(m_file):
            self.logger.log({"name": self.__name__, "msg": f"Loading existing model from {m_file}", "debug": True})
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
