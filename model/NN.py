import torch.nn as nn
import torch

from model.UAVModel import UAVModel
from train.Hyperparams import Hyperparams
from data.Constants import SIM_METRICS, SIM_OUTCOME
from utils.utils import HeadList


class SimNN(UAVModel):
    """
    Standard fully-connected model that can do both/either metric prediction (regression task) and outcome prediction
    (classification task)

    """
    def __init__(self, hparams: Hyperparams):
        super(SimNN, self).__init__(hparams)

        self.in_features = hparams.feature_dim
        self.dropout = hparams.dropout
        self.hidden_sizes = hparams.model_hparams["fc_hparams"]["hidden_sizes"]
        self.reg_sizes = hparams.model_hparams["reg_hparams"]["hidden_sizes"]
        self.clf_sizes = hparams.model_hparams["clf_hparams"]["hidden_sizes"]
        self.num_outcomes = hparams.num_outcomes
        self.target_cols = hparams.dataset_hparams["target_cols"]
        self.num_metrics = len(set(self.target_cols).intersection(set(SIM_METRICS)))
        self.is_classifier = SIM_OUTCOME in self.target_cols
        self.is_regressor = self.num_metrics > 0

        self.hidden = None
        self.reg_out = None
        self.clf_out = None

        self.build()

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        # Assumption: Batch is the second dimension (batch_first=false)
        x = torch.transpose(x, dim0=1, dim1=0).squeeze()

        out = self.hidden(x)

        if self.is_regressor:
            reg_pred = self.reg_out(out)
        else:
            reg_pred = None

        if self.is_classifier:
            clf_pred = self.clf_out(out)
        else:
            clf_pred = None

        return reg_pred, clf_pred

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            # self.set_device(torch.device('cpu'))  # Inference done on CPU
            reg_out, clf_out = self.forward(x)

        if reg_out is not None:
            reg_out = reg_out.detach().cpu()

        if clf_out is not None:
            clf_out = torch.round(torch.sigmoid(clf_out.detach())).type(torch.long).cpu()

        return reg_out, clf_out

    def build(self):
        self.hidden = FCNet(self.in_features, self.hidden_sizes, dropout=self.dropout)
        hidden_out = self.hidden.out_features

        if self.is_regressor:
            self.reg_out = FCNet(hidden_out, self.reg_sizes, self.num_metrics, act_out=False)

        if self.is_classifier:
            self.clf_out = FCNet(hidden_out, self.clf_sizes, self.num_outcomes, act_out=False)

        self.load()
        
        
class MHSimNN(UAVModel):
    """
    Standard fully-connected model that can do both/either metric prediction (regression task) and outcome prediction
    (classification task)

    """
    def __init__(self, hparams: Hyperparams):
        super(MHSimNN, self).__init__(hparams)

        self.in_features = hparams.feature_dim
        self.dropout = hparams.dropout
        self.hidden_sizes = hparams.model_hparams["fc_hparams"]["hidden_sizes"]
        self.reg_sizes = hparams.model_hparams["reg_hparams"]["hidden_sizes"]
        self.clf_sizes = hparams.model_hparams["clf_hparams"]["hidden_sizes"]
        self.num_outcomes = hparams.num_outcomes
        self.target_cols = hparams.dataset_hparams["target_cols"]
        self.num_metrics = len(set(self.target_cols).intersection(set(SIM_METRICS)))
        self.is_classifier = SIM_OUTCOME in self.target_cols
        self.is_regressor = self.num_metrics > 0

        self.hidden = None
        self.reg_out = None
        self.clf_out = None

        self.build()

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        # Assumption: Batch is the second dimension (batch_first=false)
        x = torch.transpose(x, dim0=1, dim1=0).squeeze()
        
        if len(x.size()) == 1:
            x = x.reshape(1, -1)

        out = self.hidden(x)

        if self.is_regressor:
            reg_pred = self.reg_out(out)
        else:
            reg_pred = None

        if self.is_classifier:
            clf_pred = self.clf_out(out)
        else:
            clf_pred = None

        return reg_pred, clf_pred

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            # self.set_device(torch.device('cpu'))  # Inference done on CPU
            reg_out, clf_out = self.forward(x)

        if reg_out is not None:
            reg_out = reg_out.detach().cpu()

        if clf_out is not None:
            clf_out = torch.round(torch.sigmoid(clf_out.detach())).type(torch.long).cpu()

        return reg_out, clf_out

    def build(self):
        self.hidden = FCNet(self.in_features, self.hidden_sizes, dropout=self.dropout)
        hidden_out = self.hidden.out_features

        if self.is_regressor:
            heads = [FCNet(hidden_out, self.reg_sizes, 1, act_out=False) for _ in range(self.num_metrics)]
            
            self.reg_out = HeadList(heads)

        if self.is_classifier:
            self.clf_out = FCNet(hidden_out, self.clf_sizes, self.num_outcomes, act_out=False)

        self.load()


class FCNet(nn.Module):
    """
    Fully connected network with custom number of hidden layers.

    Intended for use within other networks

    """
    def __init__(self, in_features, hidden_sizes, out_features=None, act_out=True, dropout=0.0):
        super(FCNet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sizes = hidden_sizes
        self.dropout = dropout
        self.act_out = act_out

        self.model = None
        self.build()

    def build(self):
        fc_net = nn.Sequential()
        fc_in = self.sizes[:-1]
        fc_in.insert(0, self.in_features)

        for idx, fc_out in enumerate(self.sizes):
            fc_net.append(nn.Linear(in_features=fc_in[idx], out_features=fc_out))
            fc_net.append(nn.ReLU())
            if 0.0 < self.dropout < 1.0:
                fc_net.append(nn.Dropout(self.dropout))

        if self.out_features is not None:
            fc_net.append(nn.Linear(in_features=self.sizes[-1], out_features=self.out_features))
            if self.act_out:
                fc_net.append(nn.ReLU())
        else:
            self.out_features = self.sizes[-1]

        self.model = fc_net

    def forward(self, x: torch.Tensor):
        return self.model(x)
