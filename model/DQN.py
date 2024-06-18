"""
Rainbow Deep Q-Learning implementation from https://github.com/Kaixhin/Rainbow by Kai Arulkumaran et al.

Modified by Aleksanteri Sladek
    - Generalized the DQN class input layer
    - Moved layer initialization into a build function
    - Refactoring
"""

from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from model.NN import FCNet


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args
        self.atoms = args.atoms
        self.action_space = args.action_space

        # Initialize layers
        self.in_layer = None
        self.in_output_size = None  # Size of the output of the input layer
        self.rw_in_layer = None
        
        self.fc_h_v_1 = None
        self.fc_h_a_1 = None
        
        self.fc_h_v_2 = None
        self.fc_h_a_2 = None
        
        self.fc_h_v_3 = None
        self.fc_h_a_3 = None
        
        self.fc_z_v = None
        self.fc_z_a = None

        self.build()

    def build(self):
        if self.args.architecture == 'canonical':
            self.in_layer = nn.Sequential(nn.Conv2d(self.args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                          nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
            self.in_output_size = 3136
        elif self.args.architecture == 'data-efficient':
            self.in_layer = nn.Sequential(nn.Conv2d(self.args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                          nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
            self.in_output_size = 576

        # Hidden FC layer
        self.fc_h_v = NoisyLinear(self.in_output_size, self.args.hidden_size, std_init=self.args.noisy_std)
        self.fc_h_a = NoisyLinear(self.in_output_size, self.args.hidden_size, std_init=self.args.noisy_std)

        # Output layer
        self.fc_z_v = NoisyLinear(self.args.hidden_size, self.atoms, std_init=self.args.noisy_std)
        self.fc_z_a = NoisyLinear(self.args.hidden_size, self.action_space * self.atoms, std_init=self.args.noisy_std)

    def forward(self, x, log=False):
        x = self.in_layer(x)
        x = x.view(-1, self.in_output_size)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


class UAVDQN(DQN):
    """
    Extension of the DQN class to accommodate the UAV MDP problem

    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = hparams.device
        self.agg_seqs = hparams.agg_seqs
        self.in_features = hparams.encoding_size * 2
        self.rw_out = 16

        # Noisy FC params
        self.fc_hparams = self.hparams.model_hparams["noisy_fc_hparams"]
        self.fc_hidden_size = self.fc_hparams["hidden_size"]
        self.fc_noisy_std = self.fc_hparams["noisy_std"]

        super(UAVDQN, self).__init__(hparams)

    def build(self):
        # UAV input layer
        self.in_layer = FCNet(self.in_features, self.hparams.model_hparams.fc_hparams.hidden_sizes[:-1],
                              out_features=self.hparams.model_hparams.fc_hparams.hidden_sizes[-1] - self.rw_out)
        self.in_output_size = self.in_layer.out_features

        # Reward function parameter input layer
        self.rw_in_layer = FCNet(3, [64, 64, self.rw_out])
        
        # Hidden FC layers
        self.fc_h_v_1 = NoisyLinear(self.in_output_size + self.rw_out, 256 - self.rw_out, std_init=self.fc_noisy_std)
        self.fc_h_a_1 = NoisyLinear(self.in_output_size + self.rw_out, 256 - self.rw_out, std_init=self.fc_noisy_std)

        self.fc_h_v_2 = NoisyLinear(256, 128 - self.rw_out, std_init=self.fc_noisy_std)
        self.fc_h_a_2 = NoisyLinear(256, 128 - self.rw_out, std_init=self.fc_noisy_std)
        
        self.fc_h_v_3 = NoisyLinear(128, self.fc_hidden_size, std_init=self.fc_noisy_std)
        self.fc_h_a_3 = NoisyLinear(128, self.fc_hidden_size, std_init=self.fc_noisy_std)

        # Output layer
        self.fc_z_v = NoisyLinear(self.fc_hidden_size, self.atoms, std_init=self.fc_noisy_std)

        # Note change from DQN, output size (num neurons) is just N atoms, not N actions * N atoms
        self.fc_z_a = NoisyLinear(self.fc_hidden_size, self.atoms, std_init=self.fc_noisy_std)

    def forward(self, x, rparams=None, log=False):
        """
        Args:
            x : A list of tensors of inputs of shape (n_steps, n_actions, num_features)
            rparams: Reward function parameters (tensor of shape batch_size, 3)
        Returns:
            q - Note, these are (log-)probabilities

        """
        
        # Take the last timestep (either t, or t+n depending on what is passed)
        x = x[-1].to(self.args.device)
        x = self.in_layer(x)
        x = x.view(-1, self.in_output_size)
        
        assert rparams is not None
        rparams = rparams.to(self.args.device)
        
        if len(rparams.size()) == 1:
            rparams = rparams.reshape(1, -1)
        
        rw_z = self.rw_in_layer.forward(rparams)

        batch_size = x.shape[0]  # Note, this may be equivalent to n_actions for some forward passes
        
        # Append the reward parameters to x
        if rparams.size()[0] == 1:
            rw_z = rw_z.repeat((batch_size, 1))
        else:
            assert rparams.size()[0] == batch_size
            
        x = torch.hstack((x, rw_z))
        
        # Val. Adv. layer 1
        v_h = F.relu(self.fc_h_v_1(x))
        a_h = F.relu(self.fc_h_a_1(x))
        
        # Append reward to 2nd hidden layer
        v_h = torch.hstack((v_h, rw_z))
        a_h = torch.hstack((a_h, rw_z))

        # Val. Adv. layer 2
        v_h = F.relu(self.fc_h_v_2(v_h))
        a_h = F.relu(self.fc_h_a_2(a_h))

        # Append reward to 3rd hidden layer
        v_h = torch.hstack((v_h, rw_z))
        a_h = torch.hstack((a_h, rw_z))

        # Val. Adv. layer 3
        v_h = F.relu(self.fc_h_v_3(v_h))  # Value stream
        a_h = F.relu(self.fc_h_a_3(a_h))  # Advantage stream
        
        # Val. Adv. out layer
        v = self.fc_z_v(v_h)
        a = self.fc_z_a(a_h)
        
        v, a = v.view(-1, 1, self.atoms), a.view(-1, batch_size, self.atoms)

        # State value + (State,Action Advantage value - mean Advantage over all Actions for State)
        a = a - a.mean(1, keepdim=True)  # Combine streams

        q = v.squeeze() + a.squeeze()
        
        # Special case for singleton batches
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        if log:  # Use log softmax for numerical stability
            out = F.log_softmax(q, dim=1)  # Log probabilities with action over second dimension
        else:
            out = F.softmax(q, dim=1)

        return out
