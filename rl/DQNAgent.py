"""
Rainbow Deep Q-Learning implementation from https://github.com/Kaixhin/Rainbow by Kai Arulkumaran et al.
https://github.com/Kaixhin/Rainbow/blob/master/agent.py

Modified by Aleksanteri Sladek, 27.7.2022
    - Major refactoring to fit UAV domain

"""
from __future__ import division
from typing import List, Tuple
import os

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rl.DesignState import UAVDesign
from rl.rl_utils import obj2tensor


class Agent:
    def __init__(self, hparams):
        # Number of bins in the discrete value distribution
        self.atoms = hparams.atoms
        self.args = hparams
        self.greedy = False
        
        # Upper and lower limits for the value distribution
        self.Vmin = hparams.V_min
        self.Vmax = hparams.V_max

        # The "atom", or bin, values z_i
        self.support = torch.linspace(hparams.V_min, hparams.V_max, self.atoms).to(device=hparams.device)  # Support (range) of z

        # The bin width
        self.delta_z = (hparams.V_max - hparams.V_min) / (self.atoms - 1)
        self.batch_size = hparams.batch_size
        self.n = hparams.multi_step
        self.discount = hparams.discount
        self.norm_clip = hparams.norm_clip

        self.online_net = hparams.model_class(hparams).to(device=hparams.device)

        if hparams.model_file:  # Load pretrained model if provided
            if os.path.isfile(hparams.model_file):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(hparams.model_file, map_location='cpu')

                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + hparams.model_file)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(hparams.model_file)

        self.online_net.train()
        self.target_net = hparams.model_class(hparams).to(device=hparams.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=hparams.learning_rate, eps=hparams.adam_eps)

        self.agg_seqs = hparams.agg_seqs
        self.device = hparams.device
        self.symmetric_actions = hparams.symmetric_actions if hparams.symmetric_actions is not None else False
        self.no_size = hparams.no_size if hparams.no_size is not None else False

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    def act(self, state: UAVDesign, objective=None):
        """
        Acts based on single state (no batch) and objective, returns action

        """
        
        with torch.no_grad():
            actions = state.get_successors(symmetric=self.symmetric_actions, no_size=self.no_size)
            
            state = [[state.to_string()]]

            # return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
            state_batch = self.batch_fn(state)[0]
            bin_probs = self.online_net(state_batch, obj2tensor(objective))

            # Calculation of expected Q-value (?)
            q_values = (bin_probs * self.support).sum(1)

            # Action index of action with highest expected Q-value
            action_idx = q_values.argmax(0).item()
            return actions[action_idx]

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, objective=None, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        actions = state.get_successors(symmetric=self.symmetric_actions, no_size=self.no_size)
        
        if np.random.random() < epsilon:
            self.greedy = True
            action = actions[np.random.randint(0, len(actions))]
        else:
            self.greedy = False
            action = self.act(state, objective)
        
        return action

    def learn(self, mem):
        # Sample transitions
        try:
            idxs, states, actions, returns, next_states, objectives, nonterminals, weights = mem.sample(self.batch_size)
        except RuntimeError as e:
            # Redundant, but necessary for code comprehension
            raise RuntimeError(e)
        
        states_batch = None
        obj_batch = None

        # Construct a batch of transitions at time t and chosen action a at time t
        for batch_idx in range(self.batch_size):

            s = UAVDesign(states[batch_idx][-1]).to_tensor()  # Get the state
            a = UAVDesign(actions[batch_idx]).to_tensor()  # Get the selected action

            s_a = torch.cat([s, a], dim=0)

            obj_tensor = obj2tensor(objectives[batch_idx][0]).to(self.device)

            # Append to batch
            if states_batch is not None:
                states_batch = torch.cat([states_batch, s_a.unsqueeze(0)], dim=0)
                obj_batch = torch.cat([obj_batch, obj_tensor.unsqueeze(0)], dim=0)
            else:
                states_batch = s_a.unsqueeze(0)
                obj_batch = obj_tensor.unsqueeze(0)
        
        # Get the probabilities of the actions from net output - predicted probs of online net log p(s_t, a_t; θonline)
        log_ps_a = self.online_net(states_batch.unsqueeze(0), obj_batch, log=True)  # Log probabilities log p(s_t, ·; θonline)
        # One side of the loss fn, d_t

        with torch.no_grad():
            # Take nth timestep for each batch element
            nth_state_batch = self.batch_fn(next_states)
            pns_a = torch.zeros(self.batch_size, self.atoms).to(self.device)
            
            # Iterate over each batch element
            for batch_idx in range(self.batch_size):
                # Calculate nth next state probabilities

                # Step 2: Argmax action selection using θonline (online network)
                #
                nth_state = nth_state_batch[batch_idx]
                objective = obj2tensor(objectives[batch_idx][0]).to(self.device)

                # Evaluate the Q value probs of the nth state actions using the online network
                p_ns_online = self.online_net(nth_state, objective)

                # pns_t is of shape (batch_size, n_actions, n_atoms), i.e probabilities for each z atom
                dns_n = self.support.expand_as(p_ns_online) * p_ns_online  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

                # Perform the argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                argmax_index_ns = dns_n.unsqueeze(0).sum(2).argmax(1)

                # Step 3: Q-value calculation using θtarget (target network)

                # Evaluate the Q values of the nth state actions using the target network
                self.target_net.reset_noise()  # Sample new target net noise
                p_ns_target = self.target_net(nth_state, objective)  # Probabilities p(s_t+n, ·; θtarget)

                # Select target net's Q-value probabilities using action argmax indices from online network
                # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                # i.e probabilities for the q-values of the chosen action
                pns_a[batch_idx, :] = p_ns_target[argmax_index_ns, :]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)

            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values

            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz

            # lower and upper INTEGER bounds for the b value, i.e the lower and upper bounds of the z bins
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros((self.batch_size, self.atoms), device=self.device, dtype=torch.float32)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(l)

            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))

            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        model_path = os.path.join(path, name)
        torch.save(self.online_net.state_dict(), model_path)
        self.args.model_file = model_path

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state, objective=None):
        with torch.no_grad():
            state_tensor = self.batch_fn([state])
            return (self.online_net(state_tensor[0], obj2tensor(objective)) * self.support).unsqueeze(0).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def batch_fn(self, in_batch: List[List[str]]) -> List[List[torch.Tensor]]:
        """
        Args:
            A batch size long list of seq_len long lists of states.

        Returns:
            batch: Tensor of batch_size * seq_len * n_actions, state_size + action_size (2*279*279)
            action_lens: Tensor of number of actions for each state in each sequence
        """

        batch_tensor = []
        # Go over every batch element (unfortunately need to use a loop to unpack)
        for state_seq in in_batch:
            s_a_seq_tensors = None
            
            # TODO: Only considering the last element of the sequence currently! No processing implemented for
            # TODO: self.agg_seqs = true
            if not self.agg_seqs:
                state_seq = [state_seq[-1]]
            
            # TODO: A redundant loop because state_actions_seq is sequence of length 1
            for state in state_seq:
                
                # Convert state to tensor
                state_obj = UAVDesign(state)
                state_tensor = state_obj.to_tensor()
                
                # Generate actions and also convert to tensor
                actions = torch.stack([act.to_tensor() for act in state_obj.get_successors(symmetric=self.args.symmetric_actions, no_size=self.args.no_size)])
                n_actions = len(actions)

                # Repeat the state for each action and concatenate
                state_rep = torch.repeat_interleave(state_tensor.reshape(1, -1), n_actions, dim=0)
                state_act_tensor = torch.cat([state_rep, actions], dim=1)

                if s_a_seq_tensors is None:
                    s_a_seq_tensors = state_act_tensor.unsqueeze(0)
                else:
                    s_a_seq_tensors = torch.cat([s_a_seq_tensors, state_act_tensor.unsqueeze(0)], dim=0)

            batch_tensor.append(s_a_seq_tensors)

        return batch_tensor
