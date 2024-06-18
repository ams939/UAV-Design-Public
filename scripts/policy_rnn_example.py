"""
Script demonstrating usage of PolicyRNN with the UAVDesign state

"""

import random

import torch

from model.RNN import PolicyRNN
from train.Hyperparams import Hyperparams
from rl.DesignState import UAVDesign


def main():
	# Load policy rnn hyperparams from file
	# (Alternatively, specify as dict and use DummyHyperparams)
	hyperparams_file = "hparams/policyrnn_hparams.json"
	hparams = Hyperparams(hyperparams_file)
	
	if torch.cuda.is_available() and hparams.device == "gpu":
		hparams.device = torch.device('cuda:0')
	else:
		hparams.device = torch.device('cpu')
	
	# Initialize the model
	model = PolicyRNN(hparams).to(hparams.device)
	
	# Initialize the data (random sequence of states for demo purpose)
	init_state = UAVDesign("*aMM0,0,3")
	
	seq_len = 5
	state_seq_list = [init_state]
	current_state = init_state
	for i in range(1, seq_len):
		next_states = current_state.get_successors()
		next_state = next_states[random.randint(0, len(next_states))]
		
		state_seq_list.append(next_state)
		
		current_state = next_state
	
	# Get some action for the last state
	action = state_seq_list[-1].get_successors()[0]
	
	# Convert state sequence and action to tensors, 'to_tensor' uses 'matrix' encoding method by default, which converts
	# the UAVDesign object into a 295 element long vector
	# End result is a (batch_size, seq_len, n_features) tensor
	state_seq_tensor = torch.stack([s.to_tensor() for s in state_seq_list], dim=0)
	state_seq_tensor_batch = state_seq_tensor.unsqueeze(0)  # Add the batch dimension (just one in this case)
	
	# Same deal for action, except there's just one action per sequence, so it's a (batch_size, n_features) tensor
	act_tensor = action.to_tensor()
	act_tensor_batch = act_tensor.unsqueeze(0)
	
	# Pass tensors through the net, net outputs (batch_size, out_dim) tensor
	net_out = model(state_seq_tensor_batch, act_tensor_batch)
	
	#
	# ..... Next steps, train/eval/inference/??? .....
	#
	
	print(net_out)
	
	
if __name__ == "__main__":
	main()