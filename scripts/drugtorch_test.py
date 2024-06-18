

def main():
	from torchdrug import datasets, models, tasks, core
	import pickle
	from torch import nn, optim
	
	# dataset = datasets.ZINC250k("./", kekulize=True,
	#                            atom_feature="symbol")
	# with open("./zinc250k.pkl", "wb") as fout:
	#    pickle.dump(dataset, fout)
	
	with open("./zinc250k.pkl", "rb") as fin:
		dataset = pickle.load(fin)
	
	model = models.RGCN(input_dim=dataset.node_feature_dim,
						num_relation=dataset.num_bond_type,
						hidden_dims=[256, 256, 256, 256], batch_norm=False)
	task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
								max_node=38, criterion="nll")
	
	optimizer = optim.Adam(task.parameters(), lr=1e-3)
	solver = core.Engine(task, dataset, None, None, optimizer,
						 batch_size=32, log_interval=10)
	
	solver.train(num_epoch=1)
	solver.save("./gcpn_zinc250k_1epoch.pkl")


if __name__ == "__main__":
	main()