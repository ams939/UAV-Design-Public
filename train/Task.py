import torch
from torchdrug import tasks
from torchdrug.layers import functional
from torch_scatter import scatter_add, scatter_max

import data
from data.UAVGraph import UAVGraph


class UAVGenerationGraphAF(tasks.AutoregressiveGeneration):
	pass


class UAVGenerationGCPN(tasks.GCPNGeneration):
	"""
	Extends the GCPNGeneration task from torchdrug; Remove Molecule dependencies and allow for generation of Graphs
	
	"""
	def __init__(self, model, node_types, **kwargs):
		super(UAVGenerationGCPN, self).__init__(model, node_types, **kwargs)
	
	@torch.no_grad()
	def generate(self, num_sample, max_resample=20, off_policy=False, max_step=30 * 2, initial_uav="*aMM0+++,0,0", verbose=0):
		"""
		Copy of the generate function from torchdrug.tasks.GCPNGeneration with Molecule dependencies removed
		
		"""
		is_training = self.training
		self.eval()
		
		graph = data.UAVGraph.UAVGraph().from_uav_string(initial_uav).repeat(num_sample)
		
		if self.device.type == "cuda":
			graph = graph.cuda(self.device)
		
		result = []
		for i in range(max_step):
			new_graph = self._apply_action(graph, off_policy, max_resample, verbose=1)
			if i == max_step - 1:
				# last step, collect all graph that is valid
				result.append(new_graph[(new_graph.num_nodes <= (self.max_node))])
			else:
				result.append(new_graph[new_graph.is_stopped | (new_graph.num_nodes == (self.max_node))])
				
				is_continue = (~new_graph.is_stopped) & (new_graph.num_nodes < (self.max_node))
				graph = new_graph[is_continue]
				if len(graph) == 0:
					break
		
		self.train(is_training)
		
		result = self._cat(result)
		return result
	
	@torch.no_grad()
	def _apply_action(self, graph, off_policy, max_resample=10, verbose=0, min_node=5):
		# action (num_graph, 4)
		
		# stopped graph is removed, initialize is_valid as False
		is_valid = torch.zeros(len(graph), dtype=torch.bool, device=self.device)
		stop_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
		node1_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
		node2_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
		edge_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
		
		for i in range(max_resample):
			# maximal resample time
			mask = ~is_valid
			if max_resample == 1:
				tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
					self._top1_action(graph, off_policy)
			else:
				tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
					self._sample_action(graph, off_policy)
			
			stop_action[mask] = tmp_stop_action[mask]
			node1_action[mask] = tmp_node1_action[mask]
			node2_action[mask] = tmp_node2_action[mask]
			edge_action[mask] = tmp_edge_action[mask]
			
			stop_action[graph.num_nodes <= 5] = 0
			# tmp add new nodes
			has_new_node = (node2_action >= graph.num_nodes) & (stop_action == 0)
			new_atom_id = (node2_action - graph.num_nodes)[has_new_node]
			new_atom_type = self.id2atom[new_atom_id]
			
			atom_type, num_nodes = functional._extend(graph.atom_type, graph.num_nodes, new_atom_type, has_new_node)
			
			# tmp cast to regular node ids
			node2_action = torch.where(has_new_node, graph.num_nodes, node2_action)
			
			# tmp modify edges
			new_edge = torch.stack([node1_action, node2_action], dim=-1)
			edge_list = graph.edge_list.clone()
			edge_list[:, :2] -= graph._offsets.unsqueeze(-1)
			is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
							   (stop_action[graph.edge2graph] == 0)
			has_modified_edge = scatter_max(is_modified_edge.long(), graph.edge2graph, dim_size=len(graph))[0] > 0
			edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]
			# tmp modify reverse edges
			new_edge = new_edge.flip(-1)
			is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
							   (stop_action[graph.edge2graph] == 0)
			edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]
			
			# tmp add new edges
			has_new_edge = (~has_modified_edge) & (stop_action == 0)
			new_edge_list = torch.stack([node1_action, node2_action, edge_action], dim=-1)[has_new_edge]
			edge_list, num_edges = functional._extend(edge_list, graph.num_edges, new_edge_list, has_new_edge)
			
			# tmp add reverse edges
			new_edge_list = torch.stack([node2_action, node1_action, edge_action], dim=-1)[has_new_edge]
			edge_list, num_edges = functional._extend(edge_list, num_edges, new_edge_list, has_new_edge)
			
			"""
			self, edge_list=None, edge_weight=None, num_node=None, num_relation=None,
					node_feature=None, edge_feature=None, graph_feature=None, **kwargs):
			
			"""
			
			# TODO: Implement validity check for graph here
			# tmp_graph = type(graph)(edge_list, num_nodes=num_nodes, num_edges=num_edges, num_relation=graph.num_relation)
			# tmp_graph.is_valid
			is_valid = stop_action == 1
			if is_valid.all():
				break
				
		if not is_valid.all() and verbose:
			num_invalid = len(graph) - is_valid.sum().item()
			num_working = len(graph)
			print("%d / %d molecules are invalid even after %d resampling" % (num_invalid, num_working, max_resample))
		
		# apply the true action
		# inherit attributes
		data_dict = graph.data_dict
		meta_dict = graph.meta_dict
		
		# pad 0 for node / edge attributes
		for k, v in data_dict.items():
			if "node" in meta_dict[k]:
				shape = (len(new_atom_type), *v.shape[1:])
				new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
				data_dict[k] = functional._extend(v, graph.num_nodes, new_data, has_new_node)[0]
			if "edge" in meta_dict[k]:
				shape = (len(new_edge_list) * 2, *v.shape[1:])
				new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
				data_dict[k] = functional._extend(v, graph.num_edges, new_data, has_new_edge * 2)[0]
		
		new_graph = type(graph)(edge_list, num_nodes=num_nodes,
								num_edges=num_edges, num_relation=graph.num_relation,
								meta_dict=meta_dict, **data_dict)
		with new_graph.graph():
			new_graph.is_stopped = stop_action == 1
		
		new_graph, feature_valid = self._update_graph_feature(new_graph)
		return new_graph[feature_valid]
	
	def _update_graph_feature(self, graphs):
		# This function is very slow
		graphs_list = [graph.to_uav_string(ignore_error=True) for graph in graphs.unpack()]
		valid = [g is not None for g in graphs_list]
		valid = torch.tensor(valid, device=graphs.device)
		
		new_graphs = type(graphs)([UAVGraph().from_uav_string(graph_str) for graph_str in graphs_list])
		
		node_feature = torch.zeros(graphs.num_node, *new_graphs.node_feature.shape[1:],
								   dtype=new_graphs.node_feature.dtype, device=graphs.device)
		edge_feature = torch.zeros(graphs.num_edge, *new_graphs.edge_feature.shape[1:],
								   dtype=new_graphs.edge_feature.dtype, device=graphs.device)
		bond_type = torch.zeros_like(graphs.bond_type)
		node_mask = valid[graphs.node2graph]
		edge_mask = valid[graphs.edge2graph]
		node_feature[node_mask] = new_graphs.node_feature.to(device=graphs.device)
		edge_feature[edge_mask] = new_graphs.edge_feature.to(device=graphs.device)
		
		with graphs.node():
			graphs.node_feature = node_feature
		with graphs.edge():
			graphs.edge_feature = edge_feature
			graphs.bond_type = bond_type
		
		return graphs, valid