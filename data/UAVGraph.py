import torch
from torchdrug.data import Graph, PackedGraph

from utils.utils import graph2uav, uav2graph
from rl.DesignState import UAVDesign
from data.datamodel.Grammar import UAVGrammar


class UAVGraph(Graph):
	def __init__(self, edge_list=None, edge_weight=None, num_node=None, num_relation=None,
					node_feature=None, edge_feature=None, graph_feature=None, **kwargs):
		
		super(UAVGraph, self).__init__(edge_list, edge_weight, num_node, num_relation, node_feature, edge_feature,
										graph_feature, **kwargs)
		
		with self.node():
			self.atom_type = torch.arange(self.num_node)
			
		# graph.meta_dict["atom_type"] = {''}
	
	def to_uav_string(self, ignore_error=False):
		# TODO: Account for payload, coords (here and in graph2drone)
		try:
			comps, conns = graph2uav(self.node_feature, self.edge_list)
			uav = UAVDesign(elems=(comps, conns, 0, 0))
		except Exception as e:
			if ignore_error:
				return None
			raise e
		return uav.to_string()
	
	def is_valid(self):
		# TODO: Add some kind of validity check here
		return True
	
	def from_uav_string(self, uav_str, ignore_error=False):
		""" Mimics the torchdrug.data.Molecule.from_smiles() method """
		try:
			comps, conns, _, _ = UAVGrammar().parse(uav_str)
			feat_matrix, edge_list = uav2graph(comps, conns)
			g = UAVGraph(node_feature=feat_matrix, edge_list=edge_list, num_node=len(comps), num_relation=1)
		except Exception as e:
			if ignore_error:
				return None
			raise e
		
		return g
	

class PackedUAVGraph(PackedGraph):
	def __init__(self, edge_list=None, edge_weight=None, num_node=None, num_relation=None,
					node_feature=None, edge_feature=None, graph_feature=None, **kwargs):
		super(PackedUAVGraph, self).__init__(self, edge_list=edge_list, edge_weight=edge_weight, num_nodes=num_node,
											 num_relation=num_relation, node_feature=node_feature,
											 edge_feature=edge_feature, graph_feature=graph_feature, **kwargs)
		
		self.unpacked_type = UAVGraph