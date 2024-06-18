import importlib
import time
from typing import Tuple, List
from copy import copy, deepcopy

import numpy as np
import torch
from torch.nn.functional import one_hot

from data.Constants import CAPACITY_SCALING, SIZE_SCALING, COMPONENT_IDS, COORD_GRID_SIZE, COMPONENT_TYPE_IDS, \
    X_LETTER_COORDS, Z_LETTER_COORDS, EOS_TOKEN, SOS_TOKEN, TOKEN_TO_IDX
from data.datamodel.Grammar import UAVGrammar
from utils.graphs import UAVGraph, dijkstra


def idx_to_onehot(idx_list: torch.Tensor, max_idx: int) -> torch.Tensor:
    """
    Function for converting vocabulary index to a onehot encoding

    """
    # Single tensor case
    if 0 < len(idx_list.shape) < 2:

        seq_len = len(idx_list)

        idx_one_hot = torch.zeros(seq_len, max_idx, dtype=torch.float32)

        idx_one_hot[torch.arange(seq_len, dtype=torch.long), idx_list] = 1.0

    # Batch case
    else:
        idx_one_hot = one_hot(idx_list, num_classes=max_idx)

    return idx_one_hot


def onehot_to_idx(onehot_idx_list: torch.Tensor):
    """
    Args:
        onehot_idx_list: Tensor of indices in one-hot encoding format, shape (seq_len, vocab_size)

    Returns:
        Tensor of indices, shape (seq_len)

    """
    return torch.argmax(onehot_idx_list, dim=1)


def load_class(class_str):
    try:
        class_str_elems = class_str.split(".")
        module = importlib.import_module(".".join(class_str_elems[:-1]))
        cls_name = class_str_elems[-1]
        cls = getattr(module, cls_name)
    except Exception as e:
        print(e)
        cls = None

    return cls


def generate_experiment_id(prefix=None, postfix=None):
    exp_id = f"{time.strftime('%m%d%y%H%M%S', time.localtime())}"

    if prefix is not None:
        exp_id = f"{prefix}_{exp_id}"

    if postfix is not None:
        exp_id = f"{exp_id}_{postfix}"

    return exp_id


def prepended_join(lst, c):
    """
        Function from Sebastiaan's 'UAV design' jupyter notebook

        Similar to char.join - joins a list of strings $l by interleaving chars $c. But this
        one also adds $c at the front of the joined string, so that every element of $l is
        prepended by $c
    """

    if len(lst) == 0:
        return ""
    return c + c.join(lst)


class ddict(dict):
    """
    Dictionary class supporting dot notation

    Author: Stackoverflow user 'epool' (https://stackoverflow.com/users/845296/epool)
    Source: https://stackoverflow.com/a/32107024

    16.10.22 Added deepcopy support: 
    Author: Alex Hall (https://stackoverflow.com/users/2482744/alex-hall)
    Source: https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict 

    """
    def __init__(self, *args, **kwargs):
        super(ddict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = ddict(v)
                    else:
                        self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ddict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ddict, self).__delitem__(key)
        del self.__dict__[key]

    def __deepcopy__(self, memo=None):
        return ddict(deepcopy(dict(self), memo=memo))


def json_bool_parser(json_struct, key='root'):
    """
    Recursively searches through dictionary for values that are 'True' or 'False', i.e booleans in string format, then
    changes them to actual booleans.
    """
    if isinstance(json_struct, dict):
        for k, v in json_struct.items():
            json_struct[k] = json_bool_parser(v, key=k)
    elif isinstance(json_struct, list):
        return [json_bool_parser(l_elem) for l_elem in json_struct]
    elif json_struct in ["True", "T", "true"]:
        return True
    elif json_struct in ["False", "F", "false"]:
        return False
    else:
        return json_struct

    return json_struct


def json_class_parser(json_struct, key='root'):
    """
    Recursively searches through dictionary for keys with '_class' in their name, and converts their values to a class
    object
    """

    # Recursively parse sub-dictionaries
    if isinstance(json_struct, dict):
        for k, v in json_struct.items():
            json_struct[k] = json_class_parser(v, key=k)
    elif isinstance(json_struct, list):
        return [json_class_parser(l_elem) for l_elem in json_struct]
    elif '_class' in key:
        return load_class(json_struct)

    return json_struct


def hparams_serialize(hparams, key='root'):
    # Convert the hparams object into something that can be dumped by json module
    if key == 'root':
        hparams = deepcopy(hparams)

    if isinstance(hparams, dict):
        for hparam_name, value in hparams.items():
            hparams[hparam_name] = hparams_serialize(value, key=hparam_name)
    elif isinstance(hparams, list):
        return [hparams_serialize(l_elem) for l_elem in hparams]

    elif "_class" in key:
        try:
            return f"{hparams.__module__}.{hparams.__name__}"
        except TypeError:
            return ""
        except AttributeError:
            return ""
    elif "device" in key:
        try:
            return hparams.type
        except AttributeError:
            return hparams
    elif key == "logger":
        return f"{hparams.__module__}.{hparams.__name__}"
    
    elif isinstance(hparams, torch.Tensor):
        return f"{hparams.item()}"

    return hparams


def make_mask(tensor_lengths: torch.Tensor, mask_shape: torch.Size, batch_first=False):
    """
    Fills a tensor of shape 'mask_shape' with zeros and ones. Ones occupy indices up to index specified in
    'tensor_lengths'

    """

    mask = torch.zeros(mask_shape)
    if batch_first:
        batch_size, seq_len, n_features = mask.shape
    else:
        seq_len, batch_size, n_features = mask.shape

    for idx in range(batch_size):
        tensor_len = tensor_lengths[idx]

        if batch_first:
            mask[idx, :tensor_len, :] = 1
        else:
            mask[:tensor_len, idx, :] = 1

    return mask


def uav_str_to_mat(uav_str: str, encode_connections=True, encoding=None) -> torch.Tensor:
    """
    Function for encoding a uav string to a matrix representation, that is flattened into a vector

    """

    parser = UAVGrammar()

    components, connections, payload, _ = parser.parse(uav_str)

    if not encode_connections:
        connections = []

    # set up component and size matrices
    id_to_loc = {}
    x_size, z_size = COORD_GRID_SIZE
    n_component_types = len(COMPONENT_TYPE_IDS)

    components_matrix = np.zeros((x_size, z_size, n_component_types))
    component_size_matrix = np.zeros((x_size, z_size, 1))

    # process components
    for c in components:
        comp_id = c[0]
        comp_x, comp_z = X_LETTER_COORDS.index(c[1]), Z_LETTER_COORDS.index(c[2])
        comp_type = COMPONENT_TYPE_IDS.index(c[3])

        id_to_loc[comp_id] = tuple([comp_x, comp_z])

        try:
            comp_size = c[4:]
        except IndexError:
            comp_size = []

        # Encode the component type and location
        components_matrix[comp_x, comp_z, comp_type] = 1.0

        # Encode the size of the component
        ps = comp_size.count("+")
        ns = comp_size.count("-")
        if (ps > 0) and (ns == 0):
            size = ps / SIZE_SCALING
        elif (ps == 0) and (ns > 0):
            size = -ns / SIZE_SCALING
        else:
            size = 0

        # Encode the component_size
        component_size_matrix[comp_x, comp_z] = size

    uav_vector = np.concatenate([components_matrix, component_size_matrix], axis=2).flatten()

    if encode_connections:
        # Encode the connections as an adjacency matrix
        if encoding == "adjacency":
            connections_matrix = adjacency_encode(connections)
        else:
            connections_matrix = dir_encode(connections, id_to_loc)

        uav_vector = np.concatenate([
            uav_vector,
            connections_matrix.flatten()
        ])

    # Encode the capacity of the drone
    capacity = payload / CAPACITY_SCALING

    uav_vector = np.concatenate([
        uav_vector,
        np.asarray([capacity])
    ])

    return torch.Tensor(uav_vector)


def adjacency_encode(connections: List[str]) -> np.ndarray:
    """
    Encodes the connections as an adjacency matrix

    """
    connections_matrix = np.zeros((len(COMPONENT_IDS), len(COMPONENT_IDS)))
    for c in connections:
        orig = c[0]
        dest = c[1]

        orig_idx = COMPONENT_IDS.index(orig)
        dest_idx = COMPONENT_IDS.index(dest)

        connections_matrix[orig_idx, dest_idx] = 1.0

    return connections_matrix


def dir_encode(connections: List[str], id_to_loc: dict):
    connections_matrix = np.zeros((7, 7))

    for c in connections:
        cp1_x, cp1_z = id_to_loc[c[0]]
        cp2_x, cp2_z = id_to_loc[c[1]]

        conn_x = np.sort([cp1_x, cp2_x])
        conn_z = np.sort([cp1_z, cp2_z])

        if cp1_x != cp2_x:
            connections_matrix[conn_x[0]:conn_x[1]+1, min(conn_z)] = 1.0

        if cp1_z != cp2_z:
            connections_matrix[min(conn_x), conn_z[0]:conn_z[1] + 1] = 1.0

    return connections_matrix


def uav_string_to_matrix(uav_string: str) -> Tuple:
    """
    Function for encoding an UAV string into a numpy ndarray

    Author: Sebastiaan De Peuter

    Adapted by Aleksanteri Sladek
    """

    s = uav_string.split(",")
    capacity = int(s[1]) / CAPACITY_SCALING
    connections = s[0].split("^")
    components = connections[0].split("*")[1:]
    connections = connections[1:]

    # track locations
    ID_to_loc = {}

    # set up component and size matrices
    sizes = np.zeros((7, 7))
    structures = np.zeros((7, 7))
    motor_cws = np.zeros((7, 7))
    motor_ccws = np.zeros((7, 7))
    foils = np.zeros((7, 7))
    batteries = np.zeros((7, 7))
    x_dir_connections = np.zeros((6, 7))
    z_dir_connections = np.zeros((7, 6))

    # intial state
    if len(components) == 0:
        return sizes, structures, motor_cws, motor_ccws, foils, batteries, x_dir_connections, z_dir_connections, capacity

    # process components
    for c in components:
        # letter meanings (starting from 0)
        # 0   : ID
        # 1   : x pos (first dim in matrices)
        # 2   : z pos (second dim in matrices)
        # 3   : type
        # 4...: size
        assert ord(c[1]) in range(ord('J'), ord('P') + 1), "incorrect component x pos in UAV string"
        assert ord(c[2]) in range(ord('J'), ord('P') + 1), "incorrect component y pos in UAV string"
        loc = (ord('P') - ord(c[1]), ord(c[2]) - ord('J'))

        ID_to_loc[c[0]] = loc

        if c[3] == '0':
            structures[loc] = 1.0
        elif c[3] == '1':
            motor_cws[loc] = 1.0
        elif c[3] == '2':
            motor_ccws[loc] = 1.0
        elif c[3] == '3':
            foils[loc] = 1.0
        elif c[3] == '4':
            batteries[loc] = 1.0
        else:
            assert False, "incorrect component type " + c[3] + " in UAV string"

        ps = c[4:].count("+")
        ns = c[4:].count("-")
        if (ps > 0) and (ns == 0):
            sizes[loc] = ps / SIZE_SCALING
        elif (ps == 0) and (ns > 0):
            sizes[loc] = -ns / SIZE_SCALING
        elif (ps == 0) and (ns == 0):
            # leave size at 0
            pass
        else:
            assert False, "incorrect component size specification " + c[4:] + " in UAV string"

    for c in connections:
        orig = c[0]
        dest = c[1]

        assert orig in ID_to_loc.keys(), "nonexistent component in connections in UAV string"
        assert dest in ID_to_loc.keys(), "nonexistent component in connections in UAV string"

        cp1 = ID_to_loc[c[0]]
        cp2 = ID_to_loc[c[1]]

        if cp1[1] == cp2[1]:
            # connection in x direction
            x_dir_connections[min(cp1[0], cp2[0]), cp1[1]] = 1.0

        elif cp1[0] == cp1[0]:
            # connection in z direction
            z_dir_connections[cp1[0], min(cp1[1], cp2[1])] = 1.0
        else:
            # illegal connection
            assert False, "illegal connection in UAV string"

    return sizes, structures, motor_cws, motor_ccws, foils, batteries, x_dir_connections, z_dir_connections, capacity


def uav_string_to_vec(uav_string: str) -> torch.Tensor:
    """
    Function for encoding an UAV string into a tensor

    Author: Sebastiaan De Peuter

    """
    sizes, structures, motor_cws, motor_ccws, foils, batteries, x_dir_connections, z_dir_connections, capacity = \
        uav_string_to_matrix(uav_string)

    return torch.tensor(np.hstack(
        [sizes.flatten(), structures.flatten(), motor_cws.flatten(), motor_ccws.flatten(), foils.flatten(),
         batteries.flatten(), x_dir_connections.flatten(), z_dir_connections.flatten(), capacity]))


def uav_str_to_count_vec(uav_str: str) -> torch.Tensor:
    parser = UAVGrammar()

    uav_vector = np.zeros(len(COMPONENT_TYPE_IDS))
    components, _, payload, _ = parser.parse(uav_str)

    for c in components:
        c_type = c[3]
        c_type_idx = COMPONENT_TYPE_IDS.index(c_type)
        uav_vector[c_type_idx] = uav_vector[c_type_idx] + 1.0

    # uav_vector[-1] = payload

    return torch.Tensor(uav_vector)


def result_csv(results_json):
    from data.Constants import SIM_METRICS, SIM_OUTCOME, EVAL_METRICS

    header_str = []
    val_str = []
    for set_name in ["train", "val", "test"]:
        set_results = {}
        if set_name in results_json.keys():
            set_results = results_json[set_name]

        for metric in EVAL_METRICS[:-1]:
            for feature_name in SIM_METRICS:
                if feature_name in set_results.keys():
                    val = str(set_results[feature_name][metric])
                else:
                    val = ""

                header_str.append(f"{set_name}_{feature_name}_{metric}")
                val_str.append(val)

        if EVAL_METRICS[-1] in set_results[SIM_OUTCOME]:
            header_str.append(f"{set_name}_{SIM_OUTCOME}_{EVAL_METRICS[-1]}")
            val_str.append(str(set_results[SIM_OUTCOME][EVAL_METRICS[-1]]))

    return ",".join(header_str), ",".join(val_str)


def json_to_csv(json_obj) -> str:
    if isinstance(json_obj, dict):
        csv_str = []

        for key, val in json_obj.items():
            if not isinstance(val, dict):
                val = str(val)

            obj_str = json_to_csv(val)

            csv_str.append(obj_str)

        return ",".join(csv_str)
    else:
        return json_obj


def all_connected(uav_str: str):
    """
    Function for checking that all components in a uav string are connected to one another by a path of one or more
    edges
    """

    g = UAVGraph(uav_str)

    # Get distances to all other nodes from node 0, unreachable nodes encoded as -1
    dists = dijkstra(g, 0)
    del dists[0]

    return np.all(np.asarray(list(dists.values())) != -1)


def uav_str_to_index(uav_str: str) -> torch.Tensor or None:
    # Remove double-quotes
    uav_str = uav_str.replace('"', '')
    # Remove whitespace
    uav_str = uav_str.replace(" ", "")
    # Turn the string into a list of characters
    uav_str_elems = list(uav_str)

    # Add the NN tokens for start of sequence and end of sequence (SOS, EOS)
    uav_str_elems.insert(0, SOS_TOKEN)
    uav_str_elems.append(EOS_TOKEN)

    try:
        uav_idx_list = torch.Tensor([int(TOKEN_TO_IDX[c]) for c in uav_str_elems]).type(torch.long)
    except ValueError as e:
        print(e)
        return None
    return uav_idx_list


def remap_components(components, connections):
    new_components = []
    new_connections = []
    
    old_new_map = {"a": "a"}
    
    last_idx = 0
    components.sort(key=lambda x: COMPONENT_IDS.index(x[0]))
    for idx, c in enumerate(components):
        exp_next_idx = last_idx
        true_next_idx = COMPONENT_IDS.index(c[0])
        
        if exp_next_idx != true_next_idx:
            old_new_map[c[0]] = COMPONENT_IDS[exp_next_idx]
        last_idx += 1
        
    # Remap the component and connection names
    for c in components:
        c_new = c
        for k, v in old_new_map.items():
            if c[0] == k:
                c_new = f"{v}{c[1:]}"
                break

        new_components.append(c_new)
    
    for c in connections:
        c_new = c
        for k, v in old_new_map.items():
            
            if c[0] == k:
                c_new = f"{v}{c[1]}"
                
            if c[1] == k:
                c_new = f"{c_new[0]}{v}"
            
        new_connections.append(c_new)
        
    return new_components, new_connections
        
        
class HeadList(torch.nn.ModuleList):
    def __init__(self, modules: List):
        super(HeadList, self).__init__(modules)
        self.n_heads = len(modules)
        
    def forward(self, x_in):
        out = None
        for i in range(self.n_heads):
            out_i = self[i].forward(x_in)
            
            if out is None:
                out = out_i
            else:
                out = torch.concat([out, out_i], dim=1)
                
        return out
    
    
def uav2graph(components: List[str], connections: List[str], features=("type", "size")) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for converting a parsed UAV string into a graph representation. Graph is represented as a list of edges
    and a feature matrix. Features given in features argument added into the feature matrix. Row of feature matrix
    corresponds to UAV component ID. Edge list denotes edges between feature matrix nodes (node_in idx, node_out idx,
    relation_type).
    
    NOTE: For UAVs only one relation_type for edges
    
    Returns:
        feature_matrix: [len(components), len(features)] matrix
        edge_list: [len(connections), 3] matrix
    """
    n_edges = len(connections)
    
    # edges are of form (node_in, node_out, relation)
    edge_list = np.zeros((n_edges, 3))
    n_feats = 2
    feat_matrix = np.zeros((len(components), n_feats))
    comp_ids = []
    
    for idx, c in enumerate(components):
        comp_type = COMPONENT_TYPE_IDS.index(c[3])
        comp_ids.append(c[0])
    
        ps = c[4:].count("+")
        ns = c[4:].count("-")
        if (ps > 0) and (ns == 0):
            comp_size = ps / SIZE_SCALING
        elif (ps == 0) and (ns > 0):
            comp_size = -ns / SIZE_SCALING
        elif (ps == 0) and (ns == 0):
            # leave size at 0
            comp_size = 0
        else:
            assert False, "incorrect component size specification " + c[4:] + " in UAV string"
    
        feat_matrix[idx, features.index("type")] = comp_type
        feat_matrix[idx, features.index("size")] = comp_size
    
    for c_idx, c in enumerate(connections):
        node_in = c[0]
        node_out = c[1]
        
        assert node_in in COMPONENT_IDS, f"Unknown node ID {node_in}"
        assert node_out in COMPONENT_IDS, f"Unknown node ID {node_out}"
        
        node_in_id = comp_ids.index(node_in)
        node_out_id = comp_ids.index(node_out)
        
        edge_list[c_idx, :2] = np.asarray([node_in_id, node_out_id])
        
    return feat_matrix, edge_list


def graph2uav(feat_matrix: np.ndarray, edge_list: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    Inverse function for drone2graph, returns lists of components and connections as defined by the graph structure
    
    TODO: Implement and use as a sanity check for drone2graph functionality
    TODO: Deal with coordinates somehow?
    """
    
    component_ids = np.asarray(COMPONENT_IDS)[:len(feat_matrix)]
    component_types = np.asarray(COMPONENT_TYPE_IDS)[feat_matrix[:, 0].flatten().astype(int)]
    component_sizes = (feat_matrix[:, 1] * SIZE_SCALING).astype(int)
    
    components = []
    for idx, comp_id in enumerate(component_ids):
        size = component_sizes[idx]
        if size < 0:
            size = abs(size) * "-"
        elif size > 0:
            size = size * "+"
        else:
            size = ""
            
        # TODO: NOTE THE PLACEHOLDER COORDS
        comp = f"{comp_id}XY{component_types[idx]}{size}"
        components.append(comp)
    
    connections = []
    for edge in edge_list:
        c = f"{component_ids[int(edge[0])]}{component_ids[int(edge[1])]}"
        connections.append(c)
        
    return components, connections
    
    
if __name__ == "__main__":
    default_uav = "*aMM0+++++*bNM2+++*cMN1+++*dLM2+++*eML1+++^ab^ac^ad,5,3"
    
    comps, conns, _, _ = UAVGrammar().parse(default_uav)
    feat_mat, edge_list = uav2graph(comps, conns, features=['type', 'size'])
    comps, conns = graph2uav(feat_mat, edge_list)
    uav = f"*{'*'.join(comps)}^{'^'.join(conns)}"
    print(uav)


