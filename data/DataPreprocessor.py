"""
UAVDesignPreprocessor - Responsible for turning UAV Grammar strings into formats
usable by NNs

Author: Aleksanteri Sladek
20.6.2022

"""
from abc import ABC
from typing import List, Tuple, Any
from copy import copy
import sys
import os

import torch
import pandas as pd
import numpy as np
from tqdm import trange

from data.Constants import *
from utils.utils import idx_to_onehot, uav_string_to_vec, uav_str_to_mat, uav_str_to_count_vec, uav2graph
from data.datamodel.Grammar import UAVGrammar


class UAVPreprocessor(ABC):
    def __init__(self):
        self.ch_to_idx = copy(TOKEN_TO_IDX)
        self.idx_to_ch = copy(IDX_TO_TOKEN)
        self.vocab_size = VOCAB_SIZE
        self.uav_design_col = UAV_STR_COL

    def preprocess(self, raw_data) -> Any:
        """
        The preprocessing pipeline

        """

        # Extract needed data from the raw data
        dataset = self.collect_dataset(raw_data)

        # Transform the extracted data into desired format
        dataset = self.transform_dataset(dataset)

        # Return the preprocessed dataset
        return dataset

    def collect_dataset(self, dataset):
        return dataset

    def transform_dataset(self, dataset):
        return dataset

    def get_designs(self, raw_data: pd.DataFrame) -> List[str] or None:
        try:
            uav_str_list = list(raw_data[self.uav_design_col].values)
        except KeyError:
            print(f"Could not find column '{self.uav_design_col}' containing the UAV string")
            return None

        return uav_str_list

    def get_ch_idx(self, c: str) -> int:
        """
        Returns index assigned to given character
        """
        try:
            idx = self.ch_to_idx[c]
        except KeyError:
            raise ValueError(f"Out-of-vocab character: {c}")
        return idx

    def get_idx_ch(self, idx: int) -> str:
        """
        Returns character assigned to given index
        """
        try:
            ch = self.idx_to_ch[idx]
        except KeyError:
            raise ValueError(f"Out-of-range index: {idx}")
        return ch

    def parse_design(self, uav_str: str) -> torch.Tensor or None:
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
            uav_idx_list = torch.Tensor([int(self.get_ch_idx(c)) for c in uav_str_elems]).type(torch.long)
        except ValueError as e:
            print(e)
            return None

        return uav_idx_list


class UAVDesignPreprocessor(UAVPreprocessor):
    """
    Class for preprocessing string representations of UAV designs into numerical representations for an NN

    Converts each character in the UAV string into a unique index, determined by conversion dictionary in Constants.
    Also adds unique NN specific tokens (EOS, SOS) to the string

    """

    def __init__(self, one_hot=True):
        self.one_hot = one_hot
        super(UAVDesignPreprocessor, self).__init__()

    def transform_dataset(self, uav_tensor_list: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transforms the raw dataset into input-target sequence pairs, where the target sequence is identical to the
        input sequence except for being shifted forwards by one index.

        """
        uav_item_list = []
        for uav_idxs in uav_tensor_list:
            # Extract input and target sequences
            in_seq = uav_idxs[:-1]
            tgt_seq = uav_idxs[1:]

            # Convert input sequence to one-hot encoding if requested
            if self.one_hot:
                in_seq = idx_to_onehot(in_seq, self.vocab_size)

            uav_item_list.append((in_seq, tgt_seq))

        return uav_item_list

    def collect_dataset(self, raw_data: pd.DataFrame) -> List[torch.Tensor] or None:
        """
        Returns a list of UAV design sequence tensors

        """

        uav_str_list = self.get_designs(raw_data)

        uav_item_list = []
        for uav_str in uav_str_list:
            uav_idxs = self.parse_design(uav_str)

            if uav_idxs is None:
                continue

            uav_item_list.append(uav_idxs)

        return uav_item_list


class UAVSequencePreprocessor(UAVDesignPreprocessor):
    """
    Version of the UAVDesignPreprocessor that instead of treating designs as individual datapoints,
    collects sequences of designs of a specified length.
    """
    def __init__(self, seq_len: int, one_hot=True):
        self.one_hot = one_hot
        self.seq_len = seq_len + 1  # +1 because we're making auto-regression input-target pairs, i.e need to offset
        super(UAVSequencePreprocessor, self).__init__(one_hot)

    def collect_dataset(self, raw_data: pd.DataFrame) -> List[torch.Tensor] or None:
        """
        Collects 'uav design sequences', i.e,  sequences of fixed length (self.seq_len) containing uav design
        characters. Sequences are thus simply UAV designs concatenated together.

        Hence, a uav design sequence can contain one or more partial designs since uav designs are different lengths.

        """
        uav_str_list = self.get_designs(raw_data)

        seq_item_list = []

        # Collect UAV designs into seq_len long tensors
        seq_item = None
        curr_seq_len = 0
        for uav_str in uav_str_list:
            uav_idx_seq = self.parse_design(uav_str)

            if uav_idx_seq is None:
                continue

            if seq_item is None:
                seq_item = torch.clone(uav_idx_seq)
            else:
                seq_item = torch.cat([seq_item, uav_idx_seq])

            curr_seq_len = len(seq_item)
            while curr_seq_len > self.seq_len:
                seq_item_list.append(torch.clone(seq_item[:self.seq_len]))
                seq_item = torch.clone(seq_item[self.seq_len:])
                curr_seq_len = len(seq_item)

        if curr_seq_len < self.seq_len:
            seq_item = torch.cat([seq_item, torch.full((self.seq_len - curr_seq_len,), fill_value=PAD_VALUE)])
            seq_item_list.append(torch.clone(seq_item))

        return seq_item_list


class UAVStringPreprocessor(UAVPreprocessor):
    """
    String preprocessor returns raw UAV design strings

    """
    def __init__(self):
        super(UAVStringPreprocessor, self).__init__()

    def collect_dataset(self, raw_dataset: pd.DataFrame) -> List[str]:
        return self.get_designs(raw_dataset)


class UAVIndexPreprocessor(UAVPreprocessor):
    """
    String preprocessor returns UAV design index tensors

    """
    def __init__(self, one_hot=True):
        self.one_hot = one_hot
        super(UAVIndexPreprocessor, self).__init__()

    def collect_dataset(self, raw_dataset: pd.DataFrame) -> List[torch.Tensor]:
        uav_str_list = self.get_designs(raw_dataset)

        uav_item_list = []
        for uav_str in uav_str_list:
            uav_idx_list = self.parse_design(uav_str)

            if uav_idx_list is None:
                continue

            if self.one_hot:
                uav_idx_list = idx_to_onehot(uav_idx_list, self.vocab_size)

            uav_item_list.append(uav_idx_list)

        return uav_item_list


class UAVDataPreprocessor(UAVPreprocessor):
    """
    UAVDataPreprocessor collects UAV design sequences (in one-hot or index format) + simulator data associated with it
    """
    def __init__(self, target_cols: List[str], one_hot: Any,
                 encoding_scheme="index"):
        """
        Args:
            data_cols: List of column names to extract from the dataset
            one_hot: Boolean indicating whether to convert designs to onehot format
        """
        super(UAVDataPreprocessor, self).__init__()
        self.one_hot = one_hot
        self.target_cols = target_cols

        self.encoding_scheme = encoding_scheme

        if self.one_hot and encoding_scheme != "index":
            print("Cannot use one-hot encoding for non-index encodings.")
            sys.exit(-1)

    def collect_dataset(self, raw_data: pd.DataFrame):
        uav_strings = self.get_designs(raw_data)

        # Process the simulator outputs
        sim_metrics = np.zeros((len(raw_data), len(self.target_cols))).astype(object)
        for col_idx, col_name in enumerate(self.target_cols):
            if col_name == SIM_RESULT_COL:
                sim_metrics[:, col_idx] = raw_data[col_name].values
            else:
                sim_metrics[:, col_idx] = raw_data[col_name].values.astype(float)

        # Process the UAV string
        uav_tensors = []
        for idx, uav_str in enumerate(uav_strings):
            uav_tensor = self.parse_design(uav_str)

            # Remove rows for which the UAV string couldn't be parsed
            if uav_str is None:
                sim_metrics = np.delete(sim_metrics, idx, axis=0)
                continue

            uav_tensors.append(uav_tensor)

        return uav_tensors, sim_metrics

    def parse_design(self, uav_str: str):
        if self.encoding_scheme == "matrix":
            return uav_str_to_mat(uav_str, encode_connections=False)
        elif self.encoding_scheme == "sdpe":
            return uav_string_to_vec(uav_str)
        elif self.encoding_scheme == "index":
            return super().parse_design(uav_str)
        elif self.encoding_scheme == "count":
            return uav_str_to_count_vec(uav_str)
        elif self.encoding_scheme is None:
            return super().parse_design(uav_str)
        else:
            print(f"Invalid encoding scheme provided: {self.encoding_scheme}")
            sys.exit(-1)

    def transform_dataset(self, uav_data):
        uav_tensors, sim_metrics = uav_data

        # Convert the simulator outcome column to result encodings (indices)
        try:
            sr_col_idx = self.target_cols.index(SIM_RESULT_COL)
            sim_outcomes = sim_metrics[:, sr_col_idx]
            sim_metrics = np.delete(sim_metrics, sr_col_idx, axis=1).astype(np.float)

            for outcome, code in SIM_OUTCOME_CODES.items():
                sim_outcomes[sim_outcomes == outcome] = code

            sim_outcomes = sim_outcomes.astype(int)

        except ValueError:
            print("Error processing outcomes! Could not convert to int.")
            sim_outcomes = None
            sr_col_idx = None
            sim_metrics = sim_metrics.astype(np.float)

        # Add the column of encoded results back to the array
        if sim_outcomes is not None:
            sim_metrics = np.insert(sim_metrics, sr_col_idx, sim_outcomes, axis=1)

        uav_item_list = []
        # Convert to list of tensor tuples
        for i in range(len(uav_tensors)):
            in_seq = uav_tensors[i]
            if self.one_hot:
                in_seq = idx_to_onehot(in_seq, self.vocab_size)

            tgt = torch.Tensor(sim_metrics[i, :]).type(torch.float32)

            uav_item_list.append((in_seq, tgt))

        return uav_item_list
    
    
class UAVGraphPreprocessor(UAVPreprocessor):
    """
    Converts raw UAV csv file data into torchdrug.Graph objects.
    
    Graph objects are characterized by an adjacency matrix, node feature matrix, edge feature matrix and graph features.
    
    In this case, nodes are components, edges are connections between components, node features are the component
    type & size, edge features are not defined and graph features are the UAV simulator metrics
    
    """
    
    def __init__(self, target_cols: List[str], node_encoding):
        self.node_encoding = node_encoding
        self.target_cols = target_cols
        
        if self.node_encoding != "vanilla":
            # TODO
            #
            # e.g "coord" encoding type could be here
            #
            assert False, "Not implemented"
        else:
            self.node_features = ["type", "size"]

        super(UAVGraphPreprocessor, self).__init__()
    
    def transform_dataset(self, dataset: pd.DataFrame) -> List[object]:
        """
        Transform raw data into the required format, in this case Graph -objects.
        
        """
        from data.UAVGraph import UAVGraph
        
        # Convert the simulator outcome strings into their integer encodings
        if SIM_OUTCOME in self.target_cols:
            uav_outcome = dataset[SIM_OUTCOME].values
            try:
                for outcome, code in SIM_OUTCOME_CODES.items():
                    uav_outcome[uav_outcome == outcome] = code
    
                uav_outcome = uav_outcome.astype(int)
                dataset[SIM_OUTCOME] = uav_outcome
    
            except ValueError:
                print("Error processing outcomes! Could not convert to int.")
                sys.exit(-1)
        
        parser = UAVGrammar()
        
        graph_dataset = []
        for row_idx in trange(len(dataset)):
            data_row = dataset.iloc[row_idx]
            uav_str = data_row[self.uav_design_col]
            targets = data_row[self.target_cols]
            
            components, connections, payload, _ = parser.parse(uav_str)
            
            node_feats, edges = uav2graph(components, connections, self.node_features)
            
            if len(node_feats) < 2:
                continue
            
            # Make a torchdrug graph object
            graph = UAVGraph(edge_list=edges, node_feature=node_feats,
                             num_node=len(components), num_relation=1)
            
            graph_dataset.append((graph, torch.Tensor(targets)))
            
        return graph_dataset



