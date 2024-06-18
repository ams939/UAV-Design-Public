"""
UAVDataset class - stores information concerning the dataset

"""
from abc import abstractmethod
import os
import sys
from typing import List, Tuple
from copy import deepcopy, copy

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import RobustScaler
import joblib

from train.Hyperparams import Hyperparams
from data.Constants import PAD_VALUE, SIM_RESULT_COL, SIM_OUTCOME_VALS, COMPONENT_TYPE_IDS
import data.DataPreprocessor as dp


class UAVDataset(Dataset):
    def __init__(self, hparams: Hyperparams = None, load=True):
        super(UAVDataset, self).__init__()
        self.hparams = hparams
        self.data = None
        self.preprocessor = None
        self.load = load

        try:
            data_filepath = self.hparams.dataset_hparams["datafile"]
            self.datafile = os.path.basename(data_filepath)
            self.datapath = os.path.dirname(data_filepath)
        except AttributeError:
            self.datafile = None
            self.datapath = None

        self._initialize()

    @property
    def __name__(self):
        return self.__class__.__name__

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _initialize(self) -> None:
        self._init_preprocessor()

        if self.datafile is not None and self.load:
            self._load_data()
            self._preprocess_data()

    def set_data(self, raw_data: object):
        self.data = raw_data

    def get_data(self):
        return self.data

    def _init_preprocessor(self) -> None:
        self.preprocessor = None

    def _load_data(self) -> None:
        with open(f"{self.datapath}/{self.datafile}", 'r') as f:
            self.data = pd.read_csv(f)

    def _preprocess_data(self) -> None:
        if self.preprocessor is not None:
            self.data = self.preprocessor.preprocess(self.data)

    @staticmethod
    @abstractmethod
    def batch_function(tensor_list: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Returns 'well-formed' batches from the dataset, for use with DataLoader
        """
        pass
    
    def serialize(self, fname: str):
        print(f"Saving self to {fname}")
        
        # A stupid workaround for problem with serializing hyperparams
        tmp_hparams = self.hparams
        self.hparams = tmp_hparams.hparams_file
        torch.save(self, fname)
        self.hparams = tmp_hparams
        
    def deserialize(self, fname: str):
        
        print(f"Initializing from {fname}")
        
        try:
            dataset = torch.load(fname)
            dataset.hparams = self.hparams
            return dataset
            
        except Exception as e:
            print(f"Error, could not initialize from {fname}")
            print(e)
            return self


class UAVDesignDataset(UAVDataset):
    def __init__(self, hparams: Hyperparams, load=True):
        try:
            self.one_hot = hparams.dataset_hparams["one_hot"]
        except KeyError:
            self.one_hot = False

        super(UAVDesignDataset, self).__init__(hparams, load)

    def _init_preprocessor(self):
        self.preprocessor = dp.UAVDesignPreprocessor(self.one_hot)

    @staticmethod
    def batch_function(tensor_list: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Merges a list of samples to form a mini-batch.

        Args:
          list_of_samples is a list of tuples (src_seq, tgt_seq):
              src_seq is of shape (src_seq_length,)
              tgt_seq is of shape (tgt_seq_length,)

        Returns:
          src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
              The sequences should be sorted by length in a decreasing order, that is src_seqs[:,0] should be
              the longest sequence, and src_seqs[:,-1] should be the shortest.
          src_seq_lengths: List of lengths of source sequences.
          tgt_seqs of shape (max_tgt_seq_length, batch_size): Tensor of padded target sequences.
        """

        # Get the sequence lengths
        src_tgt_lens = torch.Tensor([[len(x), len(y)] for x, y in tensor_list])
        srt_src_lens, srt_src_idx = torch.sort(src_tgt_lens[:, 0], descending=True)

        # Unpack the tuple and apply padding
        pad_src_seqs = pad_sequence([x for x, _ in tensor_list], padding_value=PAD_VALUE, batch_first=False)
        pad_tgt_seqs = pad_sequence([y for _, y in tensor_list], padding_value=PAD_VALUE, batch_first=False)

        # Sort in decreasing order
        srt_pad_src_seqs = pad_src_seqs[:, srt_src_idx]
        srt_pad_tgt_seqs = pad_tgt_seqs[:, srt_src_idx]

        return srt_pad_src_seqs, srt_src_lens.int().tolist(), srt_pad_tgt_seqs


class UAVStringDataset(UAVDataset):
    def __init__(self, hparams: Hyperparams, load=True):
        super(UAVStringDataset, self).__init__(hparams, load)

    def _init_preprocessor(self):
        self.preprocessor = dp.UAVStringPreprocessor()

    @staticmethod
    def batch_function(uav_strings: List[str]) -> List[str]:
        return uav_strings


class UAVIndexDataset(UAVDataset):
    def __init__(self, hparams: Hyperparams = None, load=True):
        super(UAVIndexDataset, self).__init__(hparams, load)

    def _init_preprocessor(self):
        self.preprocessor = dp.UAVIndexPreprocessor()

    @staticmethod
    def batch_function(tensor_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merges a list of samples to form a mini-batch.

                Args:
                  list_of_samples is a list of (src_design) of length batch_size:
                    src_seq is of shape (src_seq_length, encoding_size)

                Returns:
                  A Tuple with the batch of input tensors of shape (max_seq_len, batch_size, encoding size)
                  and the true lengths of the inputs of shape (batch_size)
        """

        # Get the sequence lengths
        seq_lens = torch.Tensor([len(x) for x in tensor_list]).type(torch.long)
        srt_seq_lens, srt_seq_idx = torch.sort(seq_lens, descending=True)

        # Apply padding
        pad_src_seqs = pad_sequence([x for x in tensor_list], padding_value=PAD_VALUE, batch_first=False)

        # Sort batch sequences within batch in decreasing order (required by rnn packing util)
        srt_pad_src_seqs = pad_src_seqs[:, srt_seq_idx]
        inp_batch = (srt_pad_src_seqs, srt_seq_lens)

        return inp_batch


class UAVSequenceDataset(UAVDataset):
    def __init__(self, hparams: Hyperparams, load=True):
        self.seq_len = hparams.dataset_hparams["seq_len"]
        self.one_hot = hparams.dataset_hparams["one_hot"]
        super(UAVSequenceDataset, self).__init__(hparams, load)

    @staticmethod
    def batch_function(seq_tensor_list: List[Tuple[torch.Tensor, torch.Tensor]]):

        # Unpack tuples
        inp_seqs = torch.stack([x for x, _ in seq_tensor_list])
        tgt_seqs = torch.stack([y for _, y in seq_tensor_list])

        # Swap the batch dimension
        inp_seqs = torch.transpose(inp_seqs, dim0=1, dim1=0)
        tgt_seqs = torch.transpose(tgt_seqs, dim0=1, dim1=0)

        return inp_seqs, tgt_seqs

    def _init_preprocessor(self):
        self.preprocessor = dp.UAVSequencePreprocessor(self.seq_len, self.one_hot)


class UAVRegressionDataset(UAVDesignDataset):
    def __init__(self, hparams: Hyperparams, load=True):
        self.hparams = hparams
        self.tgt_cols = hparams.dataset_hparams["target_cols"]
        self.is_scaled = False

        # Assumption that needs to be true due to the way things are processed
        if SIM_RESULT_COL in self.tgt_cols:
            assert self.tgt_cols[-1] == SIM_RESULT_COL, "Set the last element of the target columns hparam as the sim " \
                                                        f"outcome column '{SIM_RESULT_COL}'."

        try:
            self.scale = hparams.dataset_hparams["scale"]

            if self.scale:
                try:
                    scaler_class = hparams.dataset_hparams["scaler_class"]
                    self.scaler = scaler_class()
                except KeyError:
                    self.scaler = RobustScaler()
                    self.hparams.logger.log({"name": self.__name__, "msg": "No scaler specified, using RobustScaler"})
        except KeyError:
            self.scale = False

        # hparams.logger.log({"name": self.__name__, "msg": f"Scale data: {self.scale}"})

        super(UAVRegressionDataset, self).__init__(hparams, load)

    def _init_preprocessor(self):
        try:
            self.encoding_scheme = self.hparams.dataset_hparams["encoding_scheme"]
        except KeyError:
            self.encoding_scheme = "index"
    
        if self.verbose:
            self.hparams.logger.log({"name": self.__name__, "msg": f"Using encoding scheme '{self.encoding_scheme}'"})
        
        self.preprocessor = dp.UAVDataPreprocessor(self.tgt_cols, self.one_hot, self.encoding_scheme)

    @staticmethod
    def batch_function(tensor_list: List[Tuple[torch.Tensor, torch.Tensor]]) \
            -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """Merges a list of samples to form a mini-batch.

        Args:
          list_of_samples is a list of tuples (src_seq, tgt) of length batch_size:
              src_seq is of shape (src_seq_length,)
              tgt is of shape (n_target_features)

        Returns:
          A tuple:
            1st elem is a tuple with:
                src_seqs: Tensor of padded source sequences of shape (max_src_seq_length, batch_size).
                      The sequences should be sorted by length in a decreasing order, that is src_seqs[:,0] should be
                      the longest sequence, and src_seqs[:,-1] should be the shortest.
                src_seq_lengths: List of lengths of un-padded source sequences.
            2nd elem is targets

        """

        # Get the sequence lengths
        seq_lens = torch.Tensor([len(x) for x, _ in tensor_list]).type(torch.long)
        srt_seq_lens, srt_seq_idx = torch.sort(seq_lens, descending=True)

        # Unpack the tuple and apply padding
        pad_src_seqs = pad_sequence([x for x, _ in tensor_list], padding_value=PAD_VALUE, batch_first=False)
        tgt_batch = torch.stack([y for _, y in tensor_list])

        # Move the batch dimension
        tgt_batch = torch.transpose(tgt_batch, dim0=1, dim1=0)

        # Sort batch sequences within batch in decreasing order (required by rnn packing util)
        srt_pad_src_seqs = pad_src_seqs[:, srt_seq_idx]
        tgt_batch = tgt_batch[:, srt_seq_idx]

        inp_batch = (srt_pad_src_seqs, srt_seq_lens)

        return inp_batch, tgt_batch

    def split_dataset(self, split_proportion: float = None, split_indices: Tuple[np.ndarray, np.ndarray] = None) -> \
            Tuple[UAVDataset, UAVDataset]:
        from sklearn.model_selection import train_test_split
        """
        Method for getting a train-test split of the data
        
        Provide EITHER  split_proportion OR split indices
        Args:
            split_proportion: Proportion of data to put into the test set. Rest goes to train set
            split_indices: A tuple of indices (train_indices, test_indices)

        Returns:
            Tuple with two new UAVRegressionDataset objects, train_dataset and test_dataset

        """

        # Get record indices for both sets
        if split_indices is None:
            assert split_proportion is not None, "Split proportion not specified."
            data_indices = np.arange(len(self.data)).astype(int)
            split_indices = train_test_split(data_indices, test_size=split_proportion, shuffle=True)

        set1_indices, set2_indices = split_indices

        try:
            index_union = set(set1_indices.tolist()).union(set2_indices.tolist())
            assert len(index_union) == len(self.data), "WARNING: Not all data included in dataset split indices."
            assert len(set(set1_indices.tolist()).intersection(set2_indices.tolist())) == 0, \
                "WARNING: Dataset split indices are NOT mutually exclusive!!!"

        except AssertionError as e:
            print(e)

        # Split into two sets
        data1 = list()
        data2 = list()
        for idx in set1_indices:
            data1.append(copy(self.data[idx]))
        
        for idx in set2_indices:
            data2.append(copy(self.data[idx]))

        # Workaround for issue with deepcopy and class names
        temp_hparams = self.hparams
        self.hparams = None
        temp_data = self.data
        self.data = []

        dataset1 = deepcopy(self)
        dataset2 = deepcopy(self)

        self.data = temp_data

        self.hparams = temp_hparams
        dataset1.hparams = temp_hparams
        dataset2.hparams = temp_hparams

        dataset1.data = data1
        dataset2.data = data2

        return dataset1, dataset2

    def scale_dataset(self, inverse=False, fit=True):
        if fit:
            assert self.is_scaled is not True, "Trying to fit scaler to already scaled dataset"
        if inverse:
            assert fit is False, "Can't fit and inverse scale at same time."
            assert self.is_scaled is True, "Dataset needs to be scaled before calling inverse"

        # Unpack the tensor tuples
        tgt_tensors = torch.stack([y for _, y in self.data]).numpy()

        # Pull out columns that shouldn't be scaled (class labels)
        try:
            sr_col_idx = self.tgt_cols.index(SIM_RESULT_COL)
            sim_outcomes = tgt_tensors[:, sr_col_idx]
            tgt_tensors = np.delete(tgt_tensors, sr_col_idx, axis=1).astype(np.float)
        except ValueError:
            sim_outcomes = None
            sr_col_idx = None

        # Check scaling mode: Either scaling new data or scaling scaled data back to original values
        if inverse:
            scaled_tgt_tensors = self.scaler.inverse_transform(tgt_tensors)
            self.is_scaled = False
        else:
            if fit:
                scaled_tgt_tensors = self.scaler.fit_transform(tgt_tensors)
            else:
                scaled_tgt_tensors = self.scaler.transform(tgt_tensors)

            self.is_scaled = True
            self.hparams.dataset_hparams["scaler_hparams"] = self.get_scaler_params()

        # Add the columns taken out earlier back to the results
        if sim_outcomes is not None:
            scaled_tgt_tensors = np.insert(scaled_tgt_tensors, sr_col_idx, sim_outcomes, axis=1)

        # Convert back to tensor tuples
        tensor_list = [(self.data[i][0], torch.Tensor(scaled_tgt_tensors[i]).type(torch.float32))
                       for i in range(len(self.data))]

        self.data = tensor_list

    def get_scaler_params(self):
        if self.is_scaled is False:
            return None
        
        scaler_params = {
                "center_": list(self.scaler.center_.astype(float)),
                "scale_": list(self.scaler.scale_.astype(float)),
                "n_features_in_": self.scaler.n_features_in_
            }

        return scaler_params

    def set_scaler_params(self, scaler_params):
        for k, v in scaler_params.items():
            if isinstance(v, list):
                v = np.asarray(v)
            
            self.scaler.__setattr__(k, v)
            
    def get_binary_class_counts(self):
        # Unpack the tensor tuples
        tgt_tensors = torch.stack([y for _, y in self.data]).numpy()
    
        try:
            sr_col_idx = self.tgt_cols.index(SIM_RESULT_COL)
            sim_outcomes = tgt_tensors[:, sr_col_idx]
        except ValueError:
            return 0, 0
        
        n_pos = np.sum(sim_outcomes == SIM_OUTCOME_VALS["Success"])
        
        return len(self.data) - n_pos, n_pos


class UAVMatrixDataset(UAVRegressionDataset):
    def __init__(self, hparams: Hyperparams, load=True, verbose=True):
        self.verbose = verbose
        super(UAVMatrixDataset, self).__init__(hparams, load)

    def _init_preprocessor(self):
        try:
            self.encoding_scheme = self.hparams.dataset_hparams["encoding_scheme"]
        except KeyError:
            self.encoding_scheme = "matrix"
        
        if self.verbose:
            self.hparams.logger.log({"name": self.__name__, "msg": f"Using encoding scheme '{self.encoding_scheme}'"})
        super(UAVMatrixDataset, self)._init_preprocessor()

    @staticmethod
    def batch_function(tensor_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs

        Args:
            tensor_list: A batch_len sized list with input, target tensor tuples.


        Returns a tuple of inputs and targets

        """
        inp = torch.stack([x for x, _ in tensor_list])
        tgt = torch.stack([y for _, y in tensor_list])

        # Move the batch dimension
        inp = torch.transpose(inp, dim0=1, dim1=0).unsqueeze(-1).type(torch.float32)
        tgt = torch.transpose(tgt, dim0=1, dim1=0).type(torch.float32)

        return inp, tgt
    
    
class UAVGraphDataset(UAVRegressionDataset):
    """
    Class for transforming UAV grammar strings into a set of torchdrug.data.Graph objects
    
    Graph objects are characterized by an adjacency matrix, node feature matrix, edge feature matrix and graph features.
    
    In this case, nodes are components, edges are connections between components, node features are the component
    type & size, edge features are not defined and graph features are the UAV simulator metrics
    
    """
    def __init__(self, hparams: Hyperparams, load=True, verbose=True):
        self.verbose = verbose
        self.node_encoding = hparams.node_encoding
        self.node_feature_dim = None
        self.cache_path = hparams.dataset_hparams.cache_path
        self.is_cached = False
        
        self.atom_types = [int(c_id) for c_id in COMPONENT_TYPE_IDS]
        self.transform = None

        super(UAVGraphDataset, self).__init__(hparams, load)
        
    def _initialize(self) -> None:
        if self.cache_path is not None and os.path.exists(self.cache_path):
            self.is_cached = True
    
        if self.node_encoding is None:
            self.node_encoding = "vanilla"
        else:
            self.node_encoding = "vanilla"
    
        if self.node_encoding == "vanilla":
            self.node_feature_dim = 2
            
        self._init_preprocessor()

        if self.load:
            if self.is_cached:
                self.hparams.logger.log({"name": self.__name__,
                                         "msg": f"Loading data from cache file: {self.cache_path}."})
                try:
                    cache = joblib.load(self.cache_path)
                    data = cache["data"]
                    
                    self.set_data(data)
                    self.hparams.logger.log({"name": self.__name__,
                                             "msg": "Loading from cache successful."})
                except Exception as e:
                    self.hparams.logger.log({"name": self.__name__, "msg": f"Exception occurred while loading cache:\n{e}"})
                    sys.exit(-1)
            elif self.datafile is not None:
                self._load_data()
                self.hparams.logger.log({"name": self.__name__,
                                         "msg": f"Loaded raw data from {self.hparams.dataset_hparams.datafile}."})
                self.hparams.logger.log({"name": self.__name__,
                                         "msg": f"Pre-processing..."})
                self._preprocess_data()
                cache = {"data": self.data}
                
                joblib.dump(cache, f"{os.path.dirname(self.hparams.dataset_hparams.datafile)}/{self.__name__}.jb")
            else:
                self.hparams.logger.log({"name": self.__name__,
                                         "msg": "Error: No datafile or cached data path provded."})
                raise ValueError
            
    def batch_function(self, data_list: List[Tuple[object, torch.Tensor]]):
        from torchdrug.data import Graph
        inp = Graph.pack([x for x, _ in data_list])
        tgt = torch.stack([y for _, y in data_list])
    
        # Move the batch dimension
        tgt = torch.transpose(tgt, dim0=1, dim1=0).type(torch.float32)
    
        return inp, tgt
        
    def _init_preprocessor(self):
        try:
            self.encoding_scheme = self.hparams.dataset_hparams["encoding_scheme"]
        except KeyError:
            self.encoding_scheme = "graph"
    
        if self.verbose:
            self.hparams.logger.log({"name": self.__name__, "msg": f"Using encoding scheme '{self.encoding_scheme}'"})
        
        self.preprocessor = dp.UAVGraphPreprocessor(self.tgt_cols, self.node_encoding)
            
    def scale_dataset(self, inverse=False, fit=True):
        if fit:
            assert self.is_scaled is not True, "Trying to fit scaler to already scaled dataset"
        if inverse:
            assert fit is False, "Can't fit and inverse scale at same time."
            assert self.is_scaled is True, "Dataset needs to be scaled before calling inverse"
            
        # Retrieve metrics from graph
        tgt_tensors = torch.vstack([target for (_, target) in self.data]).numpy()

        # Pull out columns that shouldn't be scaled (class labels)
        try:
            sr_col_idx = self.tgt_cols.index(SIM_RESULT_COL)
            sim_outcomes = tgt_tensors[:, sr_col_idx]
            tgt_tensors = np.delete(tgt_tensors, sr_col_idx, axis=1).astype(np.float)
        except ValueError:
            sim_outcomes = None
            sr_col_idx = None

        # Check scaling mode: Either scaling new data or scaling scaled data back to original values
        if inverse:
            scaled_tgt_tensors = self.scaler.inverse_transform(tgt_tensors)
            self.is_scaled = False
        else:
            if fit:
                scaled_tgt_tensors = self.scaler.fit_transform(tgt_tensors)
            else:
                scaled_tgt_tensors = self.scaler.transform(tgt_tensors)

            self.is_scaled = True
            self.hparams.dataset_hparams["scaler_hparams"] = self.get_scaler_params()

        # Add the columns taken out earlier back to the results
        if sim_outcomes is not None:
            scaled_tgt_tensors = np.insert(scaled_tgt_tensors, sr_col_idx, sim_outcomes, axis=1)

        for g_idx in range(len(self.data)):
            self.data[g_idx] = (self.data[g_idx][0], torch.Tensor(scaled_tgt_tensors[g_idx], device=self.hparams.device))

    def get_item(self, index):
        
        # item = {"graph": self.data[index][0]}
        #
        # # TODO: implement this method
        # # item.update({k: v[index] for k, v in self.targets.items()})
        # if self.transform:
        #     item = self.transform(item)
            
        return self.data[index]

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]
    
    def cuda(self):
        self.data = [(graph.cuda(), target.to(torch.device("cuda"))) for (graph, target) in self.data]
        
    def cpu(self):
        self.data = [(graph.cpu(), target.to(torch.device("cpu"))) for (graph, target) in self.data]
            

if __name__ == "__main__":
    from train.Hyperparams import DummyHyperparams
    hparams = DummyHyperparams({""})

