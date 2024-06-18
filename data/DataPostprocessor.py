from typing import List, Any
from abc import ABC, abstractmethod
from copy import copy

import torch
import pandas as pd
import numpy as np

from data.Constants import TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE, NN_TOKENS, EOS_TOKEN, UAV_STR_COL, \
    SIM_OUTCOME_KEYS, UAV_CONFIG_COL, SIM_RESULT_COL
from train.Hyperparams import Hyperparams
from utils.utils import ddict, onehot_to_idx


class UAVPostprocessor(ABC):
    def __init__(self):
        self.ch_to_idx = copy(TOKEN_TO_IDX)
        self.idx_to_ch = copy(IDX_TO_TOKEN)
        self.vocab_size = VOCAB_SIZE

    @abstractmethod
    def postprocess(self, *args) -> Any:
        pass

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

    def parse_design(self, uav_idxs: torch.Tensor) -> str or None:

        try:
            uav_ch_list = [self.get_idx_ch(int(i)) for i in uav_idxs]
        except ValueError as e:
            print(e)
            return None

        # Turn the list of characters into a string
        uav_str = "".join(uav_ch_list)

        return uav_str


class UAVSequencePostprocessor(UAVPostprocessor):
    def __init__(self, allow_partial=False):
        self.allow_partial = allow_partial
        super(UAVSequencePostprocessor, self).__init__()

    def postprocess(self, uav_idxs: List[torch.Tensor]) -> pd.DataFrame:

        design_strs = []
        for idx_tensor in uav_idxs:
            seq_str = self.parse_design(idx_tensor)

            if self.allow_partial:
                design_strs.append(seq_str)

            seq_designs = seq_str.split(EOS_TOKEN)

            while len(seq_designs) > 1:
                seq_design = seq_designs.pop(0)

                # Remove the NN tokens
                for token in NN_TOKENS:
                    seq_design = seq_design.replace(token, "")

                design_strs.append(seq_design)

        design_df = pd.DataFrame({f"{UAV_STR_COL}": design_strs})

        return design_df


class UAVDataPostprocessor(UAVPostprocessor):
    """
    Processes outputs from a NN trained on the UAVDataDataset into the same format as the original dataset

    """
    def __init__(self, hparams: Hyperparams):
        self.dataset_hparams = ddict(hparams.dataset_hparams)
        self.scale = self.dataset_hparams.scale
        self.target_cols = self.dataset_hparams.target_cols
        self.one_hot = self.dataset_hparams.one_hot

        if self.scale:
            self.scaler = self.dataset_hparams.scaler_class()
            self.scaler_params = self.dataset_hparams.scaler_hparams
            for k, v in self.scaler_params.items():
                if isinstance(v, list):
                    v = np.asarray(v)
                    
                self.scaler.__setattr__(k, v)

        super(UAVDataPostprocessor, self).__init__()

    def postprocess(self, uav_idxs: torch.Tensor, pred_metrics: torch.Tensor = None,
                    pred_outcomes: torch.Tensor = None, target_metrics=None, target_outcomes=None) -> pd.DataFrame:
       
        # TODO: Solve this weird issue. Result of GPU, module versions, what??
        # Hack: For some reason when running on the work computer the dataloader returns a list.
        if isinstance(uav_idxs, list):
            uav_idxs = uav_idxs[0]

        # Hack: For when using with dataloader that returns indices and sequence lengths
        if isinstance(uav_idxs, tuple):
            uav_idxs = uav_idxs[0]
            
        batch_size = pred_metrics.shape[0]
        
        if len(uav_idxs) != batch_size:
            uav_idxs = torch.transpose(uav_idxs, dim0=1, dim1=0)
            
        try:
            assert len(uav_idxs) == batch_size
        except AssertionError:
            raise Exception(f"Couldn't infer batch dimension of uav representation data {uav_idxs.shape}")

        uav_data = []
        for idx in range(batch_size):
            uav_tensor = uav_idxs[idx]

            if self.one_hot:
                uav_tensor = onehot_to_idx(uav_tensor)

            uav_str = self.parse_design(uav_tensor)

            # Remove the NN tokens
            for token in NN_TOKENS:
                uav_str = uav_str.replace(token, "")

            # Use scaler to rescale outputs into original scale
            pred_metric = None
            if pred_metrics is not None:
                if self.scale:
                    pred_metric = self.scaler.inverse_transform(pred_metrics[idx].numpy().reshape(1, -1)).flatten()

                else:
                    pred_metric = pred_metrics[idx].numpy().flatten()

            tgt_metric = None
            if target_metrics is not None:
                if self.scale:
                    tgt_metric = self.scaler.inverse_transform(target_metrics[idx].numpy().reshape(1, -1)).flatten()
                else:
                    tgt_metric = target_metrics[idx].numpy().flatten()

            # Decode the outcome
            pred_outcome = None
            if pred_outcomes is not None:
                pred_outcome = SIM_OUTCOME_KEYS[int(pred_outcomes[idx])]

            tgt_outcome = None
            if target_outcomes is not None:
                tgt_outcome = SIM_OUTCOME_KEYS[int(target_outcomes[idx])]

            # Record results into the results dictionary
            uav_result = {UAV_CONFIG_COL: uav_str}
            for metric in self.target_cols:
                if metric == SIM_RESULT_COL:
                    uav_result[f"{metric}_pred"] = pred_outcome
                    if tgt_outcome is not None:
                        uav_result[f"{metric}_tgt"] = tgt_outcome
                else:
                    uav_result[f"{metric}_pred"] = pred_metric[self.target_cols.index(metric)]

                    if tgt_metric is not None:
                        uav_result[f"{metric}_tgt"] = tgt_metric[self.target_cols.index(metric)]

            uav_data.append(uav_result)

        return pd.DataFrame(uav_data)


class UAVMatrixPostprocessor(UAVDataPostprocessor):
    """
    Processes outputs from a NN trained on the UAVMatrixDataset into the same format as the original dataset

    """
    def __init__(self, hparams):
        super(UAVMatrixPostprocessor, self).__init__(hparams)

    def parse_design(self, uav_tensor) -> str:
        # TODO
        return ""
    
    
class UAVGraphPostprocessor(UAVDataPostprocessor):
    def __init__(self, hparams):
        super(UAVGraphPostprocessor, self).__init__(hparams)
        
    def postprocess(self, uav_graphs: object, pred_metrics: torch.Tensor = None,
                    pred_outcomes: torch.Tensor = None, target_metrics=None, target_outcomes=None):
        # Unpack the packed graph that the dataloader sends us
        uav_graphs = [uav_graphs.unpack()]
        return super().postprocess(uav_graphs, pred_metrics, pred_outcomes, target_metrics, target_outcomes)
        
    def parse_design(self, uav_tensor) -> str:
        # TODO
        return "ParseNotImplemented"
    

if __name__ == "__main__":
    from train.Hyperparams import Hyperparams
    hparams = Hyperparams("experiments/SimRNN/070622140745/simrnn_hparams.json")
    dpp = UAVDataPostprocessor(hparams)
