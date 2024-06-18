"""
Class implementing UAVSimulator using an NN model

"""

from typing import List
import os

import torch

from train.Hyperparams import Hyperparams
from inference.UAVSimulator import UAVSimulator
from data.Constants import SIM_METRICS, SIM_OUTCOME, UAV_CONFIG_COL


class NNSimulator(UAVSimulator):
    """
    Class wrapping the NN surrogate for the UAV simulator. Facilitates getting simulation results in the same fashion
    as from the HyForm simulator

    """

    def __init__(self, nn_hparams: Hyperparams, debug=False):
        self.hparams = nn_hparams
        
        # Make sure a trained model exists!
        model_file = f"{self.hparams.experiment_folder}/{self.hparams.model_file}"
        assert os.path.exists(model_file), f"Error, couldn't load {self.hparams.experiment_folder}/{self.hparams.model_file}" \
                                           "\nMake sure to specify the correct path and file in " \
                                           "hparams.experiment_folder and hparams.model_file, s.t the model is found " \
                                           "in hparams.experiment_folder/hparams.model_file"
        
        super(NNSimulator, self).__init__(debug)

    def _init_simulator(self):
        self._simulator = self.hparams.model_class(self.hparams)
        self._simulator = self._simulator.to(self.hparams.device)

    def simulate(self, uav_str: str, return_df=True):
        if return_df:
            return self.simulate_batch([uav_str], return_df)
        else:
            return self.simulate_batch([uav_str], return_df)[0]

    def simulate_batch(self, uav_str_list: List[str], return_df=True):
        """
        Simulates a batch of UAVs. Returns as dataframe or json, depending on "return_df"
        """
        
        # Convert the UAV strings into a format suitable for the NN via the UAVDataset object
        dataset = self.hparams.dataset_class(self.hparams, load=False, verbose=False)
        uav_tensors = [(dataset.preprocessor.parse_design(uav_str),
                        torch.zeros(len(dataset.tgt_cols))) for uav_str in uav_str_list]
        uav_tensors = dataset.batch_function(uav_tensors)[0]

        # Initialize the network output postprocessor
        post_processor = self.hparams.postprocessor_class(self.hparams)

        # Feed batches to NN and get predictions, append the results dataframe
        self._simulator.eval()
        sim_metrics, outcomes = self._simulator.predict(uav_tensors)

        if len(uav_str_list) == 1:
            sim_metrics = sim_metrics.unsqueeze(dim=0)
            outcomes = outcomes.unsqueeze(dim=0)

        # Post process the results from the net
        sim_results = post_processor.postprocess(uav_tensors, sim_metrics, outcomes)
        
        if return_df:
            if sim_results["config"].values[0] == "":
                sim_results["config"] = uav_str_list
            return sim_results
        
        # Reformat from dataframe into json
        results = []
        for row_idx in range(len(sim_results)):
            result_dic = dict()
            result_dic[UAV_CONFIG_COL] = sim_results[UAV_CONFIG_COL]
            for metric in SIM_METRICS:
            
                result_dic[metric] = sim_results[f"{metric}_pred"].values[row_idx]

            result_dic[SIM_OUTCOME] = sim_results[f"{SIM_OUTCOME}_pred"].values[row_idx]

            results.append(result_dic)
        return results


if __name__ == "__main__":
    from train.Logging import init_logger
    default_uav = "*aMM0+++++*bNM2+++*cMN1+++*dLM2+++*eML1+++^ab^ac^ad,5,3"
    uav = "*aMM0++*bNM2++*cMN1++*dLM2++*eML1++^ab^ac^ad^ae,20,3"
    exp_id = "072522221739"
    hparams = Hyperparams(f"trained_models/sim_nn/nn_hparams.json")
    hparams.logger = init_logger(hparams)

    if torch.cuda.is_available() and hparams.device == "gpu":
        hparams.device = torch.device('cuda:0')
    else:
        hparams.device = torch.device('cpu')

    sim = NNSimulator(hparams)
    results = sim.simulate(uav, False)
    print(results)
