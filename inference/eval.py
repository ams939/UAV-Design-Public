from tqdm import tqdm

import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from data.Constants import UAV_STR_COL, SIM_OUT_COLS, UAV_CONFIG_COL, SIM_METRICS, SIM_RESULT_COL, SIM_SUCCESS
from inference.UAVSimulator import HyFormSimulator

# experiments\DQN\052123144523


def eval_surrogate(exp_id: str):
    # Get the surrogate predictions and designs
    raw_data = json.load(open(f"experiments/DQN/{exp_id}/datafile_log.json", "r"))
    
    json_metrics = []
    for d in raw_data:
        
        if "config" not in d.keys():
            continue
        new_d = dict()
        for k in d.keys():
            if k in ["config", "range", "cost", "velocity", "result"]:
                new_d[k] = d[k]
        json_metrics.append(new_d)
    
    print(f"Extracted {len(json_metrics)} predictions from {len(raw_data)} log entries.")
    
    # Take a random sample of the predictions
    sample_size = int(0.1 * len(json_metrics))
    
    target_cols = ["range", "cost", "velocity", "result"]
    sample = np.random.choice(np.asarray(json_metrics), sample_size).tolist()
    sample_df = pd.DataFrame(sample)
    sample_ndarray = np.zeros((len(sample), len(target_cols)), dtype=object)
    
    for i, t in enumerate(target_cols):
        sample_ndarray[:, i] = sample_df[t].values
            
    print(f"Sampled {len(sample)} predictions.")
    print(f"    n_stable: {np.sum(sample_ndarray[:, -1] == 'Success')}")
    
    uav_strings = list(map(lambda x: x["config"], sample))
    
    sim = HyFormSimulator()
    sim_results = sim.simulate_batch(uav_strings)
    
    sim_results_json = []
    for result in tqdm(sim_results, total=len(sample)):
        sim_results_json.append(result)
    
    sim_df = pd.DataFrame(sim_results_json)
    sim_ndarray = np.zeros((len(sample), len(target_cols)), dtype=object)

    for i, t in enumerate(target_cols):
        sim_ndarray[:, i] = sim_df[t].values
    
    scores = dict()
    scores["all"] = eval_metrics(sim_ndarray, sample_ndarray, target_cols)
    scores["stable"] = eval_stable_metrics(sim_ndarray, sample_ndarray, target_cols)
    
    print(scores)


def eval_predictions(hparams, train_preds, val_preds=None) -> dict:
    dataset_file = hparams.dataset_hparams["datafile"]

    dataset = pd.read_csv(dataset_file)

    train_preds["set"] = np.full(len(train_preds), fill_value='train')

    dataset_train = pd.merge(dataset, train_preds, on='config', how='inner', suffixes=('_true', '_pred'))

    if val_preds is not None:
        val_preds["set"] = np.full(len(val_preds), fill_value='val')
        dataset_val = pd.merge(dataset, val_preds, on='config', how='inner', suffixes=('_true', '_pred'))
        new_dataset = pd.concat([dataset_train, dataset_val])
    else:
        new_dataset = dataset_train

    if 'result_true' in new_dataset.columns:
        new_dataset.loc[new_dataset['result_true'] != 'Success', 'result_true'] = 'Failure'

    scores = get_sim_scores(new_dataset)

    return scores


def rmse(x, y): return np.sqrt(mean_squared_error(x, y))


def eval_metrics(y, y_pred, target_cols):
    
    results = dict()
    results["n"] = len(y)
    for idx, tgt_name in enumerate(target_cols):
        results[f"{tgt_name}"] = {}

        if tgt_name == SIM_RESULT_COL:
            metric_fns = [accuracy_score]
        else:
            metric_fns = [mean_squared_error, mean_absolute_error]  #  , rmse]

        for metric_fn in metric_fns:
            results[f"{tgt_name}"][f"{metric_fn.__name__}"] = float(metric_fn(y[:, idx], y_pred[:, idx]))
            
    return results


def eval_stable_metrics(y, y_pred, target_cols):
    assert SIM_RESULT_COL in target_cols, "Must provide result column"
    result_col_idx = target_cols.index(SIM_RESULT_COL)
    
    stable_idxs = y[:, result_col_idx] == SIM_SUCCESS
    y_stable, y_pred_stable = y[stable_idxs], y_pred[stable_idxs]
    scores = eval_metrics(y_stable, y_pred_stable, target_cols)
    scores["n_stable"] = int(np.sum(stable_idxs))
    
    return scores
    

def get_sim_scores(df):
    scores = {}
    metric_cols = SIM_METRICS
    outcome_cols = [SIM_RESULT_COL]

    sets = np.unique(df["set"].values).tolist()

    for set_name in sets:
        set_scores = {}
        set_df = df[df["set"] == set_name]
        for col in metric_cols + outcome_cols:
            col_scores = {}
            try:
                true_vals = set_df[f"{col}_true"].values
                pred_vals = set_df[f"{col}_pred"].values
            except KeyError:
                # print(f"No col {col} found.")
                continue

            if col in metric_cols:
                mse = mean_squared_error(true_vals, pred_vals)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(true_vals, pred_vals)

                col_scores["mse"] = mse
                col_scores["rmse"] = rmse
                col_scores["mae"] = mae
            elif col in outcome_cols:
                col_scores["acc"] = accuracy_score(true_vals, pred_vals)

                classes, counts = np.unique(true_vals, return_counts=True)
                maj_class = classes[np.argmax(counts)]
                set_scores["acc_bline"] = accuracy_score(true_vals, np.full(len(true_vals), fill_value=maj_class))

            set_scores[f"{col}"] = col_scores

        scores[set_name] = set_scores

    return scores


class UAVMetricTracker():
    """ Tracks the simulator performance on a given set of UAV designs, comparing them to the ground truth """
    

def main():
    from train.Hyperparams import DummyHyperparams
    from data.UAVDataset import UAVStringDataset

    exp_id = "052123144523"
    
    eval_surrogate(exp_id)
    
    # eval_design_novelty(f"./experiments/{exp_id}/valid_designs.csv",
    #                     "./data/datafiles/preprocessed/preprocessed_validunique.csv", f"./experiments/{exp_id}/")

    # data_dir = "./data/datafiles/preprocessed"
    # data_file = "preprocessed_validunique.csv"
    #data_dir = f"./experiments/CharRNN/{exp_id}"
    #data_file = "novel_valid_sample_2.csv"
    #data_path = f"{data_dir}/{data_file}"

    #model_hparams = DummyHyperparams({"dataset_hparams": {"datafile": data_path}})
    #uav_strings = UAVStringDataset(model_hparams).data
    #sim = HyFormSimulator()

    #sim_results = sim.simulate_batch(uav_strings)

    #result_file = open(f"{data_dir}/simresults_{data_file}", "w")
    #result_file.write(",".join(SIM_OUT_COLS) + "\n")
    #for sim_result in tqdm(sim_results, total=len(uav_strings)):

    #    # Add double quotes to config string as it contains commas
    #    sim_result[UAV_CONFIG_COL] = f'"{sim_result[UAV_CONFIG_COL]}"'

    #    result_file.write(",".join([str(sim_result[RESULT_COL]) for RESULT_COL in SIM_OUT_COLS]) + "\n")

    #result_file.close()


if __name__ == "__main__":
    main()



