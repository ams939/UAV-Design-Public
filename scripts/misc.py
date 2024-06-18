import json
import os
import sys

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc

from rl.DesignState import UAVDesign
from data.DataLoader import UAVSequenceLoader
from data.Constants import VOCAB_SIZE
from train.Loss import SCELoss, SNLLLoss
from train.Hyperparams import Hyperparams
from utils.utils import idx_to_onehot
import data.Constants as C


def get_sce_baseline_loss(seq_dataset):
    """
    Generates random numbers as a dummy NN output and calculates the loss using the SCELoss (cross entropy) function

    """
    dataloader = UAVSequenceLoader(seq_dataset, batch_size=64, shuffle=True)
    loss_fn = SCELoss()

    cum_loss = 0
    for inp_seqs, tgt_seqs in dataloader:
        rand_inp = torch.zeros(inp_seqs.shape).uniform_(0, 1)

        loss = loss_fn(rand_inp, tgt_seqs)
        cum_loss += loss.item()

    avg_loss = cum_loss / len(dataloader)

    return avg_loss


def get_sce_optimal_loss(seq_dataset):

    dataloader = UAVSequenceLoader(seq_dataset, batch_size=64, shuffle=True)

    # loss_fn = SCELoss()
    loss_fn = SNLLLoss()

    cum_loss = 0
    for inp_seqs, tgt_seqs in dataloader:
        onehot_tgt_seqs = idx_to_onehot(tgt_seqs, max_idx=VOCAB_SIZE)
        onehot_tgt_seqs = onehot_tgt_seqs.type(torch.float32)

        loss = loss_fn(onehot_tgt_seqs, tgt_seqs)
        cum_loss += loss.item()

    avg_loss = cum_loss / len(dataloader)

    return avg_loss


def plot_loss(experiment_folder):
    loss_log_file = f"{experiment_folder}/datafile_log.json"
    hparams = Hyperparams(f"{experiment_folder}/crnn_hparams.json")

    loss_name = hparams.loss_class.__name__

    with open(loss_log_file, 'r') as f:
        loss_data = json.load(f)

    loss_df = pd.DataFrame(loss_data)

    # Get some baselines for the graph
    seq_dataset = hparams.dataset_class(**hparams.dataset_hparams)

    base_loss = get_sce_baseline_loss(seq_dataset)
    optimal_loss = get_sce_optimal_loss(seq_dataset)

    plt.axhline(y=base_loss, color='r', linestyle='-', label='baseline')
    plt.axhline(y=optimal_loss, color='b', linestyle='-', label='optimal')

    plt.plot(loss_df['epoch'].values, loss_df[loss_name].values, color='#fc8803', label='train')
    plt.xlabel('Epoch')
    plt.ylabel(loss_name)
    plt.title(f'Experiment ID {hparams.experiment_id} model training loss plot')
    plt.legend()
    plt.savefig(f"{experiment_folder}/train_loss.png")


def parse_log(experiment_folder):
    """
    For (TOTALLY unlikely) case of screwed up loss logging and needing to parse the loss out of the text log
    (definitely didn't happen, and cause me to write this function)
    """

    with open(f"{experiment_folder}/log.txt") as f:
        log_lines = f.readlines()

    hparams = Hyperparams(f"{experiment_folder}/crnn_hparams.json")

    loss_name = hparams.loss_class.__name__

    loss_items = []
    for line in log_lines:
        if "Train Loss" not in line:
            continue

        line_items = line.split(" ")
        epoch = int(line_items[4])
        loss = float(line_items[9])

        loss_items.append({"epoch": epoch, loss_name: loss})

    with open(f"{experiment_folder}/datafile_log.json", "w") as f:
        json.dump(loss_items, f, indent=4)


def join_on_uav(hparams_path):
    hparams = Hyperparams(hparams_path)
    dataset_file = hparams.dataset_hparams["datafile"]
    experiment_folder = hparams.experiment_folder

    dataset = pd.read_csv(dataset_file)

    train_preds_file = f"{experiment_folder}/train_set_preds.csv"
    val_preds_file = f"{experiment_folder}/val_set_preds.csv"

    train_preds = pd.read_csv(train_preds_file)
    val_preds = pd.read_csv(val_preds_file)

    train_preds["set"] = np.full(len(train_preds), fill_value='train')
    val_preds["set"] = np.full(len(val_preds), fill_value='val')

    dataset_train = pd.merge(dataset, train_preds, on='config', how='inner', suffixes=('_true', '_pred'))
    dataset_val = pd.merge(dataset, val_preds, on='config', how='inner', suffixes=('_true', '_pred'))

    new_dataset = pd.concat([dataset_train, dataset_val])

    new_dataset.loc[new_dataset['result_true'] != 'Success', 'result_true'] = 'Failure'

    scores = get_scores(new_dataset)
    # print(scores)
    with open(f"{experiment_folder}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    new_dataset.to_csv(f"{experiment_folder}/dataset_comparison.csv", index=False)


def get_scores(df):
    scores = {}
    metric_cols = C.SIM_METRICS
    outcome_cols = [C.SIM_RESULT_COL]

    for set in ["train", "val"]:
        set_scores = {}
        set_df = df[df["set"] == set]
        for col in metric_cols + outcome_cols:
            try:
                true_vals = set_df[f"{col}_true"].values
                pred_vals = set_df[f"{col}_pred"].values
            except KeyError:
                print(f"No col {col} found.")
                continue

            if col in metric_cols:
                set_scores[f"{col}_mse"] = mse(true_vals, pred_vals)
            elif col in outcome_cols:
                set_scores[f"{col}_acc"] = acc(true_vals, pred_vals)

                classes, counts = np.unique(true_vals, return_counts=True)
                maj_class = classes[np.argmax(counts)]
                set_scores[f"{col}_acc_bline"] = acc(true_vals, np.full(len(true_vals), fill_value=maj_class))

        scores[set] = set_scores

    return scores


def viz_conn_dists(datafile: str):
    # dists == distance distributions
    from data.datamodel.Grammar import UAVGrammar
    from data.Constants import X_LETTER_COORDS, Z_LETTER_COORDS
    data = pd.read_csv(datafile)

    designs = data['config'].values
    vals = data.values

    parser = UAVGrammar()

    long_cons = []
    distances = []
    for idx, uav_str in enumerate(designs):
        components, connections, payload, _ = parser.parse(uav_str)

        # set up component and size matrices
        id_to_loc = {}

        # process components
        for c in components:
            comp_id = c[0]
            comp_x, comp_z = X_LETTER_COORDS.index(c[1]), Z_LETTER_COORDS.index(c[2])

            id_to_loc[comp_id] = np.asarray([comp_x, comp_z])

        processed = []

        for c in connections:
            comp1_id = c[0]
            comp2_id = c[1]
            c1c2 = f"{comp1_id}{comp2_id}"  # Same as c, but add for consistency/clarity
            c2c1 = f"{comp2_id}{comp1_id}"

            if c1c2 in processed or c2c1 in processed:
                continue
            else:
                processed.append(c1c2)
                processed.append(c2c1)

                comp1_loc = id_to_loc[comp1_id]
                comp2_loc = id_to_loc[comp2_id]

                # L1 distance computation
                dist = np.round((np.sum(np.abs(comp1_loc - comp2_loc))), 0).astype(int)
                distances.append(dist)

                if dist > 1.0:
                    long_cons.append(vals[idx])

    distances = np.asarray(distances)
    fname = os.path.basename(datafile)
    c, e, b = plt.hist(distances[distances > 1.0], bins=[x + 0.5 for x in range(1, 11)])
    plt.title(f"'{fname}' distance histogram for d > 1.0\n"
              f"(num distances <= 1.0 is {len(distances[distances <= 1.0])})")
    plt.xlabel("L1 distance")
    plt.ylabel("Frequency")
    plt.bar_label(b)
    plt.savefig(f"docs/metadata/{fname[:-4]}_conn_dists.png")
    plt.clf()

    pd.DataFrame(long_cons, columns=data.columns).to_csv("long_cons.csv", index=False)
    
    
def parse_dqn_nohup():
    from rl.DesignState import UAVDesign
    with open("C:/Users/Aleksanteri/Desktop/gl_nohup.out") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if "Current state" not in line:
            continue
        
        line = line.rstrip()
        item = {}
        line_list = line.split(" ")
        design = line_list[4][:-1]
        
        try:
            _ = UAVDesign(design)
        except Exception:
            continue
        
        metrics_str = line_list[6]
        stable_str = line_list[-1]
        
        metrics = metrics_str.split(",")[:3]
        
        try:
            item["config"] = design
            item["range"] = float(metrics[0])
            item["cost"] = float(metrics[1])
            item["velocity"] = float(metrics[2])
            item["result"] = "Success" if stable_str == "True" else "Failure"
        except Exception:
            continue
         
        data.append(item)
        
    df = pd.DataFrame(data)
    df.to_csv("dqn_designs.csv", index=False)


def parse_cprofiler_out(file):
    import re
    fdir = os.path.dirname(file)
    
    with open(file, 'r') as f:
        raw_data = f.readlines()
    
    clean_data = []
    for idx, line in enumerate(raw_data[1:]):
        line_items = re.split('\s+', line)
        if line_items[0] == '':
            line_items = line_items[1:]
        
        # If ncalls item in form total_invocations/recursive_calls, grab first element
        if '/' in line_items[0]:
            line_items[0] = line_items[0].split("/")[0]
        
        try:
            stat_row = {
                "ncalls": int(line_items[0]),
                "tottime": float(line_items[1]),
                "percall": float(line_items[2]),
                "cumtime": float(line_items[3]),
                "file_line_fun": "_".join(line_items[5:])
            }
        except (TypeError, ValueError, IndexError) as e:
            print(f"Error: '{e}' for line '{' '.join(line_items)}'")
            continue
            
        clean_data.append(stat_row)
        
    clean_data = pd.DataFrame(clean_data)
    
    clean_data.to_csv(f"{fdir}/cprofile.csv", index=False)
    
    
def cprofile_stats(file):
    data = pd.read_csv(file)
    
    np.sum(data["cumtime"].values)
    
    
def sample_dqn_log(fname, sample_size):
    from data.Constants import UAV_CONFIG_COL, SIM_RESULT_COL, SIM_METRICS
    
    data = pd.read_csv(fname)
    
    sample_size = min(sample_size, len(data))
    
    designs = data["config"].values
    
    _, indices = np.unique(designs, return_index=True)
    
    indices_sample = np.random.permutation(indices)[0:sample_size]
    
    sample = data.iloc[indices_sample]
    
    sample = sample[[UAV_CONFIG_COL] + SIM_METRICS + [SIM_RESULT_COL]]
    
    sample.to_csv(f"{os.path.dirname(fname)}/sample.csv", index=False)
    sample["config"].to_csv(f"{os.path.dirname(fname)}/sample_designs.csv", index=False)
    
    
def viz_reward():
    from rl.Reward import DeltaReward
    from rl.DesignState import UAVDesign
    from rl.rl_utils import inverse_normalize_metrics
    from train.Hyperparams import DummyHyperparams
    from train.Logging import ConsoleLogger
    from data.Constants import METRIC_NORM_FACTORS
    hparams = DummyHyperparams({
        "range_weight": 0.333,
        "cost_weight": 0.333,
        "velocity_weight": 0.333,
        "objective": "2a",
        "objective_type": "pre-defined",
        "logger": ConsoleLogger()
    })
    reward_class = DeltaReward(hparams)
    
    #uav_design = UAVDesign("*aMM0++*bNM2++*cMN1++*dLM2++*eML1++*fMO1++++++++++++++++*gOM4++++++++++++++++^ab^ac^ad^ae^cf^bg,20,3")
    #uav_design_next = UAVDesign(
    #    "*aMM0++*bNM2++*cMN1++*dLM2++*eML1++*fMO1++++++++++++++++*gOM4++++++++++++++++^ab^ac^ad^ae^cf^bg,20,3")
    #uav_design.set_metrics(velocity=0.780272364616394,cost=2561.919189453125,range=10.317264556884766, result='Success')
    #uav_design_next.set_metrics(velocity=0.780272364616394, cost=2561.919189453125, range=10.317264556884766,
    #                            result='Failure')
    
    #r = reward_class.get_reward(uav_design, uav_design_next)
    
    N = 1000
    ranges = np.linspace(0, 100, N)
    costs = np.linspace(0, 20000, N)
    velocities = np.linspace(0, 75, N)
    
    fixed_metrics = {}
    
    try:
        fixed_metrics["range"] = C.METRIC_NORM_FACTORS["range"]["max"]
    except KeyError:
        fixed_metrics["range"] = 0
        
    try:
        fixed_metrics["cost"] = C.METRIC_NORM_FACTORS["cost"]["min"]
    except KeyError:
        fixed_metrics["cost"] = 0
        
    try:
        fixed_metrics["velocity"] = C.METRIC_NORM_FACTORS["velocity"]["max"]
    except KeyError:
        fixed_metrics["velocity"] = 0

    # fixed_metrics = inverse_normalize_metrics(fixed_metrics)
    fixed_range = fixed_metrics["range"]
    fixed_cost = fixed_metrics["cost"]
    fixed_velocity = fixed_metrics["velocity"]
    
    obj_metrics = reward_class.obj_definitions_raw
    try:
        obj_range = obj_metrics["range"]["lower"]
    except KeyError:
        obj_range = 0
        
    try:
        obj_cost = obj_metrics["cost"]["upper"]
    except KeyError:
        obj_cost = 0
        
    try:
        obj_velocity = obj_metrics["velocity"]["lower"]
    except KeyError:
        obj_velocity = 0
    
    max_range = METRIC_NORM_FACTORS['range']['max']
    max_cost = METRIC_NORM_FACTORS['cost']['max']
    max_velocity = METRIC_NORM_FACTORS['velocity']['max']

    # Quality function plots
    range_qs = np.zeros((N, 3))
    cost_qs = np.zeros((N, 3))
    velocity_qs = np.zeros((N, 3))
    
    for i in range(N):
        range_qs[i] = reward_class.quality_fn({"range": ranges[i], "cost": fixed_cost, "velocity": fixed_velocity, "result": "Success"})
        cost_qs[i] = reward_class.quality_fn({"range": fixed_range, "cost": costs[i], "velocity": fixed_velocity, "result": "Success"})
        velocity_qs[i] = reward_class.quality_fn({"range": fixed_range, "cost": fixed_cost, "velocity": velocities[i], "result": "Success"})

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ranges, range_qs[:, 0])
    plt.plot([obj_range, obj_range], [0, np.amax(range_qs[:, 0])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_range, max_range], [0, np.amax(range_qs[:, 0])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Quality score (range component) vs. Range (mi)")
    
    plt.subplot(3, 1, 2)
    plt.plot(costs, cost_qs[:, 1])
    plt.plot([obj_cost, obj_cost], [0, np.amax(cost_qs[:, 1])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_cost, max_cost], [0, np.amax(cost_qs[:, 1])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Quality score (cost component) vs. Cost (USD)")
    
    plt.subplot(3, 1, 3)
    plt.plot(velocities, velocity_qs[:, 2])
    plt.plot([obj_velocity, obj_velocity], [0, np.amax(velocity_qs[:, 2])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_velocity, max_velocity], [0, np.amax(velocity_qs[:, 2])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Quality score (velocity component) vs. Velocity (mph)")
    
    plt.suptitle("Plots of metrics versus quality function value (one varied, rest fixed)\n"
                 f"(fixed metrics: range={fixed_range}, cost={fixed_cost}, velocity={fixed_velocity})")
    plt.tight_layout()
    plt.show()
    
    # Penalty function graphs
    range_penalty = np.zeros((N, 3))
    cost_penalty = np.zeros((N, 3))
    velocity_penalty = np.zeros((N, 3))
    
    for i in range(N):
        range_penalty[i] = reward_class.penalty_fn({"range": ranges[i], "cost": fixed_cost, "velocity": fixed_velocity})
        cost_penalty[i] = reward_class.penalty_fn({"range": fixed_range, "cost": costs[i], "velocity": fixed_velocity})
        velocity_penalty[i] = reward_class.penalty_fn({"range": fixed_range, "cost": fixed_cost, "velocity": velocities[i]})

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ranges, range_penalty[:, 0])
    plt.plot([obj_range, obj_range], [0, np.amax(range_penalty[:, 0])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_range, max_range], [0, np.amax(range_penalty[:, 0])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Penalty (range component) vs. Range (mi)")

    plt.subplot(3, 1, 2)
    plt.plot(costs, cost_penalty[:, 1])
    plt.plot([obj_cost, obj_cost], [0, np.amax(cost_penalty[:, 1])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_cost, max_cost], [0, np.amax(cost_penalty[:, 1])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Penalty (cost component) vs. Cost (USD)")

    plt.subplot(3, 1, 3)
    plt.plot(velocities, velocity_penalty[:, 2])
    plt.plot([obj_velocity, obj_velocity], [0, np.amax(velocity_penalty[:, 2])], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_velocity, max_velocity], [0, np.amax(velocity_penalty[:, 2])], 'r', linestyle='dashed', label='max')
    plt.legend()
    plt.title("Penalty (velocity component) vs. Velocity (mph)")

    plt.suptitle("Plots of metrics versus penalty function value (one varied, rest fixed)\n"
                 f"(fixed metrics: range={fixed_range}, cost={fixed_cost}, velocity={fixed_velocity})")
    plt.tight_layout()
    plt.show()
    
    # Finally, the reward function
    range_reward = np.sum(range_qs * range_penalty, axis=1)
    cost_reward = np.sum(cost_qs * cost_penalty, axis=1)
    velocity_reward = np.sum(velocity_qs * velocity_penalty, axis=1)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ranges, range_reward)
    plt.plot([obj_range, obj_range], [0, np.amax(range_reward)], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_range, max_range], [0, np.amax(range_reward)], 'r', linestyle='dashed', label='max')
    plt.plot([0, np.amax(ranges)], [2/3, 2/3], 'b', linestyle='dashed', label='velocity + cost reward')
    plt.legend()
    plt.title("Reward vs. Range (mi)")

    plt.subplot(3, 1, 2)
    plt.plot(costs, cost_reward)
    plt.plot([obj_cost, obj_cost], [np.amin(cost_reward), np.amax(cost_reward)], 'g', linestyle='dashed', label=f'objective {hparams.objective}')
    plt.plot([max_cost, max_cost], [np.amin(cost_reward), np.amax(cost_reward)], 'r', linestyle='dashed', label='max')
    plt.plot([0, np.amax(costs)], [2 / 3, 2 / 3], 'b', linestyle='dashed', label='range + velocity reward')
    plt.legend()
    plt.title("Reward vs. Cost (USD)")

    plt.subplot(3, 1, 3)
    plt.plot(velocities, velocity_reward)
    plt.plot([obj_velocity, obj_velocity], [0, np.amax(velocity_reward)], 'g', linestyle='dashed',
             label=f'objective {hparams.objective}')
    plt.plot([max_velocity, max_velocity], [0, np.amax(velocity_reward)], 'r', linestyle='dashed', label='max')
    plt.plot([0, np.amax(velocities)], [2 / 3, 2 / 3], 'b', linestyle='dashed', label='range + cost reward')
    plt.legend()
    plt.title("Reward vs. Velocity (mph)")

    plt.suptitle("Plots of metrics versus reward function value (one varied, rest fixed)\n"
                 f"(fixed metrics: range={fixed_range}, cost={fixed_cost}, velocity={fixed_velocity})")

    plt.tight_layout()
    plt.show()
    
    # Finally finally, the reward function with all metrics linearly increasing
    reward_all = np.zeros(N)
    labels = []
    for i in range(N):
        metrics = {
            'range': ranges[i],
            'cost': costs[N - i - 1],
            'velocity': velocities[i],
            'result': 'Success'
        }
        reward_all[i] = np.sum(reward_class.quality_fn(metrics) * reward_class.penalty_fn(metrics))
        labels.append(reward_class.objective_complete(metrics))
    
    plt.title("Reward function with metrics linearly 'improving'")
    plt.plot(np.arange(N), reward_all)
    plt.ylabel("Reward fn value")
    plt.xlabel("Improvement step")
    plt.show()


def check_objective(file):
    from tqdm import trange
    from data.Constants import SIM_RESULT_COL, SIM_SUCCESS, OBJECTIVES, SIM_METRICS
    from rl.Reward import DeltaReward
    from train.Hyperparams import DummyHyperparams
    from train.Logging import ConsoleLogger
    import json
    # data = pd.read_json(file)
    data = json.load(open(file, "r"))
    
    
    # data = data.loc[data[SIM_RESULT_COL] == SIM_SUCCESS]
    # data = data.reset_index()
    
    hparams = DummyHyperparams({
        "objective": "2a",
        "objective_type": "pre-defined",
        "range_weight": 0.333,
        "cost_weight": 0.333,
        "velocity_weight": 0.333,
        "logger": ConsoleLogger()
    })
    obj_metrics = dict()

    print(f"Checking objective {hparams.objective}")

    reward_fn = DeltaReward(hparams)
    
    succ = []
    n_obj_complete = 0
    for row in trange(len(data)):
        
        it = data[row]
        
        metrics = {}
        try:
            for m in SIM_METRICS + [SIM_RESULT_COL]:
                metrics[m] = it[m]
        except Exception:
            print(f"Skiprow {it}")
            continue
        
        if reward_fn.objective_complete(metrics):
            n_obj_complete += 1
            succ.append(it)
    
    #obj_metrics[hparams.objective] = n_obj_complete
        
    with open("succ.json", "w") as f:
        json.dump(succ, f)
    
    print(n_obj_complete)


def main():
    import sys
    from rl.DesignAction import collect_size_actions
    import pandas as pd
    
    check_objective("experiments/DQN/031423095549/datafile_log.json")
    
    #from model.SKLearnModel import train_k_fold

    # check_objective("data/datafiles/preprocessed/original_dataset_preprocessed.csv")

    #hparams = Hyperparams("hparams/rf_hparams.json")
    #hparams.experiment_id = "5"
    #hparams.experiment_folder = hparams.experiment_folder + "/" + hparams.experiment_id
    #train_k_fold(hparams)
    
    #design = "*aMM0*bMN1^ab,20,1"
    #uav = UAVDesign(design)
    # print("\n".join([u.to_string() for u in collect_size_actions(uav)]))
    
    #print(len(uav.get_successors()))
    #print([u.to_string() for u in uav.get_successors()])
    # parse_cprofiler_out("experiments/DQN/092822131144_cProfiler.txt")
    
    # plog = False
    # ploss = False

    # experiment_id = "082922155425"
    # experiment_folder = f"./experiments/DQN/{experiment_id}"
    # file = f"{experiment_folder}/datafile_log.csv"
    # hparams_path = f"{experiment_folder}/simrnn_hparams.json"
    
    # viz_reward()
    
    #sample_dqn_log(file, 10000)

    #if plog:
    #    parse_log(experiment_folder)
    #if ploss:
    #    plot_loss(experiment_folder)

    # join_on_uav(hparams_path)

    # hparams = Hyperparams(hparams_path)
    # dataset_file = hparams.dataset_hparams["datafile"]
    # experiment_folder = hparams.experiment_folder

    # dataset = pd.read_csv(dataset_file)

    # labels, counts = np.unique(dataset["config"].values, return_counts=True)

    # print(counts[counts > 1], labels[counts > 1])

    # d1 = pd.read_csv("data/datafiles/generated/062722152144_sample1.csv")
    # d2 = pd.read_csv("data/datafiles/generated/062722152144_sample2.csv")
    # d3 = pd.read_csv("data/datafiles/generated/062722152144_sample3.csv")

    # dgen = pd.concat([d1, d2, d3])
    # print(f'data_len={len(dgen)} n_unique={len(np.unique(dgen["config"].values))}')
    # dgen.to_csv("data/datafiles/generated/generated_uav_designs.csv", index=False)

    files = [
        'data/datafiles/preprocessed/aggregated_uav_designs.csv',
        'data/datafiles/preprocessed/filtered_aggregated_uav_designs.csv',
        'data/datafiles/preprocessed/balanced_filtered_aggregated_uav_designs.csv',
        'data/datafiles/preprocessed/simresults_preprocessed_validunique.csv',
        'data/datafiles/generated/generated_uav_designs.csv'
    ]

    #for file in files:
    #    viz_col_dists(file)
    # viz_conn_dists(files[-2])


if __name__ == "__main__":
    from torchdrug import datasets, models, tasks, core
    import pickle
    from torch import nn, optim
    
    #dataset = datasets.ZINC250k("./", kekulize=True,
    #                            atom_feature="symbol")
    #with open("./zinc250k.pkl", "wb") as fout:
    #    pickle.dump(dataset, fout)

    with open("./zinc250k.pkl", "rb") as fin:
        dataset = pickle.load(fin)
        
    # Load into torchdrug task
    # COmpare with UAVGraph dataset

    # hparams = Hyperparams("./hparams/gcn_hparams.json")
    # dataset = hparams.dataset_class(hparams)

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

    
    #for a in dataset:
    #    print(a)
    # import data
    # df = pd.read_csv("data/datafiles/preprocessed/design_database.csv")
    #
    # for idx, uav_str in enumerate(df["config"].values):
    #     p = data.datamodel.Grammar.UAVGrammar()
    #     b, errors = p.validate(uav_str)
    #
    #     if b is False:
    #         record = df.iloc[idx]
    #         if record["result"] != "Success":
    #             print(record.values)
    
    # main()
    
    

