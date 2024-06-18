"""
main.py
Author: Aleksanteri Sladek
6.2022


Main script for performing operations with the UAV-Design codebase

Note that this is a script, and hence doesnt follow ideal coding practices. F.e there's some code duplication :(
It's more of an 'worktable' of operations

Supported operations:
- train - training a model
- generate - generating data from a trained model

"""

import argparse
import os
import json

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import wandb

from train.Hyperparams import Hyperparams
from train.Trainer import Trainer
from train.Logging import init_logger
from data.DataLoader import UAVDataLoader
from data.UAVDataset import UAVStringDataset
from utils.utils import generate_experiment_id, result_csv
from inference.eval import eval_metrics, eval_stable_metrics
from rl.Trainer import train_dqn
from data.Constants import EXPERIMENT_MODES

# os.environ["WANDB_MODE"] = "disabled"


def run_experiment(hparams_filepath: str, experiment_type=None, args=None):
    # Retrieve the experiment hyperparameters from file specified
    hparams_filename = os.path.basename(hparams_filepath)
    hparams = Hyperparams(hparams_filepath)

    if torch.cuda.is_available() and hparams.device == "gpu":
        hparams.device = torch.device('cuda:0')
    else:
        hparams.device = torch.device('cpu')

    # Get type of experiment
    if experiment_type is None:
        experiment_type = hparams.experiment_type
    else:
        hparams.experiment_type = experiment_type
        
    if experiment_type not in EXPERIMENT_MODES:
        raise AttributeError(f"Invalid experiment type '{experiment_type}' given.")

    # Identify the experiment
    if "experiment_id" not in hparams.keys() or len(hparams.experiment_id) == 0:
        if experiment_type in ["train", "simulate", "trainpredict", "traineval", "kfold", "run_dqn"]:
            experiment_id = generate_experiment_id()
            results_folder = f"{hparams.experiment_folder}/{experiment_id}"
        else:
            print("ERROR: Only training experiments can be conducted without an existing"
                  f"experiment ID. Please specify an experiment ID in {hparams_filepath}")
            return
    else:
        experiment_id = hparams.experiment_id
        results_folder = hparams.experiment_folder

    hparams.experiment_id = experiment_id
    hparams.experiment_folder = results_folder

    if not os.path.exists(results_folder):
        parent_dir = os.path.dirname(results_folder)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        print(f"Creating folder for results at {results_folder}")
        os.mkdir(results_folder)

    # Initialize logger(s)
    logger = init_logger(hparams)
    hparams.logger = logger

    hparams.logger.log({"name": "main", "msg": f"Beginning experiment ID {experiment_id} of type '{experiment_type}'"})
    hparams.logger.log({"name": "main", "msg": f"Using hyperparams: {hparams.to_string()}"})

    # Run requested experiment type
    if experiment_type == "train":
        train(hparams)

    elif experiment_type == "run_dqn":
        run_dqn(hparams)

    elif experiment_type == "predict":
        predict(hparams, fname="eval")

    elif experiment_type == "trainpredict":
        model, train_dataloader, val_dataloader, test_dataloader = train(hparams)

        predict(hparams, model, train_dataloader, fname="train")

        if val_dataloader is not None:
            predict(hparams, model, val_dataloader, fname="val")

        if test_dataloader is not None:
            predict(hparams, model, test_dataloader, fname="test")

    elif experiment_type == "traineval":
        model, train_dataloader, val_dataloader, test_dataloader = train(hparams)
        evaluate(hparams, model, train_dataloader, val_dataloader, test_dataloader)

    elif experiment_type == "eval" or experiment_type == 'evaluate':
        model = hparams.model_class(hparams)
        evaluate(hparams, model)

    elif experiment_type == "generate":
        generate(hparams)

    elif experiment_type == "simulate":

        if args.input_file[0] != "":
            hparams.dataset_hparams.datafile = args.input_file[0]

        if args.output_file[0] != "":
            hparams.simulator_hparams.simulator_results_file = args.output_file[0]

        simulate(hparams)

    elif experiment_type == "kfold":
        k_fold_experiment(hparams)

    else:
        print(f"Unknown experiment type '{experiment_type}'")

    # Save the hparams
    new_hparams_filepath = f"{results_folder}/{hparams_filename}"
    hparams.save(new_hparams_filepath)

    logger.log({"name": "main", "msg": f"Experiment {experiment_id} completed"})
    logger.close()


def simulate(hparams):

    # Perform predictions on CPU
    # hparams.device = torch.device("cpu")

    # Initialize the simulator
    simulator = hparams.simulator_class(hparams)
    hparams.logger.log({"name": "main", "msg": f"Simulating with {simulator.__name__}"})

    # Get simulation output file
    sample_file = hparams.simulator_hparams["simulator_results_file"]

    hparams.logger.log({"name": "main", "msg": f"Loading UAV strings from: {hparams.dataset_hparams.datafile}"})
    uav_string_dataset = UAVStringDataset(hparams)

    dataloader = UAVDataLoader(uav_string_dataset, batch_size=512, shuffle=False)

    hparams.logger.log({"name": "main", "msg": f"Simulating..."})
    sim_results = None
    for uav_strings in tqdm(dataloader):
        batch_results = simulator.simulate_batch(uav_strings)

        if sim_results is None:
            sim_results = batch_results
        else:
            sim_results = pd.concat([sim_results, batch_results])

    sim_results.to_csv(sample_file, index=False)

    hparams.logger.log(
        {"name": "main", "msg": f"Completed {len(sim_results)} simulations. Results saved to {sample_file}."})


def generate(hparams):
    exp_id = hparams.experiment_id

    # Perform evaluations on CPU
    hparams.device = torch.device("cpu")

    # Initialize model and set to eval mode
    model = hparams.model_class(hparams)
    model.eval()

    # Get sampling params
    sample_seq_len = hparams.sampling_hparams["seq_len"]
    batch_size = hparams.sampling_hparams["batch_size"]
    sample_size = hparams.sampling_hparams["sample_size"]
    sample_file = hparams.sampling_hparams["sample_file"]

    hparams.logger.log({"name": "main", "msg": f"Generating {sample_size} samples from experiment ID {exp_id}"})

    sample_file = f"{hparams.experiment_folder}/{sample_file}"

    sample_list = []
    while len(sample_list) < sample_size:

        # Get a sample of data from the model
        samples = model.sample(seq_len=sample_seq_len, n_samples=batch_size)

        for i in range(samples.shape[1]):
            sample_list.append(samples[:, i])

    # Convert sampled indices to strings
    postprocessor = hparams.postprocessor_class(**hparams.postprocessor_hparams)
    sample_strings = postprocessor.postprocess(sample_list)

    sample_strings.to_csv(sample_file, index=False)

    hparams.logger.log({"name": "main", "msg": f"Sampled {sample_size} sequences, "
                                               f"yielding {len(sample_strings)} designs. "
                                               f"Designs saved to {sample_file}."})


def run_dqn(hparams):
    train_dqn(hparams)


def train(hparams):

    hparams.logger.log({"name": "main", "msg": f"Beginning training experiment on device: {hparams.device.type}"})
    model = hparams.model_class(hparams)

    print(model)

    dataset = hparams.dataset_class(hparams)
    hparams.logger.log({"name": "main",
                        "msg": f"Loaded {len(dataset)} records"})

    # Decide if we're using a validation set
    try:
        val_prop = hparams.val_proportion
        assert val_prop is not None
    except (AssertionError, AttributeError):
        val_prop = 0.0

    try:
        test_prop = hparams.test_proportion
        assert test_prop is not None
    except (AssertionError, AttributeError):
        test_prop = 0.0

    val_dataset = None
    test_dataset = None
    if 0.0 < val_prop + test_prop < 1.0:
        if test_prop > 0.0:
            dataset, test_dataset = dataset.split_dataset(split_proportion=test_prop)
            hparams.logger.log({"name": "main",
                                "msg": f"Splitting the dataset into a train and test set with "
                                f"{test_prop*100:.0f}% ({len(test_dataset)} records) as the test set"})
            if hparams.save_testset is True:
                test_dataset.serialize(f"{hparams.experiment_folder}/test_set.pth")

        if val_prop > 0.0:
            dataset, val_dataset = dataset.split_dataset(split_proportion=val_prop)
            hparams.logger.log({"name": "main",
                                "msg": f"Splitting train dataset into a train and validation set with "
                                f"{val_prop * 100:.0f}% ({len(val_dataset)} records) as the validation set, "
                                f"{len(dataset)} records in the train set."})

    try:
        scale = hparams.dataset_hparams["scale"]
    except KeyError:
        scale = False

    # Scale the dataset targets
    if scale:
        dataset.scale_dataset(fit=True)
        scaler_hparams = dataset.get_scaler_params()

        if val_dataset is not None:
            val_dataset.set_scaler_params(scaler_hparams)
            val_dataset.scale_dataset(fit=False)

        if test_dataset is not None:
            test_dataset.set_scaler_params(scaler_hparams)
            test_dataset.scale_dataset(fit=False)

    # Initialize dataloaders for each dataset
    train_dataloader = UAVDataLoader(dataset=dataset, **hparams.dataloader_hparams)
    val_dataloader = UAVDataLoader(dataset=val_dataset, **hparams.dataloader_hparams) \
        if val_dataset is not None else None
    
    test_dataloader = UAVDataLoader(dataset=test_dataset, **hparams.dataloader_hparams) \
        if test_dataset is not None else None

    # If we're using a class weighted loss function, get the class distribution in dataset
    if "WeightedMAEBCELoss" in hparams.loss_class.__name__:
        neg, pos = dataset.get_binary_class_counts()
        pos_weight = neg / pos

        try:
            hparams.loss_hparams["pos_weight"] = torch.Tensor([pos_weight]).item()
        except TypeError:
            hparams["loss_hparams"] = {}
            hparams.loss_hparams["pos_weight"] = torch.Tensor([pos_weight]).item()
    
    trainer = Trainer(hparams)
    model = trainer.train_model(model, train_dataloader, val_dataloader)
    model.save()

    return model, train_dataloader, val_dataloader, test_dataloader


def evaluate(hparams, model=None, dataloader=None, val_dataloader=None, test_dataloader=None):
    """
    Evaluate performance of a model

    """
    hparams.logger.log({"name": "main", "msg": f"Beginning evaluation of model..."})
    # hparams.device = torch.device('cpu')

    target_cols = hparams.dataset_hparams["target_cols"]

    tgt_cols = [f"{col}_tgt" for col in target_cols]
    pred_cols = [f"{col}_pred" for col in target_cols]

    scores = {}
    stable_scores = {}

    if dataloader is not None:
        hparams.logger.log({"name": "main", "msg": f"Getting predictions for train set..."})
        train_preds = predict(hparams, model=model, dataloader=dataloader, to_file=False)
        y_train = train_preds[tgt_cols].values
        y_train_pred = train_preds[pred_cols].values
        scores["train"] = eval_metrics(y_train, y_train_pred, target_cols)
        stable_scores["train"] = eval_stable_metrics(y_train, y_train_pred, target_cols)

    if val_dataloader is not None:
        hparams.logger.log({"name": "main", "msg": f"Getting predictions for validation set..."})
        val_preds = predict(hparams, model=model, dataloader=val_dataloader, to_file=False)
        y_val = val_preds[tgt_cols].values
        y_val_pred = val_preds[pred_cols].values
        scores["val"] = eval_metrics(y_val, y_val_pred, target_cols)
        stable_scores["val"] = eval_stable_metrics(y_val, y_val_pred, target_cols)

    if test_dataloader is not None:
        hparams.logger.log({"name": "main", "msg": f"Getting predictions for test set..."})
        test_preds = predict(hparams, model=model, dataloader=test_dataloader, to_file=False)
        y_test = test_preds[tgt_cols].values
        y_test_pred = test_preds[pred_cols].values
        scores["test"] = eval_metrics(y_test, y_test_pred, target_cols)
        stable_scores["test"] = eval_stable_metrics(y_test, y_test_pred, target_cols)

    if dataloader is None and val_dataloader is None and test_dataloader is None:
        preds = predict(hparams, model=model, to_file=False)
        y_test = preds[tgt_cols].values
        y_test_pred = preds[pred_cols].values

        scores["all"] = eval_metrics(y_test, y_test_pred, target_cols)
        stable_scores["all"] = eval_stable_metrics(y_test, y_test_pred, target_cols)
        
    scores["stable"] = stable_scores

    with open(f"{hparams.experiment_folder}/scores.json", "a") as f:
        json.dump(scores, f, indent=4)

    hparams.logger.log({"name": "main", "msg": json.dumps(scores, indent=4)})

    return scores


def predict(hparams, model=None, dataloader=None, to_file=False, fname=None):
    # hparams.device = torch.device('cpu')

    if model is None:
        model = hparams.model_class(hparams)
    else:
        model = model.to(hparams.device)

    if dataloader is None:
        dataset = hparams.dataset_class(hparams)
        dataset.set_scaler_params(hparams.dataset_hparams["scaler_hparams"])
        dataset.scale_dataset(fit=False)
        # Initialize dataloaders for each dataset
        dataloader = UAVDataLoader(dataset=dataset, **hparams.dataloader_hparams)

    # Initialize the postprocessor
    try:
        post_processor = hparams.postprocessor_class(hparams)
    except AttributeError as e:
        hparams.logger.log(name="main", msg=f"ERROR: Postprocessor init error: {e}")
        raise AttributeError

    model.eval()
    target_cols = hparams.dataset_hparams["target_cols"]
    results = pd.DataFrame({})
    for inp, tgt in dataloader:
        tgt = torch.transpose(tgt, dim0=1, dim1=0)

        reg_pred, clf_pred = model.predict(inp)

        # A godawful hack, fix soon tm
        if len(target_cols) == 1:
            if 'result' in target_cols:
                result = post_processor.postprocess(inp, pred_metrics=None, pred_outcomes=clf_pred,
                                                    target_outcomes=tgt, target_metrics=None)
            else:
                result = post_processor.postprocess(inp, pred_metrics=reg_pred, target_metrics=tgt)
        elif 'result' not in target_cols:
            result = post_processor.postprocess(inp, pred_metrics=reg_pred, target_metrics=tgt)
        else:
            result = post_processor.postprocess(inp, pred_metrics=reg_pred, target_metrics=tgt[:, :-1],
                                                pred_outcomes=clf_pred, target_outcomes=tgt[:, -1])
        results = pd.concat([results, result])

    if to_file:
        results_file = f"{hparams.experiment_folder}/{fname}_set_preds.csv"
        results.to_csv(results_file, index=False)
        hparams.logger.log({"name": "main",
                            "msg": f"{len(results)} predictions for {fname} saved to {results_file}..."})

    return results


def k_fold_experiment(hparams):
    n_folds = hparams.n_folds

    if n_folds is None:
        n_folds = 5

    if torch.cuda.is_available() and hparams.device == "gpu":
        hparams.device = torch.device('cuda:0')
    else:
        hparams.device = torch.device('cpu')

    hparams.logger.log({"name": "main", "msg": f"Beginning {n_folds}-folds training experiment on device: "
                                               f"{hparams.device.type}"})

    dataset = hparams.dataset_class(hparams)
    hparams.logger.log({"name": "main",
                        "msg": f"Raw data loaded from {dataset.datafile} ({len(dataset)} records) "})

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    data_indices = np.arange(len(dataset))

    results = dict()
    for fold_idx, set_indices in enumerate(kf.split(data_indices)):
        hparams.logger.log(
            {"name": "main", "msg": f"\n{80*'='}\nFold {fold_idx} experiment beginning.\n{80*'='}\n"})

        # Initialize model
        model = hparams.model_class(hparams)

        print(model)

        # Split into train set and test set
        train_dataset, test_dataset = dataset.split_dataset(split_indices=set_indices)

        hparams.logger.log({"name": "main",
                            "msg": f"Splitting the dataset into a train and test set with "
                                   f"{len(train_dataset)}/{len(test_dataset)} records in the train/test set"})

        # Split the train set into a new train set and validation set
        val_prop = hparams.val_proportion
        if val_prop > 0.0:
            train_dataset, val_dataset = train_dataset.split_dataset(split_proportion=val_prop)
            hparams.logger.log({"name": "main",
                                "msg": f"Splitting the train dataset into a train and validation set with "
                                       f"{len(train_dataset)}/{len(val_dataset)} records in the train/validation set"})
        else:
            val_dataset = None

        # Scale the dataset if specified
        if hparams.dataset_hparams["scale"]:
            # Fitting scaler only on training data to avoid leakage to validation and test set.
            train_dataset.scale_dataset(fit=True)
            scaler_params = train_dataset.get_scaler_params()

            test_dataset.set_scaler_params(scaler_params)
            test_dataset.scale_dataset(fit=False)

            if val_dataset is not None:
                val_dataset.set_scaler_params(scaler_params)
                val_dataset.scale_dataset(fit=False)

        # Initialize dataloaders for each dataset
        train_dataloader = UAVDataLoader(dataset=train_dataset, **hparams.dataloader_hparams)
        val_dataloader = None
        test_dataloader = None
        if val_dataset is not None:
            val_dataloader = UAVDataLoader(dataset=val_dataset, **hparams.dataloader_hparams)

        if test_dataset is not None:
            test_dataloader = UAVDataLoader(dataset=test_dataset, **hparams.dataloader_hparams)

        trainer = Trainer(hparams)
        model = trainer.train_model(model, train_dataloader, val_dataloader)
        model.save(f"Fold-{fold_idx}_")

        fold_results = evaluate(hparams, model, train_dataloader, val_dataloader, test_dataloader)

        for set_name, result_set in fold_results.items():
            if set_name not in results.keys():
                results[set_name] = {}
            for feature, f_results in result_set.items():
                if feature not in results[set_name].keys():
                    results[set_name][feature] = {}

                for metric, m_result in f_results.items():
                    if metric not in results[set_name][feature].keys():
                        results[set_name][feature][metric] = 0

                    results[set_name][feature][metric] += result_set[feature][metric]

                    if fold_idx == n_folds - 1:
                        results[set_name][feature][metric] /= n_folds

    hparams.logger.log({"name": "main", "msg": f"\n{80*'='}\n{n_folds}-Fold Average Results\n{80*'='}\n"})
    hparams.logger.log({"name": "main", "msg": f"{json.dumps(results, indent=4)}"})

    header, values = result_csv(results)
    hparams.logger.log({"name": "main", "msg": f"\n{header}\n{values}"})
    
    
def train_gcn(hparams):
    from model.GCN import SimGCN
    from train.Trainer import GCNTrainer
    hparams.logger.log({"name": "main", "msg": f"Beginning training experiment on device: {hparams.device.type}"})
    
    dataset = hparams.dataset_class(hparams)

    model_hparams = hparams.model_hparams
    model = SimGCN(model_hparams)
    print(model)
    
    hparams.logger.log({"name": "main",
                        "msg": f"Raw data loaded from {dataset.datafile} ({len(dataset)} records) "})
    
    trainer = GCNTrainer(hparams)
    trainer.train_model(model, train_loader=UAVDataLoader(dataset, **hparams.dataloader_hparams))
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hparams_file", "-p", type=str, nargs=1)
    parser.add_argument("-experiment_type", "-t", type=str, nargs=1, default=[None])
    parser.add_argument("-input_file", "-i", type=str, nargs=1, default=[""])
    parser.add_argument("-output_file", "-o", type=str, nargs=1, default=[""])
    args = parser.parse_args()
    run_experiment(args.hparams_file[0], experiment_type=args.experiment_type[0], args=args)


if __name__ == "__main__":
    main()


