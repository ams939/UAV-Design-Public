"""
Baseline model definitions in SKLearn

"""
import os
from abc import abstractmethod

from tqdm import tqdm
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import json

from data.Constants import SIM_RESULT_COL, SIM_METRICS, UAV_STR_COL
from model.UAVModel import UAVModel
from train.Hyperparams import DummyHyperparams, Hyperparams
from train.Logging import DummyLogger
from inference.eval import eval_metrics


class SKLearnModel(UAVModel):
    def __init__(self, hparams):
        super(SKLearnModel, self).__init__(hparams)

        self.targets = hparams.dataset_hparams["target_cols"]
        self.metrics = list(set(SIM_METRICS).intersection(set(self.targets)))

        if len(self.metrics) > 0:
            self.reg = True
        else:
            self.reg = False

        if SIM_RESULT_COL in self.targets:
            self.clf = True
            assert self.targets[-1] == SIM_RESULT_COL, "Assign 'result' column as the last column"
        else:
            self.clf = False

        self.regression_model = None
        self.classification_model = None

        self.n_metrics = None

        self.build()

    @abstractmethod
    def build(self):
        pass

    def forward(self, x, y):
        self.n_metrics = len(self.metrics)
        
        assert len(x) == len(y)

        if self.reg:
            if self.clf:
                metrics = y[:, :-1]
            else:
                metrics = y

            self.reg_forward(x, metrics)

        if self.clf:
            if self.reg:
                labels = y[:, -1]
            else:
                labels = y

            self.clf_forward(x, labels)

        y_pred = self.predict(x)

        return y_pred

    def clf_forward(self, x, y):
        self.classification_model.fit(x, y)

    def reg_forward(self, x, y):
        self.regression_model.fit(x, y)

    def predict(self, x):
        pred_len = x.shape[0]

        y_pred = []
        if self.reg:
            reg_pred = self.regression_model.predict(x)
            
            y_pred.append(reg_pred)

        if self.clf:
            clf_pred = self.classification_model.predict(x)
            
            y_pred.append(clf_pred.reshape(pred_len, 1))

        if len(y_pred) > 1:
            y_pred = np.concatenate(y_pred, axis=1)
        else:
            y_pred = y_pred[0]

        return y_pred

    def load(self):
        file = f"{self.hparams.experiment_folder}/{self.hparams.model_file}"
        
        if os.path.exists(file):
            self.logger.log({"name": self.__name__, "msg": f"Loading existing model from {file}"})
            model = joblib.load(file)
            try:
                self.classification_model = model["cls"]
                self.regression_model = model["reg"]
            except Exception as e:
                self.logger.log({"name": self.__name__, "msg": f"Error loading model:\n{e}"})
        else:
            self.logger.log({"name": self.__name__, "msg": f"No existing model found, new model created."})

    def save(self, mname=""):
        file = f"{self.hparams.experiment_folder}/rf_model.jb"
        
        model = {"reg": self.regression_model, "cls": self.classification_model}
        
        try:
            joblib.dump(model, file)
            self.logger.log({"name": self.__name__, "msg": f"Model saved to {file}"})
        except Exception as e:
            self.logger.log({"name": self.__name__, "msg": f"Error saving model: \n{e}"})


class SimLinLogReg(SKLearnModel):
    def __init__(self, hparams):
        super(SimLinLogReg, self).__init__(hparams)

    def build(self):
        self.regression_model = LinearRegression()
        self.classification_model = LogisticRegression()
        self.load()


class SimRandomForest(SKLearnModel):
    def __init__(self, hparams):
        super(SimRandomForest, self).__init__(hparams)

    def build(self):
        self.regression_model = RandomForestRegressor(random_state=self.hparams.seed, criterion="absolute_error", 
                                                        verbose=1, n_jobs=6, oob_score=True)
        self.classification_model = RandomForestClassifier(random_state=self.hparams.seed, verbose=1, n_jobs=6, oob_score=True)
        self.load()
    

def train_k_fold(hparams: Hyperparams):
    dataset = hparams.dataset_class(hparams)
    target_cols = hparams.dataset_hparams["target_cols"]

    X, y = np.asarray([inp.numpy() for inp, _ in dataset.data]), np.asarray([tgt.numpy() for _, tgt in dataset.data])

    n_folds = hparams.n_folds
    
    if n_folds == 0:
        model = SimRandomForest(hparams)
        y_pred = model.forward(X, y)
        model.save()
        train_results = eval_metrics(y, y_pred, target_cols)
        print(train_results)
        return model
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=hparams.seed)

    results = {}
    for fold_index, (train_indices, test_indices) in tqdm(enumerate(kf.split(X))):
        model = SimRandomForest(hparams)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        y_train_pred = model.forward(X_train, y_train)

        # Get predictions for validation set
        y_test_pred = model.predict(X_test)

        # Calculate errors
        train_results = eval_metrics(y_train, y_train_pred, target_cols)
        test_results = eval_metrics(y_test, y_test_pred, target_cols)
        result_sets = {"train": train_results, "test": test_results}

        for set_name, result_set in result_sets.items():
            if set_name not in results.keys():
                results[set_name] = {}
            for feature, f_results in result_set.items():
                if feature not in results[set_name].keys():
                    results[set_name][feature] = {}

                for metric, m_result in f_results.items():
                    if metric not in results[set_name][feature].keys():
                        results[set_name][feature][metric] = 0

                    results[set_name][feature][metric] += result_set[feature][metric]

                    if fold_index == n_folds - 1:
                        results[set_name][feature][metric] /= n_folds

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    # import sys
    # sys.path.insert(0, '')
    
    hparams = Hyperparams("hparams/rf_hparams.json")
    hparams.experiment_id = "5"
    train_k_fold(hparams)

