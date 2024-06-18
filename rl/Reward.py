from abc import abstractmethod
import json
import sys
from copy import deepcopy

import numpy as np

from rl.rl_utils import min_max_norm, normalize_metrics
from data.Constants import SIM_SUCCESS, NOOP_TOKEN, DONE_TOKEN, RANGE_COL, COST_COL, VELOCITY_COL, \
    METRIC_NORM_FACTORS, PAYLOAD_COL, SIM_RESULT_COL, SIM_METRICS, OBJECTIVES, EPS


class Reward:
    def __init__(self, hparams):
        self.hparams = hparams
        self.rw_hparams = hparams.reward_hparams
        assert "objective" in self.rw_hparams.keys(), "Error, please specify hparams.objective"

        self.obj_file = "data/datafiles/objective_params.json"

        obj = self.rw_hparams.objective
        assert obj is not None and obj in OBJECTIVES, f"Error, please specify an objective from {OBJECTIVES}"

        try:
            with open(self.obj_file) as f:
                try:
                    self.obj_definitions = json.load(f)[obj]
                except KeyError:
                    print(f"Couldn't find objective {obj} in {self.obj_file}. Aborting...")
                    sys.exit(-1)
        except FileNotFoundError:
            print("Couldn't find the objective definition file. Aborting...")
            sys.exit(-1)

        hparams.logger.log({"name": f"{self.__name__}",
                            "msg": f"Initializing reward with objective {obj}, parameters: {self.obj_definitions}.",
                            "debug": True})

        self.n_objectives = len(self.obj_definitions.keys())

    @abstractmethod
    def get_reward(self, state, action):
        pass

    @property
    def __name__(self):
        return self.__class__.__name__

    def get_thetas(self):
        thetas = dict()
        for key, item in self.obj_definitions.items():
            if key == COST_COL:
                thetas[key] = item["upper"]
            else:
                thetas[key] = item["lower"]


class DeltaReward(Reward):
    """
    """
    def __init__(self, hparams, normalize=True):
        super(DeltaReward, self).__init__(hparams)
        self.range_weight = self.rw_hparams.range_weight
        self.cost_weight = self.rw_hparams.cost_weight
        self.velocity_weight = self.rw_hparams.velocity_weight
        self.normalize = normalize
        self.early_terminate = self.rw_hparams.early_terminate if self.rw_hparams.early_terminate is not None else False
        self.payload = None

        if None in [self.cost_weight, self.range_weight, self.velocity_weight]:
            print("WARNING: Using default weights for quality function")
            self.weights = 1/3 * np.ones(3)
        else:
            self.weights = np.asarray([self.range_weight, self.cost_weight, self.velocity_weight])

        self.obj_definitions_raw = deepcopy(self.obj_definitions)

        # Payload won't be considered a part of the objective, assumed it is known by agent
        assert PAYLOAD_COL in self.obj_definitions_raw.keys(), "Must specify a payload in the objective params"
        self.payload = int(self.obj_definitions_raw[PAYLOAD_COL]["lower"])
        del self.obj_definitions[PAYLOAD_COL]

        # Make sure weights sum to (roughly) 1
        try:
            weights_sum = np.sum(self.weights)
            assert 1 - EPS <= weights_sum <= 1 + EPS, ""
        except AssertionError:
            print(f"WARNING: Reward weights don't sum to 1! (sum={weights_sum})")

        # If normalizing metrics, also normalize the thresholds
        if normalize:
            for metric in self.obj_definitions.keys():
                if metric == PAYLOAD_COL:
                    continue

                for threshold in self.obj_definitions[metric]:
                    metric_val = self.obj_definitions[metric][threshold]
                    metric_min = METRIC_NORM_FACTORS[metric]["min"]
                    metric_max = METRIC_NORM_FACTORS[metric]["max"]
                    metric_val_norm = min_max_norm(metric_val, metric_min, metric_max)
                    self.obj_definitions[metric][threshold] = metric_val_norm

        self.n_objectives = len(self.obj_definitions.keys())

    def get_reward(self, state, action):
        # First processing reward for the action, or, next state
        # If the drone is not stable, return reward 0

        act_reward = self.obj_fn(action)

        if action.predecessor_action == DONE_TOKEN:
            terminal = True
        elif self.objective_complete(action.get_metrics()) and self.early_terminate:
            terminal = True
        else:
            terminal = False

        # Second, processing the current state: Reward 0 if not stable, q-score - penalty otherwise
        state_reward = self.obj_fn(state)

        # Final reward calculation is the delta between next state and current state rewards
        return (act_reward - state_reward), terminal

    def obj_fn(self, state):
        """
        A 'state-value' function

        """
        metrics = state.get_metrics()

        quality = self.quality_fn(metrics)
        penalty = self.penalty_fn(metrics)

        if state.is_stable:
            reward = np.sum(quality * penalty)
            if self.objective_complete(metrics):
                reward += 1

            return reward
        else:
            return 0

    def penalty_fn(self, metrics: dict):
        penalty_strictness = 100

        penalties = np.ones(3)

        if self.normalize:
            metrics = normalize_metrics(metrics)

        if RANGE_COL in self.obj_definitions.keys():
            assert RANGE_COL in metrics.keys(), f"Couldn't find metric '{RANGE_COL}' in metrics passed"
            rnge = metrics[RANGE_COL]
            theta_range = self.obj_definitions[RANGE_COL]["lower"]

            # Sigmoid function at objective threshold
            rnge_penalty = 1 / (1 + np.exp(-penalty_strictness*rnge + penalty_strictness*theta_range))

            penalties[0] = rnge_penalty

        if COST_COL in self.obj_definitions.keys():
            assert COST_COL in metrics.keys(), f"Couldn't find metric '{COST_COL}' in metrics passed"
            cost = metrics[COST_COL]
            theta_cost = self.obj_definitions[COST_COL]["upper"]

            cost_penalty = 1 / (1 + np.exp(penalty_strictness*cost - penalty_strictness*theta_cost))

            penalties[1] = cost_penalty

        if VELOCITY_COL in self.obj_definitions.keys():
            assert VELOCITY_COL in metrics.keys(), f"Couldn't find metric '{VELOCITY_COL}' in metrics passed"
            velocity = metrics[VELOCITY_COL]
            theta_velocity = self.obj_definitions[VELOCITY_COL]["lower"]

            velocity_penalty = 1 / (1 + np.exp(-penalty_strictness * velocity + penalty_strictness * theta_velocity))
            penalties[2] = velocity_penalty

        return penalties

    def quality_fn(self, metrics: dict):
        """ Quality function is just a weighted sum of metrics """

        if self.normalize:
            metrics = normalize_metrics(metrics)
            metric_max = np.ones(3)
        else:
            metric_max = np.asarray(METRIC_NORM_FACTORS[RANGE_COL]["max"],
                                    METRIC_NORM_FACTORS[COST_COL]["max"],
                                    METRIC_NORM_FACTORS[VELOCITY_COL]["max"])

        qualities = np.zeros(3)
        try:
            qualities[0] = metrics[RANGE_COL]
            qualities[1] = metric_max[1] - metrics[COST_COL]
            qualities[2] = metrics[VELOCITY_COL]
        except KeyError as e:
            print(e)
            sys.exit(-1)

        return self.weights*qualities

    def objective_complete(self, metrics):
        n_completed = 0
        for metric in self.obj_definitions.keys():
            if metric == PAYLOAD_COL:
                continue
            
            try:
                metric_lower = self.obj_definitions_raw[metric]['lower']
                assert metric_lower is not None
            except (AssertionError, AttributeError, KeyError):
                metric_lower = -np.inf

            try:
                metric_upper = self.obj_definitions_raw[metric]['upper']
                assert metric_upper is not None
            except (AssertionError, AttributeError, KeyError):
                metric_upper = np.inf

            if metric_lower <= metrics[metric] <= metric_upper:
                n_completed += 1

        stable = metrics[SIM_RESULT_COL] == SIM_SUCCESS
        done = (n_completed >= self.n_objectives) * stable

        return done
    
    def get_objective(self):
        obj = dict()
        obj["objective"] = self.rw_hparams["objective"]
        obj[PAYLOAD_COL] = self.obj_definitions_raw[PAYLOAD_COL]
        
        for key in SIM_METRICS:
            try:
                if key != COST_COL:
                    obj[key] = self.obj_definitions_raw[key]["lower"]
                else:
                    obj[key] = self.obj_definitions_raw[key]["upper"]
                    
            except KeyError:
                continue
                
        return obj

######################
# LEGACY CODE
######################


class ObjectiveReward(Reward):
    def __init__(self, hparams):
        super(ObjectiveReward, self).__init__(hparams)

    def get_reward(self, state, action):
        """
        Returns a reward function value r in range [-10, 10]

        If drone is not stable, returns -10
        Else, returns (n_objectives_met / total_objectives) * 10

        """
        metrics = action.get_metrics()
        outcome = action.result

        # If the drone is not stable, reward is zero and terminal state not reached
        if outcome != SIM_SUCCESS:
            reward = 0
            done = False
            return reward, done

        # TODO: encapsulate within another function to avoid code duplication in reward classes
        # Calculate the number of objective tasks satisfied (threshold values for metrics met)
        n_tasks_satisfied = 0
        n_tasks_total = len(self.obj_definitions.keys())
        for metric in self.obj_definitions.keys():
            try:
                metric_lower = self.obj_definitions[metric]['lower']
                assert metric_lower is not None
            except (AssertionError, AttributeError, KeyError):
                metric_lower = -np.inf

            try:
                metric_upper = self.obj_definitions[metric]['upper']
                assert metric_upper is not None
            except (AssertionError, AttributeError, KeyError):
                metric_upper = np.inf

            if metric_lower <= metrics[metric] <= metric_upper:
                n_tasks_satisfied += 1

        # Objective is satisfied iff all tasks are satisfied
        obj_satisfied = bool(n_tasks_satisfied == n_tasks_total)
        p_tasks_complete = (n_tasks_satisfied / n_tasks_total)

        # Special case for the NOOP action:
        # If the action is the noop action, terminal state reached if objective met. Otherwise, reward is -10.
        if action.predecessor_action == NOOP_TOKEN:
            if obj_satisfied:
                reward = 10 * p_tasks_complete
                done = True
            else:
                reward = -10 + 10 * p_tasks_complete
                done = False
        else:
            reward = p_tasks_complete
            done = False

        return reward, done


class QualityReward(Reward):
    def __init__(self, hparams):
        super(QualityReward, self).__init__(hparams)

    @property
    def __name__(self):
        return self.__class__.__name__

    def get_reward(self, state, action):
        """
        Returns a reward function value r in range [-10, 10]

        If drone is not stable, returns reward=-10 and done=False
        Else, returns reward=(drone_quality_score) done=(True if n_objectives_met == total_objectives else False)

        drone_quality_score = (range * payload * velocity) / cost, clamped to interval [0,10]
            -> From 22' HyForm user study by B. Song et al.

        """

        metrics = action.get_metrics()
        outcome = action.result

        n_completed = 0
        for metric in self.obj_definitions.keys():
            try:
                metric_lower = self.obj_definitions[metric]['lower']
                assert metric_lower is not None
            except (AssertionError, AttributeError, KeyError):
                metric_lower = -np.inf

            try:
                metric_upper = self.obj_definitions[metric]['upper']
                assert metric_upper is not None
            except (AssertionError, AttributeError, KeyError):
                metric_upper = np.inf

            if metric_lower <= metrics[metric] <= metric_upper:
                n_completed += 1

        stable = outcome == SIM_SUCCESS
        p_done = (n_completed / self.n_objectives)*stable
        done = (n_completed >= self.n_objectives)*stable

        # Drone quality score from the Hyform User study paper, clamped to 0-10
        reward = self.quality_fn(metrics, stable) + 10 * p_done

        return reward, done

    def quality_fn(self, metrics: dict, stable):
        """
        Returns a quality score between 0 and 1. If drone not stable, quality automatically zero.
        """
        try:
            q = min((abs(metrics['range']) * abs(metrics['velocity']) * abs(metrics['payload'])) / abs(metrics['cost']), 10) / 10
        except ZeroDivisionError:
            q = 0

        return q


if __name__ == "__main__":
    from rl.DesignState import UAVDesign
    from utils.utils import ddict
    from train.Logging import ConsoleLogger

    hparams = ddict({"objective": "CSTM",
                        "objective_type": "custom",
                        "objective_definitions": {
                        "range": 14
                        }, 'logger': ConsoleLogger(),
                        })
    reward_class = DeltaReward(hparams)

    dummy_state = UAVDesign()
    dummy_action = UAVDesign()
    dummy_state.set_metrics(**{'range': 8.251242637634277, 'cost': 3484.78125, 'velocity': 20.347713470458984, 'result': 'Success'})
    dummy_action.set_metrics(**{'range': 16.636104583740234, 'cost': 3355.6337890625, 'velocity': 9.938084602355957, 'result': 'Success'})
    dummy_action.set_payload(20)
    dummy_action.predecessor_action = NOOP_TOKEN

    r, d = reward_class.get_reward(dummy_state, dummy_action)
    print(r)
