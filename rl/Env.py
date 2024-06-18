"""
Rainbow Deep Q-Learning implementation from https://github.com/Kaixhin/Rainbow by Kai Arulkumaran et al.
https://github.com/Kaixhin/Rainbow/blob/master/env.py


Modified by Aleksanteri Sladek, 27.7.2022
    - Minor refactoring
    - Added interface Env

"""

from abc import abstractmethod
import sys

from copy import deepcopy

from rl.DesignState import UAVDesign
from train.Hyperparams import Hyperparams, DummyHyperparams
from train.Logging import init_logger
from rl.Curriculum import FixedCurriculum
from data.Constants import SIM_SUCCESS, SIM_METRICS, SIM_OUTCOME


class Env:
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        """ Returns tuple (state, reward, done) """
        pass

    @abstractmethod
    def train(self):
        """ Set into training mode """
        pass

    @abstractmethod
    def eval(self):
        """ Set into eval mode """
        pass

    @abstractmethod
    def action_space(self):
        """ Returns the number of available actions """
        pass

    @abstractmethod
    def reset(self):
        """ Reset the environment """
        pass


class UAVEnv(Env):
    """
    Class that translates the 'AtariEnv' class to the UAV domain. Goals were to keep the functioning and interface
    as close as possible.

    """
    def __init__(self, hparams, verbose=True):
        super(UAVEnv, self).__init__()
        self.hparams = hparams
        self.device = hparams.device
        self.obj = None
        
        if "use_cache" in hparams.keys():
            self.use_cache = hparams.use_cache
        else:
            self.use_cache = False
        
        self.cache = {}

        # Internal trackers
        self.drone_stats = dict()
        self.obj_stats = dict()
        self.run_stats = dict()

        self.training = True  # Consistent with model training mode
        
        if "curriculum_class" in hparams.keys():
            try:
                self.curriculum = hparams.curriculum_class(hparams)
                if verbose:
                    self.hparams.logger.log({"msg": "Curriculum successfully initialized", "debug": True})
            except Exception as e:
                if verbose:
                    self.hparams.logger.log({"msg": f"Couldn't initialize specified curriculum {str(hparams.curriculum_class)}"})
                raise e
        else:
            self.hparams.logger.log({"msg": "Couldn't initialize curriculum, trying FixedCurriculum..."})
            self.curriculum = FixedCurriculum(hparams)

        try:
            sim_hparams = Hyperparams(self.hparams.simulator_hparams.hparams_file)
            sim_hparams.device = hparams.device
            sim_hparams.logger = self.hparams.logger
            self.hparams.logger.log({"name": "Env", "msg": f"Initializing {hparams.simulator_class.__name__}", "debug": True})
        except (FileNotFoundError, AttributeError):
            
            # There are no hyperparameters for the Hyform simulator currently
            if "HyFormSimulator" in hparams.simulator_class.__name__:
                sim_hparams = DummyHyperparams()
            else:
                self.hparams.logger.log({"name": "Env", "msg": f"Error, couldn't find the hparams file for simulator."})
                sys.exit(-1)

        self.simulator = self.hparams.simulator_class(sim_hparams)
        self.max_episode_length = hparams.max_episode_length  # Maximum number of actions to perform in one design
        self.episode_length = 0

        try:
            self.episode_count = hparams.episode_offset
            assert self.episode_count is not None
        except (KeyError, AssertionError):
            self.episode_count = 0

        # Initialize the current state within environment
        if hparams.init_state is not None:
            self.init_state_str = hparams.init_state
        else:
            raise AssertionError("No initial state defined in hyperparams, define 'init_state'")

        self.init_state = None
        self.state = None
        self.state_tensor = None
        self.actions = None

        self.reward_class = None

        self.init_state = self.initial_state()
        self._set_state(self.init_state)
        
        self._init_trackers()
        
        self.reset()

    def _get_state(self):
        return self.state

    def _set_state(self, new_state: UAVDesign):
        self.state = new_state
        self.state_tensor = new_state.to_tensor(self.hparams.encoding)
        actions = self.state.get_successors(symmetric=self.hparams.symmetric_actions, no_size=self.hparams.no_size)
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))

    def step(self, action: UAVDesign):

        # Apply action (in this problem we consider actions as the next state directly)
        current_state = self.state
        self._set_state(action)

        # Get reward and whether action results in terminal state (determined by reward function)
        reward, outcome = self.reward_fn(current_state, action)
        
        self.episode_length += 1
        self.run_stats["n_iteration"] += 1
        self.run_stats["cum_reward"] += reward
        self.obj_stats[self.obj]["cum_reward"] += reward

        done = self.check_termination_condition(action, outcome)

        if done:
            self.obj_stats[self.obj]["avg_final_reward"] += reward

        # Return next state, reward, done
        return self.state, reward, done
    
    def check_termination_condition(self, action, outcome):
        # Max iterations limit
        if self.episode_length > self.max_episode_length:
            self.hparams.logger.log({"name": "Env",
                                     "msg": f"Max iteration limit met.",
                                     "debug": True})
            done = True
        else:
            done = False
        
        # Objective completed limit
        if outcome:
            done = True
        
        # Logging
        if done:
            self.hparams.logger.log({"name": "Env",
                                    "msg": "Terminal state reached.",
                                     "debug": True})
    
            objective_satisfied = self.reward_class.objective_complete(action.get_metrics())
            if objective_satisfied:
                self.obj_stats[self.obj]["n_complete"] += 1
                self.run_stats["n_successful_episode"] += 1
                self.hparams.logger.log({"name": "Env",
                                         "msg": f"Objective {self.obj} completed!",
                                         "debug": True})
            else:
                self.hparams.logger.log({"name": "Env",
                                         "msg": f"Objective {self.obj} not complete.",
                                         "debug": True})
    
        return done

    def reward_fn(self, state, action):
        # Calculate the metrics of the states if necessary
        self.check_metrics(state)
        self.check_metrics(action)

        if action.result == SIM_SUCCESS:
            self.drone_stats["n_stable"] += 1
            self.obj_stats[self.obj]["n_stable"] += 1

        act_reward, terminal_state = self.reward_class.get_reward(state, action)
        
        return act_reward, terminal_state
    
    def check_metrics(self, state):
        if not state.has_metrics:
            state_str = state.to_string()
            cache_miss = False
            if self.use_cache:
                try:
                    state_sim_metrics = self.cache[state_str]
                except KeyError as e:
                    cache_miss = True
        
            if not self.use_cache or cache_miss:
                state_sim_results = self.simulator.simulate(state_str, return_df=False)
                
                state_sim_metrics = dict()
                for key in SIM_METRICS + [SIM_OUTCOME]:
                    state_sim_metrics[key] = state_sim_results[key]
                    
                if self.use_cache:
                    self.cache[state_str] = state_sim_metrics
            
            state.set_metrics(**state_sim_metrics)

    def action_space(self):
        return len(self.actions)
    
    def get_objective(self):
        assert self.obj is not None
        return self.reward_class.get_objective()

    def get_active_objectives(self):
        return self.curriculum.get_active_objectives()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def initial_state(self):
        return UAVDesign(self.init_state_str)
    
    def _init_trackers(self):
        for obj in self.curriculum.get_all_objectives():
            self.obj_stats[obj] = dict()
            self.obj_stats[obj]["n_attempts"] = 0
            self.obj_stats[obj]["n_complete"] = 0
            self.obj_stats[obj]["cum_reward"] = 0
            self.obj_stats[obj]["n_stable"] = 0
            self.obj_stats[obj]["avg_final_reward"] = 0
            
        self.drone_stats["n_stable"] = 0
        self.drone_stats["n_successful"] = 0
        
        self.run_stats["n_episode"] = 0
        self.run_stats["n_successful_episode"] = 0
        self.run_stats["n_iteration"] = 0
        self.run_stats["cum_reward"] = 0

    def reset(self):
        """
        Called at the end of each episode, resets the environment state and increments counters
        """

        # Increment internal counters
        self.episode_count += 1
        self.run_stats["n_episode"] += 1

        self.episode_length = 0
        
        # Get the objective for the next episode
        self.obj = self.curriculum.get_objective(self.episode_count)

        self.obj_stats[self.obj]["n_attempts"] += 1
        self.hparams.logger.log({"name": "Env", "msg": f"Training with objective {self.obj}", "debug": True})
        
        # Initialize the correct reward function for the next episode
        self.hparams["reward_hparams"]["objective"] = self.obj
        self.reward_class = self.hparams.reward_class(self.hparams)
        
        # Re-initialize starting state and set correct payload (determined by objective)
        self.init_state = self.initial_state()
        self.init_state.set_payload(self.reward_class.payload)
        
        # Get metrics for the initial state
        self.reward_fn(self.init_state, self.init_state)

        # Return initial state for the UAV design
        self.state = deepcopy(self.init_state)
  
        return self.init_state

