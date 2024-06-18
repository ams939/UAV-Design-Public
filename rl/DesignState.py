from abc import ABC, abstractmethod
from copy import copy
from typing import Tuple, List

import torch

from data.datamodel.Grammar import UAVGrammar
from data.Constants import COMP_PREFIX, CONN_PREFIX, ELEM_SEP, SIM_SUCCESS, RANGE_COL, COST_COL, VELOCITY_COL, \
    SIM_RESULT_COL, PAYLOAD_COL
from utils.utils import prepended_join, uav_str_to_mat, uav_str_to_index


class DesignState(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_successors(self):
        pass
    
    @abstractmethod
    def to_tensor(self):
        pass


class UAVDesign(DesignState):
    """
    Class for representing and working with UAV designs states, based on structure,
    grammar and operations defined in:

    B. Song et al. (2020). "Toward Hybrid Teams: A Platform To Understand Human-Computer
    Collaboration During the Design of Complex Engineered Systems."

    """
    def __init__(self, uav_str: str = None, elems: Tuple[List[str], List[str], int, int] = None):
        """
        Args: Specify either uav_str or elems, not both. Specify neither for initial state / default state
            uav_str: String in UAV grammar format
            elems: Tuple of (components, connections, payload, controller_idx)

        """

        if elems is None:
            elems = ([], [], 0, 0)

        super(UAVDesign, self).__init__()
        self.uav_str = uav_str
        self.components = elems[0]
        self.connections = elems[1]
        self.payload = elems[2]
        self.controller_idx = elems[3]

        self.has_metrics = False
        self.range = None
        self.cost = None
        self.velocity = None
        self.result = None
        self.is_stable = None  # Simulator outcome (Success, Failure)

        self._initialize()
        self.actions = None
        self.uav_tensor = None
        
        self.predecessor_action = None

    def __str__(self):
        return self.uav_str

    def __eq__(self, uav_design):
        return torch.allclose(self.to_tensor(), uav_design.to_tensor())

    def _initialize(self):
        if self.uav_str is not None:
            parser = UAVGrammar()

            try:
                self.components, self.connections, self.payload, self.controller_idx = parser.parse(self.uav_str)
            except AssertionError as e:
                raise ValueError(f"Invalid UAV string '{self.uav_str}': {e}")
        else:
            self.uav_str = self.to_string()

    def to_string(self):
        """ Converts internal structures to string """
        comp_string = prepended_join(self.components, COMP_PREFIX)
        conn_string = prepended_join(self.connections, CONN_PREFIX)
        uav_struct_string = comp_string + conn_string

        uav_string = ELEM_SEP.join([uav_struct_string, str(self.payload), str(self.controller_idx)])

        return uav_string

    def to_tensor(self, encoding="matrix"):
        """ Converts state to a tensor """
        if self.uav_tensor is None:
            if encoding == "matrix":
                self.uav_tensor = uav_str_to_mat(self.uav_str, encode_connections=False, encoding=encoding)
            elif encoding == "index":
                self.uav_tensor = uav_str_to_index(self.uav_str)
        return self.uav_tensor

    def get_successors(self, symmetric=False, no_size=False) -> List[DesignState]:

        if self.actions is None:
            from rl.DesignAction import get_actions
            self.actions = get_actions(self, symmetric=symmetric, no_size=no_size)

        return self.actions

    """
    Getters and setters
    """
    def get_elements(self):
        return self.get_components(), self.get_connections(), self.get_payload(), \
               self.get_controller()

    def get_components(self):
        return copy(self.components)

    def get_connections(self):
        return copy(self.connections)

    def get_payload(self):
        return copy(self.payload)

    def get_controller(self):
        return copy(self.controller_idx)

    def set_payload(self, new_payload: int):
        self.payload = new_payload
        self.uav_str = self.to_string()
    
    def set_metrics(self, range, cost, velocity, result):
        self.range = range
        self.cost = cost
        self.velocity = velocity
        self.result = result
        self.is_stable = result == SIM_SUCCESS
        self.has_metrics = True

    def get_metrics(self) -> dict:
        assert self.has_metrics is True, 'Error, no metrics assigned to design!'
        return {
            RANGE_COL: self.range,
            COST_COL: self.cost,
            VELOCITY_COL: self.velocity,
            SIM_RESULT_COL: self.result
        }
        

if __name__ == "__main__":
    import numpy as np
    from time import time
    default_uav = ",5,3"

    uav = UAVDesign()
    n_iters = 1000

    s_time = time()
    action_counts = []
    for i in range(n_iters):
        actions = uav.get_successors()

        idx = np.random.randint(0, len(actions))
        action_counts.append(len(actions))

        next_action = actions[idx]

        uav = next_action

    e_time = time()
    time_elapsed_s = e_time - s_time
    print(f"Time: {time_elapsed_s}s")
    print(f"Avg. Time / State: {time_elapsed_s/ n_iters}s")

    print(np.mean(np.asarray(action_counts)))
    print(uav)

