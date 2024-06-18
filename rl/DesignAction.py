"""
Description: Classes defining actions for the UAV design MDP

Authors: Sebastiaan De Peuter and Aleksanteri Sladek
Created 15th June 2022
Modified 28th July 2022

Aalto University
Department of Computer Science
Probabilistic Machine Learning Group (PML)
Cooperative AI and User Modelling Project: UAV-Design

------------------------------------------------------------------------------------------------------------------------

IMPORTANT ASSUMPTIONS:
- Action cannot result in an 'disconnected' UAV design, i.e, a design where two parts of the drone graph have no path
  to one another. In practice, this would mean a single drone design consisting of two independent structures.
    - Hence, actions that would result in such a drone are not allowed, validated via Dijkstra's algorithm.

- Every new component is FULLY CONNECTED to its neighbors, i.e, connections are formed to every existing neighbor
  component
    - Neighbor, as in any component 1 grid coordinate unit away from the new component
- Connections are only formed along grid coordinate lines, diagonal connections are not formed
- Connections can only be one grid coordinate unit long
- At initial state ("",0,0) the payload change action is ignored

- Delete action remaps drone component IDs if needed. Drone component IDs must be used sequentially.

"""
from abc import ABC, abstractmethod
from typing import Tuple, List
from copy import copy

import numpy as np

from rl.DesignState import UAVDesign
from data.Constants import *
from utils.utils import all_connected, prepended_join, remap_components


class DesignAction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def apply(self, uav_design: UAVDesign) -> UAVDesign:
        """ Method for applying the action to a state, yielding the next state"""
        pass


class EditActionAdd(DesignAction):
    """
    Class originally from Sebastiaan's 'UAV design' jupyter notebook
    Modified by Aleksanteri

    UAV Design Add action: Adds a component to UAV design

    """

    def __init__(self, component_type: str, location: Tuple[str, str], connections: List[str], size=0, symmetric=False):
        super(EditActionAdd, self).__init__()
        self.component_id = None

        self.component_type = component_type
        self.location = location  # Tuple of letters
        self.sym_location = None
        self.sym_id = None
        self.connections = connections  # IDs of component to connect to
        self.sym_connections = []
        self.size_int = size
        self.is_symmetric = symmetric

        assert component_type in COMPONENT_TYPE_IDS, f"Invalid component type ID '{component_type}'"
        assert location[0] in X_LETTER_COORDS, f"Invalid X coordinate '{location[0]}'"
        assert location[1] in Z_LETTER_COORDS, f"Invalid Z coordinate '{location[1]}'"
        for conn_id in connections:
            assert conn_id in COMPONENT_IDS, f"Invalid connection component ID '{conn_id}'"

    def __str__(self):
        size_token = ""
        if self.size_int < 0:
            size_token = DECREMENT_SYMBOL
        if self.size_int > 0:
            size_token = INCREMENT_SYMBOL

        size_string = size_token * abs(self.size_int)

        # If component ID has not been assigned to the action yet, use a placeholder
        if self.component_id is None:
            comp_id = '<COMP_ID>'
        else:
            comp_id = self.component_id
        
        act_str = f"add {COMP_PREFIX}{comp_id}{self.location[0]}{self.location[1]}{self.component_type}{size_string} " \
               f"{prepended_join([conn_id + comp_id for conn_id in self.connections], CONN_PREFIX)}"
        
        if self.is_symmetric:
            act_str += f" sym-add {COMP_PREFIX}{self.sym_id}{self.sym_location[0]}{self.sym_location[1]}{self.component_type}{size_string} "
        
        return act_str

    def __repr__(self):
        return str(self)

    def apply(self, uav_design: UAVDesign) -> UAVDesign:
        """
        Function for getting successor state when applying action to given state.
        I.e Modifies the given UAV string (design state) by, in this case, adding a component
        """

        # Get the elements of the unmodified UAV design
        components, connections, payload, controller_idx = uav_design.get_elements()
        
        coord2id = get_coord_id_map(components) # mapping for later reference

        # Get the next available component ID, if it exists
        try:
            self.component_id = COMPONENT_IDS[len(components)]
        except IndexError as e:
            print("Error: No more free component IDs")
            raise e
        
        # Component 'a' cannot have a symmetric add, as it is in the center
        self.is_symmetric = self.is_symmetric and (self.component_id != COMPONENT_IDS[0])

        # Component size as number and in the UAV grammar
        size_string = DECREMENT_SYMBOL * abs(self.size_int) if self.size_int < 0 else INCREMENT_SYMBOL * self.size_int

        # Construct the component in UAV grammar
        components.append(self.component_id + self.location[0] + self.location[1] + str(self.component_type) +
                          size_string)
        
        # Make the components complement if symmetric actions
        c_component_id = None
        if self.is_symmetric:
            (c_x, c_z) = get_complement_location(self.location, dtype="str")
            self.sym_location = (c_x, c_z)

            # Construct the complement component in UAV grammar
            c_component_id = COMPONENT_IDS[COMPONENT_IDS.index(self.component_id) + 1]
            self.sym_id = c_component_id
            components.append(c_component_id + c_x + c_z + str(self.component_type) +
                              size_string)
        
        # Now make the connections to the new component(s)
        # Note that SELF.connections is a list of NEW connection ids!!!
        if len(self.connections) > 0:
            # Make sure the connection id's specified are within the component id's of the unmodified design
            existing_components = [c[0] for c in components]
            for c in self.connections:
                assert c in existing_components, f"Error, cannot specify connection to {c} " \
                                                 f"as no such component exists in {uav_design.__str__}"

                connections.append(c + self.component_id)
                
                # Make the complement of this connection
                if self.is_symmetric:
                    # Get the location of connection component, and find its complement's id
                    
                    comp_conn_id = None
                    for conn_comp in components:
                        if conn_comp[0] != c:
                            continue
                        comp_loc = get_complement_location((conn_comp[1], conn_comp[2]), dtype="int")
                        try:
                            comp_conn_id = coord2id[comp_loc]
                            break
                        except KeyError as e:
                            raise e
                        
                    assert comp_conn_id is not None
                    connections.append(comp_conn_id + c_component_id)
                    self.sym_connections.append(comp_conn_id + c_component_id)
        else:
            # The only case in which a component connected to nothing (len(connections) == 0) is when it's the first one
            assert self.component_id == COMPONENT_IDS[0]

        new_uav_design = UAVDesign(elems=(components, connections, payload, controller_idx))

        # Ensure that there are no "free floating" components left as a result of the addition
        assert all_connected(new_uav_design.__str__()), f"Error, action results in disconnected components! ({new_uav_design.__str__(), new_uav_design.predecessor_action})"

        return new_uav_design


# class EditActionSymmetricAdd(DesignAction):
#     def __init__(self, add_actions: Tuple[EditActionAdd, EditActionAdd]):
#         self.actions = add_actions
#         super(EditActionSymmetricAdd, self).__init__()
#
#     def apply(self, uav_str: UAVDesign):
#         new_uav = self.actions[0].apply(uav_str)

class EditActionDel(DesignAction):
    """
    Class originally from Sebastiaan's 'UAV design' jupyter notebook
    Modified by Aleksanteri

    Action for deleting a component from a drone

    WARNING: Renames component id's of the drone! Component IDs in a drone string must be used sequentially, which means
    that if from a drone with components a,b,c component b is removed, component c must be renamed to b.

    """
    def __init__(self, component_id, symmetric=False):
        super(EditActionDel, self).__init__()
        self.component_id = component_id
        self.symmetric = (symmetric and (component_id != COMPONENT_IDS[0]))
        self.sym_id = None

    def __str__(self):
        act_str = 'del ' + str(self.component_id)
        
        if self.symmetric:
            act_str += f" sym-del {self.sym_id}"
        
        return act_str

    def __repr__(self):
        return str(self)

    def apply(self, uav_design: UAVDesign) -> UAVDesign:
        # Parse out the UAV Design elements
        components, connections, payload, controller_idx = uav_design.get_elements()

        component_ids = [c[0] for c in components]
        assert self.component_id in component_ids, f"Error: Component ID {self.component_id} does not exist in " \
                                                   f"{uav_design.__str__}"

        c_idx = component_ids.index(self.component_id)
            
        comp = components[c_idx]
        del components[c_idx]
        del component_ids[c_idx]
        
        if self.symmetric:
            sym_loc = get_complement_location((comp[1], comp[2]), dtype="str")
            sym_idx = None
            for idx, comp in enumerate(components):
                if sym_loc[0] + sym_loc[1] in comp:
                    sym_idx = idx
                    self.sym_id = comp[0]
                    break
            if sym_idx is None:
                raise AssertionError("Error: Design not symmetric, cant do symmetric deletion!")
            
            del components[sym_idx]
            del component_ids[sym_idx]

        # Remove all connections that contain specified component
        new_connections = []
        for c in connections:
            if self.component_id not in c:
                if self.symmetric and self.sym_id in c:
                    continue
                new_connections.append(c)

        # Remap components
        new_components, new_connections = remap_components(components, new_connections)

        new_uav_design = UAVDesign(elems=(new_components, new_connections, payload, controller_idx))

        # Ensure that there are no "free floating" components left as a result of the deletion
        assert all_connected(new_uav_design.__str__()), "Error, action results in disconnected components!"

        return new_uav_design


class EditActionPayload(DesignAction):
    """ The action for editing a drone's payload """
    """ !!! Depracated, we've decided not to use this !!"""
    def __init__(self, new_payload: int):
        super(EditActionPayload, self).__init__()

        assert new_payload >= 0, f"Invalid payload value {new_payload}, must be non-negative integer."
        self.payload = new_payload

    def __str__(self):
        return f"edit payload {self.payload}"

    def apply(self, uav_design: UAVDesign):
        components, connections, payload, controller_idx = uav_design.get_elements()
        new_uav_design = UAVDesign(elems=(components, connections, self.payload, controller_idx))

        return new_uav_design
    
    
class EditActionSize(DesignAction):
    """ The action for editing a drone component's size """
    def __init__(self, component_id, new_size, symmetric=False):
        super(EditActionSize, self).__init__()
        self.component_id = component_id
        self.size = abs(new_size)
        self.symmetric = symmetric
        self.sym_id = None
        
        try:
            self.sign = int(new_size / self.size)
        except ZeroDivisionError:
            self.sign = 1
        
    def __str__(self):
        new_size = self.sign*self.size
        act_str = f"size '{self.component_id}' to {new_size}"
        
        if self.symmetric:
            act_str += f" sym-size '{self.sym_id}' to {new_size}"
            
        return act_str
    
    def apply(self, uav_design: UAVDesign):
        components, connections, payload, controller_idx = uav_design.get_elements()

        component_ids = [c[0] for c in components]
        assert self.component_id in component_ids, f"Error: Component ID {self.component_id} does not exist in " \
                                                   f"{uav_design.__str__}"
        
        # Get the component to be modified, without its current size
        comp_idx = component_ids.index(self.component_id)
        component = components[comp_idx][:4]
        
        new_size_string = f"{INCREMENT_SYMBOL*self.size}" if self.sign >= 0 else f"{DECREMENT_SYMBOL*self.size}"
        new_component = f"{component}{new_size_string}"
        components[comp_idx] = new_component
        
        # Symmetric actions for all except middle component (always 'a')
        if self.symmetric and (component[0] != COMPONENT_IDS[0]):
            complement_id = get_complement_id(self.component_id, components)
            self.sym_id = complement_id
            complement_idx = component_ids.index(complement_id)
            complement_comp = components[complement_idx]
            new_complement = f"{complement_comp[:4] + new_size_string}"
            components[complement_idx] = new_complement
        
        new_design = UAVDesign(elems=(components, connections, payload, controller_idx))
        
        return new_design
    

class EditActionNoop(DesignAction):
    """
    The "NOOP" / "No-op" action, i.e, do nothing

    """
    def __init__(self):
        super(EditActionNoop, self).__init__()
        
    def __str__(self):
        return NOOP_TOKEN
        
    def apply(self, uav_design: UAVDesign):
        return copy(uav_design)
    

class EditActionDone(DesignAction):
    """
    The DONE action, terminal state indicator, indicates that the design is complete
    
    """
    def __init__(self):
        super(EditActionDone, self).__init__()
        
    def __str__(self):
        return DONE_TOKEN

    def apply(self, uav_design: UAVDesign):
        return copy(uav_design)
    

########################################################################################################################
# Functions for constructing action class instances
########################################################################################################################

def get_sizes(size, incremental=False) -> List[int]:
    """ Returns all size increments and decrements, except 'size' """
    sizes = list(np.flip(np.asarray(COMPONENT_DECREMENTS))) + [0] + COMPONENT_INCREMENTS
    
    assert size in sizes, f"Invalid size int {size}"
    
    # If not incremental, return all except given size
    if not incremental:
        new_sizes = copy(sizes)
        new_sizes.remove(size)
    # If incremental, return only the increment and decrement sizes
    else:
        new_sizes = []
        idx = sizes.index(size)
        
        # Try getting the sizes at either "side" of current size. If at max increment/min decrement, adds nothing
        try:
            new_sizes.append(sizes[idx + 1])
        except IndexError:
            pass
        try:
            new_sizes.append(sizes[idx - 1])
        except IndexError:
            pass
        
    return new_sizes

    
def collect_size_actions(uav_design: UAVDesign, incremental=False, symmetric=False, debug=False) -> List[UAVDesign]:
    components, connections, capacity, controller_idx = uav_design.get_elements()
    
    # If no components, no size actions available
    if len(components) == 0:
        return []
    
    edit_actions = []
    
    for c in components:
        new_sizes = []
        component_id = c[0]
        c_x, c_y = c[1], c[2]
        if symmetric:
            
            if X_LETTER_COORDS.index(c_x) > X_HP:
                continue
                
            if (X_LETTER_COORDS.index(c_x) == X_HP) and (Z_LETTER_COORDS.index(c_y) > Z_HP):
                continue
                
            # elif c_x == "M" and c_y == "M":
            #    continue
        
        # No size string case
        if len(c) == 4:
            new_sizes = get_sizes(0, incremental=incremental)
        
        # Size string exists case
        else:
            component_size = len(c[4:])
            component_size_sign = c[4]
            
            # Figure out if increment or decrement
            if component_size_sign == INCREMENT_SYMBOL:
                sizes = COMPONENT_INCREMENTS
                sign = 1
            else:
                sizes = COMPONENT_DECREMENTS
                sign = -1
                
            component_size = sign*component_size
            
            # Figure out where in the range of our pre-defined "standard" the component is in
            standard_c_size = None
            for idx in range(1, len(sizes)):
                l, u = abs(sizes[idx - 1]), abs(sizes[idx])
                
                if l <= abs(component_size) <= u:
                    standard_c_size = l if abs(component_size) - l <= u - abs(component_size) else u
                    standard_c_size = standard_c_size * sign
                    break
                    
            # Case where we've reached the max size increment/decrement
            if standard_c_size is None:
                standard_c_size = sizes[-1]
            
            new_sizes = get_sizes(standard_c_size, incremental=incremental)
                    
        # Construct the size actions
        for size in new_sizes:
            edit_actions.append(EditActionSize(component_id, size, symmetric=symmetric))

    # Process the action objects into MDP actions (i.e just states)
    action_list = []
    for edit_action in edit_actions:
        try:
            action = edit_action.apply(uav_design)
            action.predecessor_action = edit_action.__str__()
            action_list.append(action)
            # print(edit_action.__str__())
        except AssertionError as e:
            if debug:
                print(e)
                
    return action_list
        
        
def collect_add_actions(uav_design: UAVDesign, fc_drones=True, no_size=False, symmetric=False, debug=False) -> List[UAVDesign]:
    """
    Originally Sebastiaan's 'collect_add_actions' function from UAVDesign.ipynb

    Modified by Aleksanteri

    no_size determines whether to return the size variants for one component as separate actions
    fc_drones determines whether actions add components in a "fully connected" manner, i.e a new component added is
    connected to all its nearest neighbors (1 grid square away)
    
    Args:
        uav_design - A UAVDesign object
        fc_drones - Boolean deciding whether to add components in "fully connected" manner, default is True.
                    If False, single connection is established for each 1-NN component as separate actions
        no_size - Boolean deciding whether to return just the new component or new component and its size variants
    """

    components, connections, capacity, controller_idx = uav_design.get_elements()
    
    # The list of collected add actions
    edit_actions = []

    # Full design case: no components can be added anymore, return nothing
    if len(components) == len(COMPONENT_IDS):
        return []
    
    # Full design case (symmetric actions): need at least 2 free IDs
    if symmetric and (len(components) == (len(COMPONENT_IDS) - 1)):
        return []
        

    # Start state case: there are no components in the design, add battery to center
    if len(components) == 0:
        
        # NOTE: Symmetric is false because there's no symmetric action for the center point
        edit_actions.append(EditActionAdd(COMPONENT_TYPE_IDS[0], ("M", "M"), [], 0, symmetric=False))
        if not no_size:
            for size in COMPONENT_INCREMENTS + COMPONENT_DECREMENTS:
                try:
                    edit_actions.append(EditActionAdd(COMPONENT_TYPE_IDS[0], ("M", "M"), [], size, symmetric=False))
                except AssertionError as e:
                    if debug:
                        print(e)
        
        return [edit_actions[i].apply(uav_design) for i in range(len(edit_actions))]

    # Intermediate state case: there are some components in the design, find the free slots and collect actions
    component_dict = {}
    neighbours = {}
    locations = {}
    
    for c in components:
        comp_id, comp_x, comp_z, comp_type, comp_size = c[0], c[1], c[2], c[3], c[4:]
        component_dict[comp_id] = c
        neighbours[comp_id] = []
        loc_tuple = (X_LETTER_COORDS.index(comp_x), Z_LETTER_COORDS.index(comp_z))
        locations[loc_tuple] = c
        
    for c in connections:
        orig_id = c[0]
        dest_id = c[1]
        neighbours[orig_id].append(dest_id)
        neighbours[dest_id].append(orig_id)
        
    # Find possible new component locations
    new_locs = {}
    for c in components:
        loc = (X_LETTER_COORDS.index(c[1]), Z_LETTER_COORDS.index(c[2]))
        
        # Ignore the right half of the coordinate space
        if symmetric:
            if loc[0] > X_HP:
                continue

            # Special case of coord locs right on the half-plane, just consider one
            if loc[0] == X_HP and loc[1] > Z_HP:
                continue
        
        for new_loc in get_neighbor_locs(loc):
            # Check that new x location is a valid coordinate
            if not (0 <= new_loc[0] < COORD_GRID_SIZE[0]):
                continue
                
            # Check that new z location is a valid coordinate
            if not (0 <= new_loc[1] < COORD_GRID_SIZE[1]):
                continue
                
            # If symmetric actions, check that not in right half space
            if symmetric:
                if new_loc[0] > X_HP:
                    continue
                
            # Check that components do no already exist at new location
            if new_loc in locations.keys():
                continue
            
            if new_loc not in new_locs.keys():
                new_locs[new_loc] = [c[0]]
            else:
                new_locs[new_loc].append(c[0])
            
            # Find connections for the new location
            if fc_drones:
                new_connections = []
                for neighbor_loc in get_neighbor_locs(new_loc):
                    if neighbor_loc in locations.keys():
                        new_connections.append(locations[neighbor_loc][0])
            else:
                new_connections = c[0]
                
            # Add actions to the list
            new_loc_s = (X_LETTER_COORDS[new_loc[0]], Z_LETTER_COORDS[new_loc[1]])
            for comp_type in COMPONENT_TYPE_IDS:
                edit_actions.append(EditActionAdd(comp_type, new_loc_s, new_connections, 0, symmetric=symmetric))
                
                # Return size variations of components
                if not no_size:
                    for size in COMPONENT_INCREMENTS + COMPONENT_DECREMENTS:
                        try:
                            edit_actions.append(EditActionAdd(comp_type, new_loc_s, new_connections, size, symmetric=symmetric))
                        except AssertionError as e:
                            if debug:
                                print(e)

    # Process the action objects into MDP actions (i.e just states)
    action_list = []
    for edit_action in edit_actions:
        try:
            action = edit_action.apply(uav_design)
            action.predecessor_action = edit_action.__str__()
            action_list.append(action)
            # print(edit_action.__str__())
        except AssertionError as e:
            if debug:
                print(e)

    return action_list


def collect_del_actions(uav_design: UAVDesign, symmetric=False, debug=False) -> List[UAVDesign]:
    """
    Sebastiaan's 'collect_del_actions' function from UAVDesign.ipynb

    Modified by Aleksanteri
    """

    components, connections, capacity, controller_idx = uav_design.get_elements()

    edit_actions = []

    component_dict = {}
    neighbours = {}
    for c in components:
        component_dict[c[0]] = c
        neighbours[c[0]] = []

    for c in connections:
        neighbours[c[0]].append(c[1])
        neighbours[c[1]].append(c[0])

    for c in components:
        loc = (c[1], c[2])
        
        # Never delete central component
        if loc[0] == "M" and loc[1] == "M":
            continue
        
        if symmetric and (X_LETTER_COORDS.index(loc[0]) > X_HP):
            continue
        
        # Special case of coord locs right on the half-plane, just consider one
        if symmetric and (X_LETTER_COORDS.index(loc[0]) == X_HP) and (Z_LETTER_COORDS.index(loc[1]) > Z_HP):
            continue
            
        if len(neighbours[c[0]]) < 4:
            # this component can be removed
            try:
                edit_actions.append(EditActionDel(c[0], symmetric=symmetric))
            except AssertionError as e:
                if debug:
                    print(e)

    actions = []
    for edit_action in edit_actions:
        try:
            action = edit_action.apply(uav_design)
            action.predecessor_action = edit_action.__str__()
            actions.append(action)
            # print(edit_action.__str__())
        except AssertionError as e:
            if debug:
                print(e)

    return actions


def collect_noop_actions(uav_design: UAVDesign):
    edit_action = EditActionNoop()
    action = edit_action.apply(uav_design)
    action.predecessor_action = edit_action.__str__()
    return [action]


def collect_done_actions(uav_design: UAVDesign):
    """ Basically just a special noop action """
    edit_action = EditActionDone()
    action = edit_action.apply(uav_design)
    action.predecessor_action = edit_action.__str__()
    return [action]

########################################################################################################################
# Utility functions
########################################################################################################################


# TODO: Remove code duplication in collection functions
def apply_actions():
    pass


def get_id_coord_map(components):
    coord2id = get_coord_id_map(components)
    id2coord = {}
    for key, val in coord2id.items():
        id2coord[val] = key
    return id2coord


def get_coord_id_map(components):
    """ Maps a coordinate tuple to ID of component there """
    coord_id_map = dict()
    for comp in components:
        c_x, c_z = (X_LETTER_COORDS.index(comp[1]), Z_LETTER_COORDS.index(comp[2]))
        coord_id_map[(c_x, c_z)] = comp[0]
        
    return coord_id_map


def get_complement_location(loc: Tuple[object, object], dtype=None) -> Tuple[object, object]:
    """ Finds the complement location for a coord tuple (letter or int)"""
    in_type = str(type(loc[0]))
    
    if dtype is None:
        dtype = in_type
    
    if "str" in in_type:
        x, z = (X_LETTER_COORDS.index(loc[0]), Z_LETTER_COORDS.index(loc[1]))
    else:
        x, z = loc

    c_x, c_z = get_complement_loc_int((x, z))
    
    if dtype == "str":
        return X_LETTER_COORDS[c_x], Z_LETTER_COORDS[c_z]
    else:
        return c_x, c_z
    

def get_complement_loc_int(loc: Tuple[int, int]) -> Tuple[int, int]:
    """
    Returns the coordinate loc's reflection about the x and z half-planes
    
    """
    x, z = loc
    c_x = 2 * (X_HP - x) + x
    
    # Note that this reflects about the z-halfplane line
    c_z = 2 * (Z_HP - z) + z
    
    return c_x, c_z


def get_complement_id(comp_id, components):
    coord_id = get_coord_id_map(components)
    id_coord = get_id_coord_map(components)
    
    c_loc = id_coord[comp_id]
    complement_loc = get_complement_location(c_loc, dtype="int")
    return coord_id[complement_loc]


def get_neighbor_locs(loc: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Utility function

    Returns grid square coordinates around a grid coordinate

    """
    return [(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)]



def get_actions(uav_design: UAVDesign, symmetric=False, no_size=False):
    # TODO: Remove me
    assert symmetric is True
    """ This function determines what actions are returned! Used by the DesignState 'get_successors' method """
    return collect_add_actions(uav_design, symmetric=symmetric, no_size=no_size) + collect_del_actions(uav_design, symmetric=symmetric) + \
        collect_size_actions(uav_design, incremental=True, symmetric=symmetric) + \
        collect_noop_actions(uav_design) + \
        collect_done_actions(uav_design)

        
if __name__ == "__main__":
    # collect_del_actions(UAVDesign(',5,3'))
    from viz.design_visualizer import draw_drone
    uav_str = "*aMM0---,5,3"
    new_uav = UAVDesign(uav_str)
    n_actions = 3
    
    uav_path = []
    n_iters = 1000
    i = 0
    tikz = []
    while True:
        if i > n_iters:
            break
        
        act_class = np.random.randint(n_actions)
        
        if act_class == 0:
            acts = collect_add_actions(new_uav, symmetric=True, no_size=True)
        elif act_class == 1:
            acts = collect_del_actions(new_uav, symmetric=True)
        else:
            acts = collect_size_actions(new_uav, incremental=True, symmetric=True)
            
        if len(acts) < 1:
            print("Do nothing")
            print()
            i += 1
            continue
        
        print(new_uav)
        new_uav = acts[np.random.randint(0, len(acts))]
        print(new_uav.predecessor_action)
        print(new_uav)
        print()
        uav_path.append(new_uav)
        tikz.append(draw_drone(new_uav.to_string(), nodef=True if i == 0 else False))
        i += 1
    
    with open("seq.tikz", 'w') as f:
        f.writelines(tikz)
