from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from copy import copy

import numpy as np

from data.Constants import *


class Grammar(ABC):
    @abstractmethod
    def parse(self, grammar_string: str) -> Tuple:
        pass


class UAVGrammar(Grammar):
    def __init__(self):
        pass

    def parse(self, uav_string: str) -> Tuple[List[str], List[str], int, int]:
        """
        Function for breaking up a string of *valid* UAV Grammar into its four main elements.
        Adapted from Sebastiaan De Peuter's UAV string parsing code in 'UAV Design.ipynb'

        Returns a 4-tuple with:
            1st elem: List of component strings
            2nd elem: List of connection strings (2 chars)
            3rd elem: Integer specifying payload value
            4th elem: Integer specifying controller idx

        """

        s = uav_string.split(ELEM_SEP)

        try:
            # Extract elements
            design_string = s[0]
            payload = int(s[1])
        except IndexError:
            raise AssertionError("Missing elements/Incorrect element separators")
        except ValueError:
            raise AssertionError(f"Invalid string {s[1]} for payload.")

        # NOTE: If controller idx is invalid, it is replaced by 3 rather than discarding the design.
        try:
            controller_idx = int(s[2])
        except IndexError:
            controller_idx = 3
        except ValueError:
            controller_idx = 3

        # Split out the component connections (separator: ^), first element is component definitions
        connections = design_string.split(CONN_PREFIX)

        # Split out individual components (separator: *)
        components = connections[0].split(COMP_PREFIX)

        # Added this check due to some designs having garbage prefixed to them
        try:
            assert components[0] == "", f"Unexpected characters preceding first component '{components[0]}'"
        except AssertionError:
            print(f"Warning: Unexpected characters preceding first component '{components[0]}")
        components = components[1:]

        connections = connections[1:]

        return components, connections, payload, controller_idx

    def validate(self, uav_string: str) -> Tuple[Any, List[str]]:
        """
        Function for determining whether a string follows valid UAV design grammar.
        Returns a tuple, first element is a boolean signifying validity and second element is a list
        of possible errors encountered.

        """
        valid = True
        errors = []

        # Use the grammar's parser to get the elements
        try:
            components, connections, payload, controller_idx = self.parse(uav_string)
        except AssertionError as e:
            errors.append(f"Invalid UAV {uav_string};{str(e)}")
            return False, errors

        # Check payload argument
        try:
            assert (payload >= 0)
        except AssertionError:
            errors.append("Payload argument must be positive.")
            return False, errors

        # Check validity of components specified
        # TODO: Check that component identifiers are used in correct order
        id_to_loc = {}
        drone_components = []
        component_locs = np.zeros((len(Z_LETTER_COORDS), len(X_LETTER_COORDS)))
        for component in components:
            component_elems = list(component)

            # Check that the component string has correct amount of elements
            try:
                assert (len(component_elems) >= 4)
            except AssertionError:
                errors.append(f"Invalid component string '{component}', missing specifiers.")
                break

            component_id = component_elems[0]
            component_z = component_elems[2]
            component_x = component_elems[1]
            component_type = component_elems[3]
            component_size = []

            # Get the component size string, if it is specified
            if len(component) > 4:
                component_size = list(component[4:])

            try:
                component_type = int(component_type)
            except ValueError:
                errors.append(f"Component type for component id '{component_id}' is not a valid integer 0-4")
                break

            # Check the component ID
            try:
                assert (component_id in COMPONENT_IDS)
            except AssertionError:
                errors.append(f"Unknown component ID: {component_id}")
                break
            
            # Check for duplicates
            try:
                assert (component_id not in drone_components)
                drone_components.append(component_id)
            except AssertionError:
                errors.append(f"Duplicate component ID: {component_id} already defined")
                break

            # Check the component coordinate definition
            try:
                assert (component_x in X_LETTER_COORDS)
                assert (component_z in Z_LETTER_COORDS)
            except AssertionError:
                errors.append(f"Invalid coordinate letter for component id '{component_id}'.")
                break

            # Check coordinate validity w.r.t design string
            x, z = letter_to_integer_coord(component_x, axis="X"), letter_to_integer_coord(component_z, axis="Z")

            try:
                assert (component_locs[z, x] != 1)
                component_locs[z, x] = 1
                id_to_loc[component_id] = np.asarray([z, x])
            except AssertionError:
                errors.append(f"Duplicate coordinate definition for component id '{component_id}', "
                              f"component at {z},{x} already exists.")
                break

            # Check the component type
            try:
                assert (str(component_type) in COMPONENT_TYPE_IDS)
            except AssertionError:
                errors.append(f"Invalid component type specifier: {component_type} for component id '{component_id}'")
                break

            # Check the component size symbol validity
            size_symbol = None
            for symbol in component_size:
                try:
                    assert symbol in [INCREMENT_SYMBOL, DECREMENT_SYMBOL]
                except AssertionError:
                    errors.append(f"Invalid component size symbol encountered for component id '{component_id}'.")
                    break

                # Make sure all the size symbols are the same
                if size_symbol is None:
                    size_symbol = symbol
                else:
                    try:
                        assert size_symbol == symbol
                    except AssertionError:
                        errors.append(f"Mixture of size symbols encountered for component id '{component_id}'.")
                        break
        
        if len(errors) != 0:
            valid = False
            errors.insert(0, f'Invalid UAV "{uav_string}".')
            return valid, errors
        
        # NOTE: Apparently this is not a requirement either, strings with missing IDs pass the simulation
        # # Check that components used in correct order
        #
        # n_components = len(drone_components)
        # for idx, c in enumerate(COMPONENT_IDS[:n_components]):
        #     try:
        #         assert c in drone_components
        #     except AssertionError:
        #         errors.append(f"Missing component identifier: {c}")
        #         break
            
        # Check validity of connections specified
        for connection in connections:

            connection_elems = list(connection)
            # Check that connection is two chars
            try:
                assert (len(connection_elems) == 2)
            except AssertionError:
                errors.append(f"Invalid connection string '{connection}'")
                break

            orig = connection_elems[0]
            dest = connection_elems[1]

            # Check that connection id's are not the same
            # NOTE: Apparently self-connections are allowed (passes simulation)
            # try:
            #    assert orig != dest
            # except AssertionError:
            #    errors.append(f"Component connection to itself encountered ({orig}-{dest}).")
            #    break

            # Check that connection id's are in drone component list
            try:
                assert (orig in drone_components)
            except AssertionError:
                errors.append(f"Non-existent component ID '{orig}' encountered in connections")
                break

            try:
                assert(dest in drone_components)
            except AssertionError:
                errors.append(f"Non-existent component ID '{dest}' encountered in connections")
                break

            # NOTE: Not treating connections longer than 1 as errors! These exist in the pre-generated data.
            #try:
            #    orig_loc = id_to_loc[orig]
            #    dest_loc = id_to_loc[dest]

            #    dist = np.sum(np.sqrt((dest_loc - orig_loc) ** 2))

            #    assert dist <= 1.0, f"WARNING: Connection {connection} in {uav_string} length exceeds 1 ({dist})"
            #except AssertionError as e:
            #    print(e)

        # If no errors were found, uav string is valid
        if len(errors) != 0:
            valid = False
            errors.insert(0, f'Invalid UAV "{uav_string}".')

        return valid, errors


def letter_to_integer_coord(letter, axis):
    if axis == "X":
        return X_LETTER_COORDS.index(letter)
    elif axis == "Z":
        return Z_LETTER_COORDS.index(letter)
    else:
        print("Invalid axis")
        return None


def main():
    default_uav = "*aMM0+++++*bNM2+++*cMN1+++*dLM2+++*eML1+++^ab^ac^ad^ae,5,3"
    nn_uav = "*aMM0+++*bNM2+++*cMN1+++*dLM2+++*eML1+++*fLN0*gLL1*hNL0*iNN2^ab^ac^ad^ae^cf^dg^eh^bi,5,3"
    # strange_uav = "*aMM0+*bNM3+++*cMN3+++*dLM3+++*eML3+++*fOM1++++*gMO2++++*hKM1++++*iMK2++++*jNN3+*kLL3+^ab^ac^ad^ae^bf^cg^dh^ei^bj^dk,18,3,$6k, 50mi, 21mph, 18lbs"
    strange_uav = "*aMM2++*bNM0+---^ab,70,3"
    parser = UAVGrammar()

    import pandas as pd
    data_file = "data/datafiles/preprocessed/simresults_preprocessed_validunique.csv"
    uavs = list(pd.read_csv(data_file)["config"].values)

    for uav in uavs:
        parser.validate(uav)


if __name__ == '__main__':
    main()
