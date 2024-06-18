"""
Script for interfacing with the HyForm simulator

"""

import subprocess
import multiprocessing
from typing import Tuple, List

from data.Constants import SIM_EXE_PATH, SIM_TIME_OUT, SIM_RESULT_COL, VELOCITY_COL, COST_COL, RANGE_COL, UAV_CONFIG_COL


def simulate_uav(uav_string: str) -> Tuple[float, float, float]:
    """
    Function for running the Unity UAV simulation, from Sebastiaan's UAV Design notebook
    """
    subprocess.run([SIM_EXE_PATH, "-batchmode", "-nographics", "-configuration", uav_string],
                   check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=None)

    with open("results.txt") as f:
        results_text = f.readlines()
    results = results_text[0].split(";")
    range_mi = float(results[1])
    cost_usd = float(results[3])
    velocity_mph = float(results[4])

    return range_mi, cost_usd, velocity_mph


def uav_simulate(uav_str: str, debug=False) -> dict:

    # Template for output
    sim_result_dic = {
        UAV_CONFIG_COL: uav_str,
        SIM_RESULT_COL: "",
        VELOCITY_COL: 0,
        RANGE_COL: 0,
        COST_COL: 0
    }

    try:
        out = subprocess.check_output([SIM_EXE_PATH, "-nofile", "-batchmode", "-nographics", "-configuration", uav_str],
                                      timeout=SIM_TIME_OUT, stderr=None, text=True)
    except subprocess.TimeoutExpired:
        if debug:
            print("Simulator timed out.")
        sim_result_dic[SIM_RESULT_COL] = "Timeout"
        return sim_result_dic
    except subprocess.SubprocessError:
        print("Simulator crashed!")
        sim_result_dic[SIM_RESULT_COL] = "SimulatorCrashed"
        return sim_result_dic

    out_list = out.split("\n")

    try:
        results_row_idx = out_list.index("RESULTS") + 1
    except ValueError:

        if debug:
            msg = "No row with string 'RESULTS' found from simulator output"
            print(msg)
            sim_result_dic["debug"] = msg

        sim_result_dic[SIM_RESULT_COL] = "Error"

        return sim_result_dic

    results_row = out_list[results_row_idx]
    results_row = results_row.split(" ")

    try:
        result, range_mi, velocity, cost = results_row
    except ValueError:
        if debug:
            msg = f"Unexpected out-row format: {' '.join(results_row)}. Result not parsed."
            print(msg)

        sim_result_dic[SIM_RESULT_COL] = "Error"

        return sim_result_dic

    try:
        velocity = float(velocity)
    except ValueError:
        if debug:
            print(f"Invalid velocity value: {velocity}. Couldn't convert to float.")
            velocity = f'"{velocity}"'

    try:
        range_mi = float(range_mi)
    except ValueError:
        if debug:
            print(f"Invalid range value: {range_mi}. Couldn't convert to float.")
            range_mi = f'"{range_mi}"'

    try:
        cost = float(cost)
    except ValueError:
        if debug:
            print(f"Invalid cost value: {cost}. Couldn't convert to float.")
            cost = f'"{cost}"'

    # Assign result to the output dictionary
    sim_result_dic[SIM_RESULT_COL] = result
    sim_result_dic[RANGE_COL] = range_mi
    sim_result_dic[VELOCITY_COL] = velocity
    sim_result_dic[COST_COL] = cost

    return sim_result_dic


def uav_batch_simulate(uav_strs: List[str], n_cores=None):
    # Passing None to number of processes uses all cores available
    pool = multiprocessing.Pool(n_cores)

    sim_results = pool.map_async(uav_simulate, uav_strs)
    sim_results = sim_results.get()
    pool.close()
    return sim_results


if __name__ == "__main__":
    pass
