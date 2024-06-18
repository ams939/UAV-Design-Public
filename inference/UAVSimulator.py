from abc import abstractmethod
import concurrent.futures as cf
from typing import List

from data.Constants import SIM_OUT_COLS
from tools.testbed_interface import uav_simulate as hyform_simulate


class UAVSimulator:
    def __init__(self, debug=False):
        self.debug = debug
        self._simulator = None
        self._init_simulator()

    @property
    def __name__(self):
        return self.__class__.__name__

    @abstractmethod
    def _init_simulator(self):
        pass

    @abstractmethod
    def simulate_batch(self, uav_str_list: List[str]):
        pass

    @abstractmethod
    def simulate(self, uav_str: str):
        pass


class HyFormSimulator(UAVSimulator):
    """
    BEWARE:

    This class incorporates multiprocessing, and hence seems to not like to play nice with some modules. Thus,
    if certain imports are imported BEFORE this is used, you will get a nasty and highly obscure error message
    regarding a "Broken Process Pool".

    """

    def __init__(self, *args, n_workers=6, debug=False):
        self.n_workers = n_workers
        self.simulator_process = None
        self.sim_futures = {}
        self.data_cols = SIM_OUT_COLS
        super(HyFormSimulator, self).__init__(debug)

    def start(self):
        self.simulator_process = cf.ProcessPoolExecutor(max_workers=self.n_workers)

    def stop(self):
        self.simulator_process.shutdown()
        self.simulator_process = None

    def _init_simulator(self):
        self._simulator = hyform_simulate

    def _submit_simulation(self, uav_str: str, idx: int):
        sim_future = self.simulator_process.submit(self._simulator, uav_str, self.debug)
        self.sim_futures[idx] = sim_future

    def simulate_batch(self, uav_str_list: List[str], return_df=False, preserve_order=True):
        """
        Note, yields results for batch strings in different order than inputs if preserve_order not set to True.
        Otherwise, simulation results returned in the order that they complete.
        """
        self.start()

        for idx, uav_str in enumerate(uav_str_list):
            self._submit_simulation(uav_str, idx)

        if not preserve_order:
            for sim_future in cf.as_completed(list(self.sim_futures.values())):
                yield sim_future.result()
        else:
            for sim_idx in range(len(self.sim_futures)):
                yield self.sim_futures[sim_idx].result()

        self.stop()

    def simulate(self, uav_str: str, return_df=False):
        return self.simulate_batch([uav_str]).__next__()


if __name__ == "__main__":
    sim = HyFormSimulator()
    # uavsd = open(r"C:\Users\Aleksanteri\Documents\School\Work\Summer2022\UAV-Design\experiments\DQN\103022012040\successful_stable.csv").readlines()[1:]
    uavsd = ["*aMM0*bNM2+*cMN1+*dLM2+*eML1+*gNN4*hNL3*iMO4*jLN3*kKM4*lLL4^ab^ac^ad^ae^bg^bh^ci^cj^dk^dl^cg^dj^eh^el,10,1"]
    uavs = [uav.rstrip() for uav in uavsd]
    for res in sim.simulate_batch(uavs):
        print(res)
