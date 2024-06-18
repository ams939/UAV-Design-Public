from typing import Any, List
from abc import ABC, abstractmethod

import numpy as np

from train.Logging import Logger, DummyLogger
from train.Hyperparams import Hyperparams


class TrainStopper(ABC):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.save_flag = False

    def stop(self) -> Any:
        """ Returns true or false """
        return self.stop_condition()

    @property
    def __name__(self):
        return self.__class__.__name__

    @abstractmethod
    def step(self, epoch, loss) -> None:
        """ Update inner state of the stopper """
        pass

    @abstractmethod
    def stop_condition(self) -> Any:
        """ Check if the inner state satisfies stopping condition """
        pass

    def save_model(self) -> Any:
        """ Method for checking if stopper recommends saving model """
        return self.save_flag


class EpochStopper(TrainStopper):
    """
    Simple class for checking if we've reached a fixed epoch number
    """
    def __init__(self, logger: Logger, stop_epoch: int):
        self.curr_epoch = 0
        self.stop_epoch = stop_epoch
        super(EpochStopper, self).__init__(logger)

        logger.log({"name": self.__name__, "msg": f"Initializing with stopping condition epoch > {stop_epoch}"})

    def stop_condition(self) -> Any:
        stop = self.curr_epoch >= self.stop_epoch

        if stop:
            self.logger.log({"name": self.__name__, "msg": "Stopping condition met.", "debug": 0})

        return stop

    def step(self, epoch, loss):
        self.logger.log({"name": self.__name__, "msg": "Stopper iterating", "debug": 1})
        self.curr_epoch = epoch


class PlateauStopper(TrainStopper):
    def __init__(self, logger: Logger, patience=10, start_epoch=0):
        self.curr_epoch = 0
        self.curr_patience = 0
        self.patience = patience
        self.active = False
        self.start_epoch = start_epoch
        self.best_loss = np.inf
        super(PlateauStopper, self).__init__(logger)

        self.logger.log({"name": self.__name__, "msg": f"Initializing with stopping condition: Loss does not decrease" +
                        f" for {patience} epochs"})

    def stop_condition(self) -> Any:
        # Stop if loss delta has not exceeded threshold in patience number of epochs
        if self.curr_patience > self.patience:
            cond = True
            self.logger.log({"name": self.__name__, "msg": "Stopping condition met."})

        # Continue if stop condition not met
        else:
            cond = False

        return cond

    def step(self, epoch, loss):
        self.curr_epoch = epoch

        if not self.active and self.curr_epoch > self.start_epoch:
            self.active = True

        if not self.active:
            return

        if loss > self.best_loss:
            self.curr_patience += 1
            self.logger.log({"name": self.__name__, "msg": f"Loss plateauing "
                                                           f"(Patience={self.curr_patience}/{self.patience})."})
        else:
            self.curr_patience = 0
            self.best_loss = loss

        # Update the save flag
        if self.curr_patience == 1:
            self.save_flag = True
        elif self.save_flag:
            self.save_flag = False



class DeltaStopper(TrainStopper):
    def __init__(self, logger: Logger, threshold=0.001, patience=10):
        self.curr_epoch = 0
        self.curr_patience = 0
        self.patience = patience
        self.threshold = threshold
        self.prev_loss = None
        super(DeltaStopper, self).__init__(logger)

        self.logger.log({"name": self.__name__, "msg": f"Initializing with stopping condition loss delta < {threshold}" +
                    f" for {patience} epochs"})

    def stop_condition(self) -> Any:
        # Stop if loss delta has not exceeded threshold in patience number of epochs
        if self.curr_patience > self.patience:
            cond = True
            self.logger.log({"name": self.__name__, "msg": "Stopping condition met."})

        # Continue if stop condition not met
        else:
            cond = False

        return cond

    def step(self, epoch, loss):
        self.curr_epoch = epoch

        if self.prev_loss is not None:
            loss_delta = abs(abs(loss) - abs(self.prev_loss))
        else:
            loss_delta = 99

        if loss_delta < self.threshold:
            self.curr_patience += 1
            self.logger.log({"name": self.__name__, "msg": f"Loss plateau (Patience={self.curr_patience}/{self.patience})."})
        else:
            self.curr_patience = 0

        self.prev_loss = loss


class MultiStopper(TrainStopper):
    def __init__(self, logger, stopper_list: List[TrainStopper]):
        super(MultiStopper, self).__init__(logger)
        self.stopper_list = stopper_list

    def stop_condition(self):
        stop_cond = False
        for stopper in self.stopper_list:
            if stopper.stop_condition():
                stop_cond = True
                break

        return stop_cond

    def step(self, *args, **kwargs):
        self.save_flag = False
        for stopper in self.stopper_list:
            stopper.step(*args, **kwargs)
            self.save_flag = stopper.save_flag or self.save_flag


def init_stopper(hparams: Hyperparams) -> TrainStopper:
    stoppers = hparams.stoppers
    logger = hparams.logger
    
    assert stoppers is not None, "No stopper class provided, please specify in hparams"

    if len(stoppers) >= 1:
        # Initialize list of TrainStopper objects from class name strings and hparam dicts
        stopper_list = [stopper["stopper_class"](logger, **stopper["stopper_hparams"])
                        for stopper in stoppers]
        return MultiStopper(logger, stopper_list)
    elif len(stoppers) == 1:
        # Initialize a singular stopper from class string
        stopper_class = stoppers[0]["stopper_class"]
        return stopper_class(logger, **stoppers[0]["stopper_hparams"])

    else:
        # Default case
        return EpochStopper(DummyLogger(), stop_epoch=50)
