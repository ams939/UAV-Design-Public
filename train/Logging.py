from abc import ABC, abstractmethod
import json
from time import localtime, strftime
from typing import List
import os

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb


class Logger(ABC):
    def __init__(self, mode=0, **kwargs):
        self.debug = bool(mode)

    @property
    def __name__(self):
        return self.__class__.__name__

    def log(self, event: dict):
        """
        Records event into log. Event expected to be dictionary.

        """

        if 'debug' in event.keys():
            if event['debug']:
                if not self.debug:
                    return

        self.record(event)

    @abstractmethod
    def record(self, data: dict):
        pass
    
    def close(self):
        """
        Call this once logging is done, performs clean up/buffer dump
        """
        pass


class ConsoleLogger(Logger):
    def __init__(self, mode=0, **kwargs):
        super(ConsoleLogger, self).__init__(mode, **kwargs)

    def record(self, data):
        if 'msg' in data.keys():
            if 'name' not in data.keys():
                data['name'] = "?"
                
            print(f"{strftime('%H:%M:%S', localtime())} {data['name']}: {data['msg']}")


class FileLogger(Logger):
    """ Writes the console log to a file log.txt"""
    def __init__(self, file_path, mode=0, write_freq=1000, **kwargs):
        super(FileLogger, self).__init__(mode, **kwargs)
        self.file_path = f"{file_path}/log.txt"
        self.buffer = []
        self.write_freq = write_freq

    def record(self, data):
        if 'msg' in data.keys():
            if "name" not in data.keys():
                data["name"] = ""
            
            msg_str = f"{strftime('%H:%M:%S', localtime())} {data['name']}: {data['msg']}\n"
            self.buffer.append(msg_str)
            
            if len(self.buffer) >= self.write_freq:
                self.dump_buffer()
        
    def dump_buffer(self):
        with open(self.file_path, 'a') as f:
            f.write("\n".join(self.buffer) + "\n")
        self.buffer = []
        
    def close(self):
        self.dump_buffer()


class WandbLogger(Logger):
    def __init__(self, project_name="uav-design", **kwargs):
        super(WandbLogger, self).__init__(**kwargs)
        wandb.init(project=project_name)

    def record(self, event: dict):
        if "data" in event.keys():
            wandb.log(event["data"])
            

class DatafileLogger(Logger):
    """ Writes data to file in JSON format, buffered output to control disk IO time """
    def __init__(self, file_path, mode=0, write_freq=1000, **kwargs):
        super(DatafileLogger, self).__init__(mode, **kwargs)
        self.buffer = []
        self.write_freq = write_freq

        self.file_path = f"{file_path}/datafile_log.json"

    def record(self, event: dict):
        if "data" in event.keys():
            data = event["data"]
            self.buffer.append(data)
            
            if len(self.buffer) >= self.write_freq:
                self.dump_buffer()
                
    def dump_buffer(self):
        # If file already exists, prepend separator
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        
        data += self.buffer
    
        with open(self.file_path, 'w') as f:
            json.dump(data, f)
            
        self.buffer = []
        
    def close(self):
        self.dump_buffer()
                
                
class PlotLogger(Logger):
    """
        Logger which iteratively creates a plot of data passed to it via 'record'. Expects 'event' dict to contain field
        'plot', which in turn is a dict with keys 'x' and 'y'.
    """
    def __init__(self, file_path, mode=0, **kwargs):
        super(PlotLogger, self).__init__(mode, **kwargs)
        
        self.file_path = file_path
        self.x_label = kwargs["x_label"]
        self.y_label = kwargs["y_label"]
        self.title = kwargs["title"]
        
        self.data = None
        
    def record(self, event: dict):
        if "plot" in event.keys():
            plot_data = pd.DataFrame([event["plot"]])
            if self.data is None:
                self.data = plot_data
            else:
                try:
                    self.data = pd.concat([self.data, plot_data], ignore_index=True)
                except Exception as e:
                    print(f"Couldn't append plotting data:\n{e}")
                    
            plt.scatter(self.data['x'], self.data['y'])
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            plt.title(self.title)
            
            plt.tight_layout()
            plt.savefig(f"{self.file_path}/plot.png")
            self.data.to_csv(f"{self.file_path}/plot_data.csv", index=False)


class DummyLogger(Logger):
    """ Does nothing, placeholder class for testing purposes"""
    def __init__(self, mode=0, **kwargs):
        super(DummyLogger, self).__init__(mode, **kwargs)

    def record(self, data):
        pass


class MultiLogger(Logger):
    """ Wrapper class for a list of Logger objects. Passes 'event' to each one """
    def __init__(self, loggers: List[Logger],  mode=0, **kwargs):
        super(MultiLogger, self).__init__(mode, **kwargs)
        self.debug = True
        self.logger_list = loggers

    def record(self, event: dict):
        for logger in self.logger_list:
            logger.log(event)
            
    def close(self):
        for logger in self.logger_list:
            logger.close()


def init_logger(hyperparams) -> Logger:
    loggers = hyperparams.loggers
    file_dir = hyperparams.experiment_folder

    if len(loggers) >= 1:
        logger_list = []
        for logger in loggers:
            logger_class = logger["logger_class"]
            logger_hparams = logger["logger_hparams"]
            logger_hparams["file_path"] = f"{file_dir}"

            logger = logger_class(**logger_hparams)
            logger_list.append(logger)
        return MultiLogger(logger_list)

    elif len(loggers) == 1:
        logger_class = loggers[0]["logger_class"]
        logger_hparams = loggers[0]["logger_hparams"]
        logger_hparams["file_path"] = f"{file_dir}"

        return logger_class(**logger_hparams)

    else:
        return DummyLogger()
