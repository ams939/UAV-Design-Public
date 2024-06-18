import json
import sys

from utils.utils import ddict, json_bool_parser, json_class_parser, hparams_serialize
from train.Logging import ConsoleLogger


class Hyperparams(ddict):
    """
    The core class for defining UAV-Design experiment parameters. Purpose is to convert the hyperparameters JSON file
    into an 'extended' dictionary object. Key features include allowing element access via dot notation and converting
    class names into class types stored by the Hyperparameters object. Also facilitates saving the hyperparams.
    """

    def __init__(self, hparam_file: str):
        self.hparam_file = hparam_file

        # Initializes hparams
        hparams = self.load()

        super(Hyperparams, self).__init__(hparams)
        
        self.logger = ConsoleLogger()
        
    def to_string(self):
        return json.dumps(hparams_serialize(self), indent=4)

    @property
    def __name__(self):
        return self.__class__.__name__

    def parse(self, params: dict):
        # Convert class name strings into class type
        hparams = json_class_parser(params)

        # Convert booleans in string form
        hparams = json_bool_parser(hparams)

        return hparams

    def load(self):
        try:
            with open(self.hparam_file, "r") as f:
                hparams = json.load(f)
        except FileNotFoundError:
            print(f"Couldn't find the hyperparams file {self.hparam_file}")
            sys.exit(-1)

        except json.JSONDecodeError as e:
            print(f"Hyperparams not valid JSON {e}")
            sys.exit(-1)

        hparams = self.parse(hparams)

        return hparams

    def save(self, path: str):
        logger = self.logger

        hparams = hparams_serialize(self)

        with open(path, "w") as f:
            json.dump(hparams, f, indent=4)

        logger.log({"name": self.__name__, "msg": f"Copy of hyperparams file saved to {path}"})
        
    def to_wandb(self):
        import wandb
        hparams = hparams_serialize(self)
        
        for k, v in hparams.items():
            if isinstance(v, dict):
                continue
            wandb.config[k] = v


class DummyHyperparams(Hyperparams):
    """
    Class for using the Hyperparams object without needing to load things from file. Instead, file contents
    specified directly as 'params' argument.

    """
    def __init__(self, params=None):
        if params is None:
            params = {}
        else:
            params = self.parse(params)
        super(Hyperparams, self).__init__(params)


if __name__ == "__main__":
    res = json_bool_parser(json.load(open("hparams/simrnn_hparams.json")))
    res = json_class_parser(res)
    res = hparams_serialize(res)
    print(res)
