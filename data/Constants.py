"""
File defining useful constants for the UAV design problem

"""
import math
import json
import platform

ERROR = "error"
MSG_TYPES = [ERROR]

SAMPLING_TIMEOUT_S = 60

########################################################################################################################
# UAV DATA CONSTANTS
########################################################################################################################
CAPACITY_SCALING = 50
SIZE_SCALING = 20

UAV_STR_COL = 'config'

SOS_TOKEN = 'SOS'
EOS_TOKEN = 'EOS'
PAD_TOKEN = 'PAD'
NN_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]

DESIGN_SEP = '\n'
ELEM_SEP = ','
COMP_PREFIX = '*'
CONN_PREFIX = '^'
INCREMENT_SYMBOL = '+'
DECREMENT_SYMBOL = '-'

COMPONENT_INCREMENTS = [1, 3, 5]
COMPONENT_DECREMENTS = [-3]

# Constants for fixed increment payloads
PAYLOAD_VALUES = [0, 5, 10, 15, 20, 30, 50, 70, 90]

# Constants for incremental payload setting
PAYLOAD_STEPSIZE = 5
MAX_PAYLOAD = 100  # Pounds (lb)
MIN_PAYLOAD = 0

# The 42 valid node IDs
COMPONENT_IDS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', '@',
            '#', '$', '%', '&', '(', ')', '_', '=', '[', ']', '{', '}', '<', '>']

# Coordinates defined as per definition in:
# drone-testbed-local-evaluation\Assets\Projects\designtool\Scripts\UAVDesigner.cs
Z_LETTER_COORDS = ['J', 'K', 'L', 'M', 'N', 'O', 'P']
X_LETTER_COORDS = ['P', 'O', 'N', 'M', 'L', 'K', 'J']
COORD_GRID_SIZE = tuple([len(X_LETTER_COORDS), len(Z_LETTER_COORDS)])
X_HP = math.ceil(COORD_GRID_SIZE[0] / 2) - 1
Z_HP = math.ceil(COORD_GRID_SIZE[1] / 2) - 1

INTEGERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

COMPONENT_TYPE_IDS = ["0", "1", "2", "3", "4"]

COMPONENT_TYPE_DIC = {
    0: 'structure',
    1: 'cw_motor',
    2: 'ccw_motor',
    3: 'aerofoil',
    4: 'empty'
}

TOKEN_TO_IDX = {
   "PAD": 0,
   "SOS": 1,
   "EOS": 2,
   "!": 3,
   "#": 4,
   "$": 5,
   "%": 6,
   "&": 7,
   "(": 8,
   ")": 9,
   "*": 10,
   "+": 11,
   ",": 12,
   "-": 13,
   "0": 14,
   "1": 15,
   "2": 16,
   "3": 17,
   "4": 18,
   "5": 19,
   "6": 20,
   "7": 21,
   "8": 22,
   "9": 23,
   "<": 24,
   "=": 25,
   ">": 26,
   "@": 27,
   "J": 28,
   "K": 29,
   "L": 30,
   "M": 31,
   "N": 32,
   "O": 33,
   "P": 34,
   "[": 35,
   "]": 36,
   "^": 37,
   "_": 38,
   "a": 39,
   "b": 40,
   "c": 41,
   "d": 42,
   "e": 43,
   "f": 44,
   "g": 45,
   "h": 46,
   "i": 47,
   "j": 48,
   "k": 49,
   "l": 50,
   "m": 51,
   "n": 52,
   "o": 53,
   "p": 54,
   "q": 55,
   "r": 56,
   "s": 57,
   "t": 58,
   "u": 59,
   "v": 60,
   "w": 61,
   "x": 62,
   "y": 63,
   "z": 64,
   "{": 65,
   "}": 66
}

IDX_TO_TOKEN = {
   0: "PAD",
   1: "SOS",
   2: "EOS",
   3: "!",
   4: "#",
   5: "$",
   6: "%",
   7: "&",
   8: "(",
   9: ")",
   10: "*",
   11: "+",
   12: ",",
   13: "-",
   14: "0",
   15: "1",
   16: "2",
   17: "3",
   18: "4",
   19: "5",
   20: "6",
   21: "7",
   22: "8",
   23: "9",
   24: "<",
   25: "=",
   26: ">",
   27: "@",
   28: "J",
   29: "K",
   30: "L",
   31: "M",
   32: "N",
   33: "O",
   34: "P",
   35: "[",
   36: "]",
   37: "^",
   38: "_",
   39: "a",
   40: "b",
   41: "c",
   42: "d",
   43: "e",
   44: "f",
   45: "g",
   46: "h",
   47: "i",
   48: "j",
   49: "k",
   50: "l",
   51: "m",
   52: "n",
   53: "o",
   54: "p",
   55: "q",
   56: "r",
   57: "s",
   58: "t",
   59: "u",
   60: "v",
   61: "w",
   62: "x",
   63: "y",
   64: "z",
   65: "{",
   66: "}"
}

VOCAB_SIZE = len(TOKEN_TO_IDX)
PAD_VALUE = TOKEN_TO_IDX[PAD_TOKEN]
SOS_VALUE = TOKEN_TO_IDX[SOS_TOKEN]
EOS_VALUE = TOKEN_TO_IDX[EOS_TOKEN]


########################################################################################################################
# EXPERIMENT CONSTANTS
########################################################################################################################


########################################################################################################################
# OTHER CONSTANTS
########################################################################################################################

SIM_TIME_OUT = 5  # Time out for the simulator in seconds
if platform.system().lower() == "windows":
   SIM_EXE_PATH = './tools/drone-testbed-local-evaluation/Build/drone-testbed-unity.exe'
else:
   SIM_EXE_PATH = './tools/drone-testbed-local-evaluation/Build/drone-testbed-unity.x86_64'
   
DRONE_DB_PATH = 'data/db/drone.db'

UAV_CONFIG_COL = "config"
SIM_RESULT_COL = 'result'
RANGE_COL = 'range'
COST_COL = 'cost'
VELOCITY_COL = 'velocity'
PAYLOAD_COL = 'payload'
SIM_OUT_COLS = [UAV_CONFIG_COL, RANGE_COL, COST_COL, VELOCITY_COL, SIM_RESULT_COL]
SIM_METRICS = [RANGE_COL, COST_COL, VELOCITY_COL]
SIM_OUTCOME = SIM_RESULT_COL

# Simulator outcome classifications
SIM_SUCCESS = 'Success'
SIM_FAILURE = 'Failure'

# Simulator failure modes
SIM_FAIL_CRASH = 'SimulatorCrashed'
SIM_FAIL_CNS = 'CouldNotStabilize'
SIM_FAIL_HB = 'HitBoundary'
SIM_FAIL_ERROR = 'Error'
SIM_FAIL_TIMEOUT = 'Timeout'

# Numerical representations for classifications
SIM_OUTCOME_CODES = {
   SIM_FAIL_TIMEOUT: 0,
   SIM_FAIL_ERROR: 0,
   SIM_FAIL_CNS: 0,
   SIM_FAIL_HB: 0,
   SIM_FAIL_CRASH: 0,
   SIM_SUCCESS: 1
}

SIM_OUTCOME_KEYS = {
   0: "Failure",
   1: "Success"
}

SIM_OUTCOME_VALS = {
   "Failure": 0,
   "Success": 1
}

EVAL_METRICS = ['mean_absolute_error', 'mean_squared_error', 'accuracy_score']

NOOP_TOKEN = "NOOP"
DONE_TOKEN = "DONE"

# Magic numbers for normalizing the metrics
METRIC_NORM_FACTORS = {
   RANGE_COL: {
      "min": 0,
      "max": 85  # Farthest flying stable drone seen in data
   },
   COST_COL: {
      "min": 0,
      "max": 17_000  # Most expensive stable drone seen in data
   },
   VELOCITY_COL: {
      "min": 0,
      "max": 45  # Fastest stable drone seen in data
   }
}

OBJECTIVES = list(json.load(open("data/datafiles/objective_params.json")).keys())

EPS = 0.001

EXPERIMENT_MODES = ["train", "evaluate", "eval", "simulate", "trainpredict", "traineval", "kfold", "run_dqn",
                    "generate", "predict"]