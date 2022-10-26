import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax
from jax import numpy as jnp
import jraph
from typing import Any
import warnings
import numpy as np
import haiku as hk
import pandas as pd
from hyperoptimization_utils import run_grid_hyperparam_pipeline, train_single_model

##############################################
###### HYPERPARAMETERS SrTiO3 ################
# OVERWRITE = True
# #
# E_DIM = 24
# R_SWITCH = 0.1
# R_CUT = 10.0
# DISTANCE_ENCODING_TYPE = "none"
# FEATURES = [64,32,32,1]
# NUM_PASSES = 2
# ACTIVATION = "relu"
# N_EPOCHS = 50
# FORMULA = "SrTiO3"

##########################################
###### HYPERPARAMETERS IL ################
# OVERWRITE = False
# #
# E_DIM = 12
# R_SWITCH = 0.1
# R_CUT = 8.0
# DISTANCE_ENCODING_TYPE = "root"
# FEATURES = [8,4,2,1]
# NUM_PASSES = 2
# ACTIVATION = "relu"
# N_EPOCHS = 30
# FORMULA = "C30H120N30O45"

# DEFAULT_DICT = {
#     "E_DIM" : E_DIM,
#     "R_SWITCH" : R_SWITCH,
#     "R_CUT" : R_CUT,
#     "DISTANCE_ENCODING_TYPE" : DISTANCE_ENCODING_TYPE,
#     "FEATURES" : FEATURES,
#     "NUM_PASSES" : NUM_PASSES,
#     "ACTIVATION" : ACTIVATION,
#     "N_EPOCHS" : N_EPOCHS,
# }

# OPTIM_DICT = {
#     # "FEATURES":[[64,32,16,8,4,2,1]],
#     # "NUM_PASSES": [3]
#     # "R_SWITCH": [4.0],
#     # "R_CUT": [10.0,12.0]
# }

# model_results, batches = train_single_model(E_DIM = E_DIM,
#                                             R_SWITCH=R_SWITCH,
#                                             R_CUT=R_CUT,
#                                             DISTANCE_ENCODING_TYPE=DISTANCE_ENCODING_TYPE,
#                                             FEATURES=FEATURES,
#                                             NUM_PASSES=NUM_PASSES,
#                                             ACTIVATION=ACTIVATION,
#                                             N_EPOCHS=N_EPOCHS,
#                                             OVERWRITE = True,
#                                             FORMULA = FORMULA,
#                                             SAVE_MODEL = False,
#                                             SAMPLE_SIZE = 168)


# run_grid_hyperparam_pipeline(DEFAULT_DICT,OPTIM_DICT,OVERWRITE,FORMULA = FORMULA)
from infervis import visualize_results, get_visualization_testbatch, infer
# MODEL_PATH = "models/SrTiO3/(24, 0.1, 10.0, 'none', '[256, 64, 32, 8, 1]', 2, 'relu', 5000, 0.001, 1e-05).pkl"
# DB_PATH = "data/SrTiO3_500.db"
# FORMULA = "SrTiO3"


MODEL_PATH = "models/C30H120N30O45/(16, 0.1, 9.0, 'root', '[64, 32, 16, 8, 4, 2, 1]', 2, 'relu', 5000, 0.001, 1e-05).pkl"
SAVE_NAME = MODEL_PATH.split("/")[-1][:-4]
DB_PATH = "data/IL_charges.db"
FORMULA = "C30H120N30O45"
# visualize_results(MODEL_PATH, FORMULA = FORMULA, db_path = DB_PATH, xrange = [-1.5,2.1], save_name = None)
# print(infer(MODEL_PATH,DB_PATH,FORMULA))