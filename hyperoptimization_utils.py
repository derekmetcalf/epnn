from preprocessing_jraph import get_init_crystal_states, get_init_crystal_states_for_inference
from pipeline_utils import create_implicitly_batched_graphsTuple_with_encoded_distances, create_model
from jax import random, numpy as jnp
import optax
import jraph
import warnings
import haiku as hk
import jax
from typing import Any, Tuple
import time
import numpy as np
import pandas as pd
import pickle
import dill

def run_model_with_hyperparameters(E_DIM : int,
                                  R_SWITCH : float,
                                  R_CUT : float,
                                  DISTANCE_ENCODING_TYPE : str,
                                  FEATURES : list,
                                  NUM_PASSES : int,
                                  ACTIVATION : str,
                                  N_EPOCHS : int,
                                  INITS : tuple,
                                  FORMULA = "SrTiO3",
                                  LR = 1e-03,
                                  WEIGHT_DECAY = 1e-04
                                  ):
    """Run the model training with selected hyperparameters and initial data.
    Input: 
        -   E_DIM [int]: Number of embedding dimensions for the edges
        -	R_SWITCH [float]: First parameter of the cutoff function. A lower value leads to a less steep slope.
        -	R_CUT [float]: Second parameter of the cutoff function. The cutoff function approaches zero close to this value.
        -	DISTANCE_ENCODING_TYPE [“none”, “root”, “log”, “gaussian”]: You can encode the distances between atoms with the following distance encoding types before they are embedded.
            -	“none”: no distance encoding before embedding the distances
            -	“root”: square root of the distances before embedding the distances.
            -	“log1”: logarithmic value of (distance+1) [to avoid negative values]
        -	FEATURES [array]: An array of dimensions for the underlying neural network. It needs to end with 1, but you can increase the complexity to improve model performance. 
            e.g. [128,64,32,16,8,1]
        -	NUM_PASSES [int]: number of passing steps for the message-passing algorithm applied to the graph. 
        -	ACTIVATION [“relu”,”switch”] Two different activation functions to choose from.
        -	N_EPOCHS [int]: The number of training epochs the model trains.
        -   INITS [tuple] : Tuple of the initial data fed into the model. (see train_single_model() for data)
        -   FORMULA: [str] Formula of the chemical compound
        -	LR [float]: The initial learning rate for the AdamW-optimizer. It will have an effect on the weight decay (which is proportional to the learning rate in the library.)
        -	WEIGHT_DECAY [float]: The weight decay after each epoch to avoid overfitting.

    Output:
        - model_results: dict -> dictionary with "model" and "params"
        - performance_results: dict -> dictionary with all performance results like
                    "time_taken" (how long it took to run the pipeline in minutes)
                    "test_rmse" (test set rmse)
                    "test_mae" (test set mae)
                    "step_list" (list of epoch steps with performance results)
                    "rmse_list_val" (list of validation rmses for each epoch)
                    "best_val_rmse" (best rmse for validation set and returned model)
        - batches: dict -> contains the batches that the model was trained on
                    "train_batches"
                    "val_batch"
                    "test_batch"
                    "train_idxs"
                    "val_idx"
                    "test_idx"
                    "true_labels_train"
                    "true_labels_val"
                    "true_labels_test"
    """
    CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    key, subkey = random.split(random.PRNGKey(0))
    descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types = INITS
    #######################################
    ### Creating batches for training #####
    total_size = descriptors.shape[0]
    if FORMULA == "SrTiO3":
        train_batch_size = 50
        test_size = 50
        val_size = test_size
    else:
        train_batch_size = 25
        test_size = 34
        val_size = test_size
    train_size = total_size - test_size - val_size
    key = random.PRNGKey(0)
    permuted_idx = random.permutation(key, total_size)
    test_idx = permuted_idx[-test_size:]
    val_idx = permuted_idx[-2*test_size:-test_size]
    train_idxs = jnp.array(jnp.split(permuted_idx[:-2*test_size],int(jnp.ceil(train_size/train_batch_size))))
    train_batches = [create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[train_idx],distances[train_idx], distances_encoded[train_idx],init_charges[train_idx], types[train_idx], cutoff_mask[train_idx], R_CUT) for train_idx in train_idxs]
    val_batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[val_idx],distances[val_idx], distances_encoded[val_idx],init_charges[val_idx],types[val_idx],cutoff_mask[val_idx], R_CUT)
    test_batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[test_idx],distances[test_idx], distances_encoded[test_idx],init_charges[test_idx],types[test_idx],cutoff_mask[test_idx], R_CUT)


    # create the model
    gep_layer = create_model(FEATURES,ACTIVATION)
    ################################
    model = hk.without_apply_rng(hk.transform(gep_layer))
    params = model.init(jax.random.PRNGKey(42), train_batches[0])

    # extract the true labels
    true_labels = [gt_charges[train_idx].flatten() for train_idx in train_idxs]
    true_labels_val = gt_charges[val_idx].flatten()
    true_labels_test = gt_charges[test_idx].flatten()

    # initialize optimizer
    opt_init, opt_update = optax.adamw(learning_rate = LR,weight_decay = WEIGHT_DECAY)
    opt_state = opt_init(params)

    batches = {
        "train_batches":train_batches,
        "val_batch":val_batch,
        "test_batch":test_batch,
        "train_idxs":train_idxs,
        "val_idx":val_idx,
        "test_idx":test_idx,
        "true_labels_train":true_labels,
        "true_labels_val":true_labels_val,
        "true_labels_test":true_labels_test,
    }

    # Create loss functions with correct NUM_PASSES and model
    @jax.jit
    def rmse_loss(params: hk.Params, graph: jraph.GraphsTuple,  ground_truth: jnp.array) -> jnp.ndarray:
        # hk.fori_loop(0,3, model.apply, graph, params=params)
        output = model.apply(params, graph)
        for i in range(NUM_PASSES-1):
          output = model.apply(params, output)
        # return jnp.sqrt(jnp.sum(jnp.square(output[0]-ground_truth)/len(ground_truth)))
        return jnp.sqrt(jnp.sum(jnp.square(output[0]-ground_truth)/len(ground_truth)))

    @jax.jit
    def mae_loss(params: hk.Params, graph: jraph.GraphsTuple,  ground_truth: jnp.array) -> jnp.ndarray:
        # hk.fori_loop(0,3, model.apply, graph, params=params)
        output = model.apply(params, graph)
        for i in range(NUM_PASSES-1):
          output = model.apply(params, output)
        return jnp.sum(jnp.abs(output[0]-ground_truth)/len(ground_truth))

    @jax.jit
    def update(train_batch: jraph.GraphsTuple, true_labels:jnp.array, params: hk.Params, opt_state) -> Tuple[hk.Params, Any]:
        """Returns updated params and state."""
        g = jax.grad(rmse_loss)(params, train_batch, true_labels)
        updates, opt_state = opt_update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    step_list = []
    rmse_list_val = []
    #################################################
    #################################################
    print("Start training.")
    print("Current Parameter-Setting:",CURRENT_INDEX)
    ct = time.time()
    best_val_rmse = np.inf
    best_model = None

    # run model training for number of epochs
    for i in range(N_EPOCHS):
        rmse_batch_train_losses = []
        for batch_no in range(len(train_batches)):
            params, opt_state = update(train_batches[batch_no],true_labels[batch_no], params, opt_state)
        for batch_no in range(len(train_batches)):
            rmse_batch_train_losses.append(rmse_loss(params, train_batches[batch_no],true_labels[batch_no]))
        val_rmse = float(rmse_loss(params, val_batch, true_labels_val))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
        if (i % 10)==0 or i in range(10):
            step_list.append(i)
            rmse_list_val.append(val_rmse)
            print("Epoch:",i,"-  Train-RMSE:", round(float(sum(rmse_batch_train_losses)/len(rmse_batch_train_losses)),5)," -  Val-RMSE:", round(val_rmse,5))
    time_taken = round((time.time()-ct)/60,2)
    test_rmse = rmse_loss(params, test_batch, true_labels_test)
    test_mae = mae_loss(params, test_batch, true_labels_test)

    performance_results = {
        "time_taken":time_taken,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "step_list": step_list,
        "rmse_list_val": rmse_list_val,
        "best_val_rmse": best_val_rmse,
    }
    model_results = {
        "model": best_model,
        "params": params
    }

    return model_results, performance_results, batches


def run_grid_hyperparam_pipeline(DEFAULT_DICT : dict,
                            OPTIM_DICT: dict,
                            OVERWRITE: bool,
                            FORMULA = "SrTiO3"):
    """Run a hyperparameter optimization pipeline with different settings to test which settings are suitable for predicting atomic charges in your chemical structure.
        Change the DEFAULT_DICT to change the default parameters for the model. Change the OPTIM_DICT to change parameters to be tested during hyperparameter optimization.
        Input:
            - DEFAULT_DICT: Dictionary with all default values in it for the following parameters 
                -   E_DIM [int]: Number of embedding dimensions for the edges
                -	R_SWITCH [float]: First parameter of the cutoff function. A lower value leads to a less steep slope.
                -	R_CUT [float]: Second parameter of the cutoff function. The cutoff function approaches zero close to this value.
                -	DISTANCE_ENCODING_TYPE [“none”, “root”, “log”, “gaussian”]: You can encode the distances between atoms with the following distance encoding types before they are embedded.
                    -	“none”: no distance encoding before embedding the distances
                    -	“root”: square root of the distances before embedding the distances.
                    -	“log1”: logarithmic value of (distance+1) [to avoid negative values]
                -	FEATURES [array]: An array of dimensions for the underlying neural network. It needs to end with 1, but you can increase the complexity to improve model performance. 
                    e.g. [128,64,32,16,8,1]
                -	NUM_PASSES [int]: number of passing steps for the message-passing algorithm applied to the graph. 
                -	ACTIVATION [“relu”,”switch”] Two different activation functions to choose from.
                -	N_EPOCHS [int]: The number of training epochs the model trains.
                -	LR [float]: The initial learning rate for the AdamW-optimizer. It will have an effect on the weight decay (which is proportional to the learning rate in the library.)
                -	WEIGHT_DECAY [float]: The weight decay after each epoch to avoid overfitting.
            - OPTIM_DICT: Dictionary with arrays of input parameters to run hyperparameter optimization
            - OVERWRITE: [bool] -> Flag to decide if the model should run a pipeline again that is already saved in the results. (False means no overwriting of previous results)
            - FORMULA: [str] Formula of the chemical compound

        Output: None (but results are saved in result list in /results)
        """
    all_params = ["E_DIM","R_SWITCH","R_CUT","DISTANCE_ENCODING_TYPE","FEATURES","NUM_PASSES","ACTIVATION","N_EPOCHS","LR","WEIGHT_DECAY"] 
    for key in OPTIM_DICT.keys():
            assert key in DEFAULT_DICT.keys()
    for param_name in all_params:
        DEFAULT_DICT[param_name] = [DEFAULT_DICT[param_name]]
    DEFAULT_DICT.update(OPTIM_DICT)
    print(DEFAULT_DICT)

    
    # Run through all hyperparameter configurations
    for E_DIM in DEFAULT_DICT["E_DIM"]:
        for R_SWITCH in DEFAULT_DICT["R_SWITCH"]:
            for R_CUT in DEFAULT_DICT["R_CUT"]:
                for DISTANCE_ENCODING_TYPE in DEFAULT_DICT["DISTANCE_ENCODING_TYPE"]:
                    # check if planned indices are already in the results
                    check_if_available = False
                    result_table = pd.read_csv(f"results/result_table_{FORMULA}.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd"])
                    for FEATURES in DEFAULT_DICT["FEATURES"]:
                        for NUM_PASSES in DEFAULT_DICT["NUM_PASSES"]:
                            for ACTIVATION in DEFAULT_DICT["ACTIVATION"]:
                                for N_EPOCHS in DEFAULT_DICT["N_EPOCHS"]:
                                    for LR in DEFAULT_DICT["LR"]:
                                        for WEIGHT_DECAY in DEFAULT_DICT["WEIGHT_DECAY"]:
                                            CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY)
                                            if (not OVERWRITE) and result_table.index.isin([CURRENT_INDEX]).any():
                                                    print(f"NO OVERWRITE: Results already in Dataframe for Index {CURRENT_INDEX}.")
                                                    check_if_available = True
                    if check_if_available:
                        continue
                    else:
                        # only run preprocessing if results are not available for this combination of hyperparameters
                        inits = get_init_crystal_states(edge_encoding_dim = E_DIM, SAMPLE_SIZE = None, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = FORMULA)
                        for FEATURES in DEFAULT_DICT["FEATURES"]:
                            for NUM_PASSES in DEFAULT_DICT["NUM_PASSES"]:
                                for ACTIVATION in DEFAULT_DICT["ACTIVATION"]:
                                    for N_EPOCHS in DEFAULT_DICT["N_EPOCHS"]:
                                        for LR in DEFAULT_DICT["LR"]:
                                            for WEIGHT_DECAY in DEFAULT_DICT["WEIGHT_DECAY"]:
                                                CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY)
                                                result_table = pd.read_csv(f"results/result_table_{FORMULA}.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd"])

                                                if (not OVERWRITE) and result_table.index.isin([CURRENT_INDEX]).any():
                                                    print(f"NO OVERWRITE: Results already in Dataframe for Index {CURRENT_INDEX}.")
                                                else:
                                                    types = inits[-1] 
                                                    # Change sample size to None if all samples should be read.
                                                    model_results, performance_results, batches = run_model_with_hyperparameters(E_DIM = E_DIM,
                                                                                                                                R_SWITCH=R_SWITCH,
                                                                                                                                R_CUT=R_CUT,
                                                                                                                                DISTANCE_ENCODING_TYPE=DISTANCE_ENCODING_TYPE,
                                                                                                                                FEATURES=FEATURES,
                                                                                                                                NUM_PASSES=NUM_PASSES,
                                                                                                                                ACTIVATION=ACTIVATION,
                                                                                                                                N_EPOCHS=N_EPOCHS,
                                                                                                                                INITS=inits,
                                                                                                                                FORMULA = FORMULA,
                                                                                                                                LR = LR,
                                                                                                                                WEIGHT_DECAY = WEIGHT_DECAY
                                                                                                                                )
                                                    batches["types"] = types
                                                    if result_table.index.isin([CURRENT_INDEX]).any():
                                                        result_table.loc[CURRENT_INDEX] = performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]
                                                    else:
                                                        result_table = result_table.reset_index().append(dict(zip(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd","time_needed","test_rmse","test_mae","steps","val_rmses", "best_val_rmse"],[E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]])), ignore_index = True).set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd"])
                                                    print("Best-Val-RMSE:",performance_results["best_val_rmse"])
                                                    result_table.to_csv(f"results/result_table_{FORMULA}.csv")
    


def train_single_model(E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, OVERWRITE = True, FORMULA = "SrTiO3", SAVE_MODEL = False, SAMPLE_SIZE = None, LR = 1e-3,WEIGHT_DECAY = 1e-4):
    """Train a single model with selected hyperparameters.
    Input: 
        -   E_DIM [int]: Number of embedding dimensions for the edges
        -	R_SWITCH [float]: First parameter of the cutoff function. A lower value leads to a less steep slope.
        -	R_CUT [float]: Second parameter of the cutoff function. The cutoff function approaches zero close to this value.
        -	DISTANCE_ENCODING_TYPE [“none”, “root”, “log”, “gaussian”]: You can encode the distances between atoms with the following distance encoding types before they are embedded.
            -	“none”: no distance encoding before embedding the distances
            -	“root”: square root of the distances before embedding the distances.
            -	“log1”: logarithmic value of (distance+1) [to avoid negative values]
        -	FEATURES [array]: An array of dimensions for the underlying neural network. It needs to end with 1, but you can increase the complexity to improve model performance. 
            e.g. [128,64,32,16,8,1]
        -	NUM_PASSES [int]: number of passing steps for the message-passing algorithm applied to the graph. 
        -	ACTIVATION [“relu”,”switch”] Two different activation functions to choose from.
        -	N_EPOCHS [int]: The number of training epochs the model trains.
        -   OVERWRITE: [bool] -> Flag to decide if the model should run a pipeline again that is already saved in the results. (False means no overwriting of previous results)
        -   FORMULA: [str] Formula of the chemical compound
        -   SAVE_MODEL: [bool] -> Should the model be saved after training or not?
        -   SAMPLE_SIZE: [int] -> How many samples from the database shall be used for training, "None" equals all data.
        -	LR [float]: The initial learning rate for the AdamW-optimizer. It will have an effect on the weight decay (which is proportional to the learning rate in the library.)
        -	WEIGHT_DECAY [float]: The weight decay after each epoch to avoid overfitting.

    Output:
        - model_results: dict -> dictionary with "model" and "params"
        - batches: dict -> contains the batches that the model was trained on
                    "train_batches"
                    "val_batch"
                    "test_batch"
                    "train_idxs"
                    "val_idx"
                    "test_idx"
                    "true_labels_train"
                    "true_labels_val"
                    "true_labels_test"
    """
    
    assert SAVE_MODEL == OVERWRITE or SAVE_MODEL == False
    CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY)
    result_table = pd.read_csv(f"results/result_table_{FORMULA}.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd"])

    if (not OVERWRITE) and result_table.index.isin([CURRENT_INDEX]).any():
        print(f"NO OVERWRITE: Results already in Dataframe for Index {CURRENT_INDEX}.")
        return (None, None)
    else:
        inits = get_init_crystal_states(edge_encoding_dim = E_DIM, SAMPLE_SIZE = SAMPLE_SIZE, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = FORMULA) #
        types = inits[-1] 
        # Change sample size to None if all samples should be read.
        model_results, performance_results, batches = run_model_with_hyperparameters(E_DIM = E_DIM,
                                                                                    R_SWITCH=R_SWITCH,
                                                                                    R_CUT=R_CUT,
                                                                                    DISTANCE_ENCODING_TYPE=DISTANCE_ENCODING_TYPE,
                                                                                    FEATURES=FEATURES,
                                                                                    NUM_PASSES=NUM_PASSES,
                                                                                    ACTIVATION=ACTIVATION,
                                                                                    N_EPOCHS=N_EPOCHS,
                                                                                    INITS=inits,
                                                                                    FORMULA = FORMULA,
                                                                                    LR = LR,
                                                                                    WEIGHT_DECAY = WEIGHT_DECAY
                                                                                    )
        model_results["index"] = CURRENT_INDEX
        batches["types"] = types
        if result_table.index.isin([CURRENT_INDEX]).any():
            result_table.loc[CURRENT_INDEX] = performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]
        else:
            result_table = result_table.reset_index().append(dict(zip(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd","time_needed","test_rmse","test_mae","steps","val_rmses", "best_val_rmse"],[E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]])), ignore_index = True).set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd"])
        print("Best-Val-RMSE:",performance_results["best_val_rmse"])
        result_table.to_csv(f"results/result_table_{FORMULA}.csv")
        if SAVE_MODEL:
            save_model(model_results,str(CURRENT_INDEX),CURRENT_INDEX, FORMULA)
        return model_results, batches


def save_model(model_results, filename, index, formula):
    """Save a model to a path.
    Input: 
        -   model_results: dict -> Dictionary from model_training
        -   filename: str -> path to saving location
        -   formula: [str] Formula of the chemical compound
        -   index: [tuple] -> Hyperparameter configuration as a tuple
    """
    model_tuple = (dill.dumps(model_results["model"]),model_results["params"], index)
    with open(f'models/{formula}/{filename}.pkl', 'wb') as handle:
        pickle.dump(model_tuple , handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model {filename}.pkl saved. Index = {index}")

def load_model_from_name(filename, formula):
    """Load model from filename.
    Output: model_results
    """
    with open(f'models/{formula}/{filename}.pkl', 'rb') as handle:
        model_tuple = pickle.load(handle)
    model, params, index = model_tuple
    print(f"Model {filename}.pkl loaded. Index = {model_tuple[2]}")
    model_results = {
        "model":dill.loads(model),
        "params":params
    }
    return model_results, index

def load_model_from_path(path):
    """Load model from path.
    Output: model_results
    """
    with open(path, 'rb') as handle:
        model_tuple = pickle.load(handle)
    model, params, index = model_tuple
    print(f"Model loaded. Index = {model_tuple[2]}")
    model_results = {
        "model":dill.loads(model),
        "params":params
    }
    return model_results, index



