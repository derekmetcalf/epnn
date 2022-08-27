from preprocessing_jraph import get_init_crystal_states
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


def run_model_with_hyperparameters(E_DIM : int,
                                  R_SWITCH : float,
                                  R_CUT : float,
                                  DISTANCE_ENCODING_TYPE : str,
                                  FEATURES : list,
                                  NUM_PASSES : int,
                                  ACTIVATION : str,
                                  N_EPOCHS : int,
                                  INITS : tuple,
                                  ):
    CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    key, subkey = random.split(random.PRNGKey(0))
    descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types = INITS
    h_dim = 126
    #######################################
    ### Creating batches for training #####
    total_size = descriptors.shape[0]
    train_batch_size = 50
    test_size = 50
    val_size = 50
    train_size = total_size - test_size - val_size
    key = random.PRNGKey(0)
    permuted_idx = random.permutation(key, jnp.arange(500))
    test_idx = permuted_idx[-test_size:]
    val_idx = permuted_idx[-2*test_size:-test_size]
    train_idxs = jnp.array(jnp.split(permuted_idx[:-2*test_size],int(jnp.ceil(train_size/train_batch_size))))
    train_batches = [create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[train_idx],distances[train_idx], distances_encoded[train_idx],init_charges[train_idx], types[train_idx], cutoff_mask[train_idx]) for train_idx in train_idxs]
    val_batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[val_idx],distances[val_idx], distances_encoded[val_idx],init_charges[val_idx],types[val_idx],cutoff_mask[val_idx])
    test_batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[test_idx],distances[test_idx], distances_encoded[test_idx],init_charges[test_idx],types[test_idx],cutoff_mask[test_idx])



    gep_layer = create_model(FEATURES,ACTIVATION)
    ################################
    model = hk.without_apply_rng(hk.transform(gep_layer))
    params = model.init(jax.random.PRNGKey(42), train_batches[0])
    true_labels = [gt_charges[train_idx].flatten() for train_idx in train_idxs]
    true_labels_val = gt_charges[val_idx].flatten()
    true_labels_test = gt_charges[test_idx].flatten()
    opt_init, opt_update = optax.adam(1e-2)
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
        updates, opt_state = opt_update(g, opt_state)
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
        if (i % 10)==0:
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

# tell me which parameters to loop

def run_grid_hyperparam_pipeline(DEFAULT_DICT : dict,
                            OPTIM_DICT: dict,
                            OVERWRITE: bool):
    all_params = ["E_DIM","R_SWITCH","R_CUT","DISTANCE_ENCODING_TYPE","FEATURES","NUM_PASSES","ACTIVATION","N_EPOCHS","PATH"]
    PATH = DEFAULT_DICT["PATH"]
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
                    # check if planned indices are 
                    check_if_available = False
                    result_table = pd.read_csv("results/result_table.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs"])
                    for FEATURES in DEFAULT_DICT["FEATURES"]:
                        for NUM_PASSES in DEFAULT_DICT["NUM_PASSES"]:
                            for ACTIVATION in DEFAULT_DICT["ACTIVATION"]:
                                for N_EPOCHS in DEFAULT_DICT["N_EPOCHS"]:
                                    CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS)
                                    if (not OVERWRITE) and result_table.index.isin([CURRENT_INDEX]).any():
                                            print(f"NO OVERWRITE: Results already in Dataframe for Index {CURRENT_INDEX}.")
                                            check_if_available = True
                    if check_if_available:
                        continue
                    else:
                        # only run preprocessing if results are not available for this combination of hyperparameters
                        print(E_DIM)
                        inits = get_init_crystal_states(PATH, edge_encoding_dim = E_DIM, SAMPLE_SIZE = None, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE)
                        for FEATURES in DEFAULT_DICT["FEATURES"]:
                            for NUM_PASSES in DEFAULT_DICT["NUM_PASSES"]:
                                for ACTIVATION in DEFAULT_DICT["ACTIVATION"]:
                                    for N_EPOCHS in DEFAULT_DICT["N_EPOCHS"]:
                                            CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS)
                                            result_table = pd.read_csv("results/result_table.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs"])

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
                                                                                                                            INITS=inits
                                                                                                                            )
                                                batches["types"] = types
                                                if result_table.index.isin([CURRENT_INDEX]).any():
                                                    result_table.loc[CURRENT_INDEX] = performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]
                                                else:
                                                    result_table = result_table.reset_index().append(dict(zip(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","time_needed","test_rmse","test_mae","steps","val_rmses", "best_val_rmse"],[E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]])), ignore_index = True).set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs"])
                                                print("Best-Val-RMSE:",performance_results["best_val_rmse"])
                                                result_table.to_csv("results/result_table.csv")
    


def train_single_model(E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, PATH, OVERWRITE):
    CURRENT_INDEX = (E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS)
    result_table = pd.read_csv("results/result_table.csv").set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs"])

    if (not OVERWRITE) and result_table.index.isin([CURRENT_INDEX]).any():
        print(f"NO OVERWRITE: Results already in Dataframe for Index {CURRENT_INDEX}.")
    else:
        inits = get_init_crystal_states(PATH, edge_encoding_dim = E_DIM, SAMPLE_SIZE = None, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE) #
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
                                                                                    INITS=inits
                                                                                    )
        batches["types"] = types
        if result_table.index.isin([CURRENT_INDEX]).any():
            result_table.loc[CURRENT_INDEX] = performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]
        else:
            result_table = result_table.reset_index().append(dict(zip(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","time_needed","test_rmse","test_mae","steps","val_rmses", "best_val_rmse"],[E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, str(FEATURES), NUM_PASSES, ACTIVATION, N_EPOCHS, performance_results["time_taken"], performance_results["test_rmse"], performance_results["test_mae"], list(performance_results["step_list"]), list(performance_results["rmse_list_val"]), performance_results["best_val_rmse"]])), ignore_index = True).set_index(["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs"])
        print("Best-Val-RMSE:",performance_results["best_val_rmse"])
        result_table.to_csv("results/result_table.csv")
    return model_results, batches