from hyperoptimization_utils import load_model_from_path
from pipeline_utils import create_implicitly_batched_graphsTuple_with_encoded_distances
from preprocessing_jraph import get_init_crystal_states_for_inference
import jax
from jax import random, numpy as jnp
import haiku as hk
import jraph
import json
import os
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import tqdm.auto
import ase

def get_parameters_from_index(index):
    """ Getting parameters from a string or tuple index.
    Input: 
        - index: str or tuple
    Output:
        - parameters as tuple
    """
    if type(index) == str:
        index = eval(index)
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = index
    if type(FEATURES) == str:
        FEATURES = eval(FEATURES)
    return E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY


# def get_inference_batch(model_config, db_path, formula, indices = None):
#     """ Run inference on a new dataset.
#     Input: 
#         - model_path: str -> Path to the saved model.
#         - db_path [optional]: str -> Path to data to infer. Should be similar to data the model was trained on!
#         - formula: str -> Formula of the chemical compound.
#         - max_batch_size: int -> Maximum number of samples per batch
#     Output:
#         - Tensor of predicted samples with dimension (n_samples, n_atom)
#     """
#     E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = get_parameters_from_index(model_config)
#     inits = get_init_crystal_states_for_inference(path = db_path,ground_truth_available = False, edge_encoding_dim = E_DIM, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = formula) #
#     descriptors, distances, distances_encoded, init_charges, cutoff_mask, types = inits
#     #######################################
#     ### Creating batches for training #####
#     total_size = descriptors.shape[0]
#     batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors,distances, distances_encoded,init_charges,types,cutoff_mask)
#     return batch

def infer(model_path, db_path, formula, max_batch_size = 50):
    """ Run inference on a new dataset.
    Input: 
        - model_path: str -> Path to the saved model.
        - db_path [optional]: str -> Path to data to infer. Should be similar to data the model was trained on!
        - formula: str -> Formula of the chemical compound.
        - max_batch_size: int -> Maximum number of samples per batch
    Output:
        - Tensor of predicted samples with dimension (n_samples, n_atom)
    """
    model_results, index = load_model_from_path(model_path)
    model = model_results["model"]
    params = model_results["params"]
    
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = index
    inits = get_init_crystal_states_for_inference(path = db_path,ground_truth_available = False, edge_encoding_dim = E_DIM, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = formula) #
    descriptors, distances, distances_encoded, init_charges, cutoff_mask, types = inits

    size = descriptors.shape[0]
    if size < max_batch_size:
        batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors,distances, distances_encoded,init_charges,types,cutoff_mask)
        output = model.apply(params, batch)
    else:
        n_batches = int(size/max_batch_size)
        last_batch_size = size % max_batch_size
        indices = []
        for batch_idx in range(n_batches):
            indices.append(np.arange((batch_idx*max_batch_size),((batch_idx+1)*max_batch_size)))
        indices.append(np.arange((n_batches*max_batch_size),(n_batches*max_batch_size+last_batch_size)))
        output = []
        # Run pipeline (creating graph tuples -> apply model to graph tuple) for each batch
        for idx in indices:
            #######################################
            ### Creating batches for training #####
            batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[idx],distances[idx], distances_encoded[idx],init_charges[idx],types[idx],cutoff_mask[idx])
            single_output = model.apply(params, batch)
            for i in range(NUM_PASSES-1):
                single_output = model.apply(params, single_output)
            output.extend(np.array(single_output[0]).reshape(len(idx),-1))
            print(f"{idx[-1]}/{size} were infered.")
        # print(np.asarray(output).shape)
    return output

def get_visualization_testbatch(model_config, db_path, formula):
    """ Visualize the results of a model on a batch of data about a chemical structure.
    Input: 
        - model_path: str -> Path to the saved model.
        - FORMULA: str -> Formula of the chemical compound.
        - batches [optional]: dict -> Dictionary from the training pipeline train_single_model()
        - db_path [optional]: str -> Path to data to infer. Should be similar to data the model was trained on!
        - xrange [optional]: (float, float) -> Range to plot for x- and y-axis
        - save_name [optional]: str -> If not None -> Name to folder to save the plot to.
    Output:
        - Image of plot.
    """
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = get_parameters_from_index(model_config)
    inits = get_init_crystal_states_for_inference(path = db_path,ground_truth_available = True, edge_encoding_dim = E_DIM, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = formula) #
    descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types = inits
    total_size = descriptors.shape[0]
    key = random.PRNGKey(0)
    if formula == "SrTiO3":
        batch_size = 50
    else:
        batch_size = 34

    if total_size < batch_size: batch_size = total_size
    permuted_idx = random.permutation(key, total_size)
    test_idx = permuted_idx[-batch_size:]
    
    #######################################
    ### Creating batches for training #####
    batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[test_idx],distances[test_idx], distances_encoded[test_idx],init_charges[test_idx],types[test_idx],cutoff_mask[test_idx])
    batch = {
        "types": types[test_idx],
        "batch":batch,
        "true_labels":gt_charges[test_idx],
    }

    return batch


def visualize_results(model_path, FORMULA, batches = None,db_path = None, xrange = [-1.5,2.1],  save_name = None):
    """Visualize the results of a model on a batch of data about a chemical structure.
    Input: 
        - model_path: str -> Path to the saved model.
        - FORMULA: str -> Formula of the chemical compound.
        - batches [optional]: dict -> Dictionary from the training pipeline train_single_model()
        - db_path [optional]: str -> Path to data to infer. Should be similar to data the model was trained on!
        - xrange [optional]: (float, float) -> Range to plot for x- and y-axis
        - save_name [optional]: str -> If not None -> Name to folder to save the plot to.
    Output:
        - Image of plot.
    """
    model_results, model_config = load_model_from_path(model_path)
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = get_parameters_from_index(model_config)
    @jax.jit
    def mae_loss(params: hk.Params, graph: jraph.GraphsTuple,  ground_truth: jnp.array) -> jnp.ndarray:
        output = model.apply(params, graph)
        for i in range(NUM_PASSES-1):
            output = model.apply(params, output)
        return jnp.sum(jnp.abs(output[0]-ground_truth)/len(ground_truth))
    model = model_results["model"]
    params = model_results["params"]
    if batches:
        types = batches["types"]
        test_idx = batches["test_idx"]
        batch = batches["test_batch"]
        true_labels = batches["true_labels_test"].flatten()
        types = types[test_idx]
    elif db_path:
        batch_dict = get_visualization_testbatch(model_config, db_path, FORMULA)
        types = batch_dict["types"]
        batch = batch_dict["batch"]
        true_labels = batch_dict["true_labels"].flatten()
    else:
        print("Please provide 'batches'-dict or db_path.")
        return
    test_mae = mae_loss(params, batch, true_labels)
    print("MAE:",test_mae)
    output = model.apply(params, batch)
    for i in range(NUM_PASSES-1):
        output = model.apply(params, output)

    # Getting the colors for different chemical formulas for different chemical formulas
    try:
        with open (os.getcwd()+"/presets.json") as f:
            presets = json.load(f)
            presets = presets[FORMULA]
    except:
        raise ValueError(f"Formula {FORMULA} not found in presets.json.")
    symbol_map = presets["symbol_map"]
    color_discrete_sequence = presets["color_sequence"]
    color_dict = {}
    for key, value in symbol_map.items():
        color_dict[value]=key
    
    # Creation of the plotted dataframe (for plotly)
    df = pd.DataFrame.from_dict({
        "preds":np.array(output[0]),
        "ground truth": true_labels,
        "types":list(np.array(types.flatten()).astype(int))
        })
    df["atom_types"] = df["types"].map(lambda x : color_dict[x])
    fig = px.scatter(df, x="preds", y="ground truth",color="atom_types", color_discrete_sequence=color_discrete_sequence, width=800, height=800,   title=f"Predicted charges against ground truth for atom charges in {FORMULA}")
    fig.update_traces(marker_opacity = 0.3)
    fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}], plot_bgcolor="white")
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey', range=xrange)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey', range=xrange)
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
    if save_name:
        if not os.path.isdir(f"charts/{save_name}"):
            os.mkdir(f"charts/{save_name}")
        fig.write_image(f"charts/{save_name}/p{str(xrange)}.png")
    return fig