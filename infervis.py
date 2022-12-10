from hyperoptimization_utils import load_model_from_path
from pipeline_utils import create_implicitly_batched_graphsTuple_with_encoded_distances
from preprocessing_jraph import get_init_crystal_states_for_inference, get_init_charges_for_comparison
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
from copy import deepcopy
from itertools import compress

def get_parameters_from_index(index):
    """ Getting parameters from a string or tuple index.
    Input: 
        - index: str or tuple
    Output:
        - parameters as tuple
    """
    if type(index) == str:
        index = eval(index)
    try:
        E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, ETA = index
    except:
        E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = index
        ETA = 2.0
    if type(FEATURES) == str:
        FEATURES = eval(FEATURES)
    return E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, ETA


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
#     batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors,distances, distances_encoded,init_charges,ohe_types,cutoff_mask)
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
    
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, ETA = index
    inits = get_init_crystal_states_for_inference(path = db_path,ground_truth_available = False, edge_encoding_dim = E_DIM, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = formula, eta = ETA) #
    descriptors, distances, distances_encoded, init_charges, cutoff_mask, types,ohe_types = inits

    size = descriptors.shape[0]
    if size < max_batch_size:
        batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors,distances, distances_encoded,init_charges,ohe_types,cutoff_mask, R_CUT)
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
            batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[idx],distances[idx], distances_encoded[idx],init_charges[idx],ohe_types[idx],cutoff_mask[idx], R_CUT)
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
        - model_config: str -> Path to the saved model config.
        - db_path [optional]: str -> Path to data to infer. Should be similar to data the model was trained on!
        - formula str -> Formula of the chemical compound.
    Output:
        - Image of plot.
    """
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, ETA = get_parameters_from_index(model_config)
    inits = get_init_crystal_states_for_inference(path = db_path,ground_truth_available = True, edge_encoding_dim = E_DIM, r_switch = R_SWITCH, r_cut = R_CUT, distance_encoding_type = DISTANCE_ENCODING_TYPE, formula = formula, eta = ETA) #
    descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types, ohe_types = inits
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
    batch = create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors[test_idx],distances[test_idx], distances_encoded[test_idx],init_charges[test_idx],ohe_types[test_idx],cutoff_mask[test_idx], R_CUT)
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
    print(model_config)
    # try:
    E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY, ETA = get_parameters_from_index(model_config)
    # except:
    #     E_DIM, R_SWITCH, R_CUT, DISTANCE_ENCODING_TYPE, FEATURES, NUM_PASSES, ACTIVATION, N_EPOCHS, LR, WEIGHT_DECAY = get_parameters_from_index(model_config)
    #     ETA = 2.0
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



def calculate_and_visualize_results_no_training(FORMULA, db_path = None, xrange = [-1.5,2.1],  save_name = None):
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
    init_charges, gt_charges, types = get_init_charges_for_comparison(path = db_path, formula = FORMULA)
    init_charges=init_charges[:,:,0].flatten()
    gt_charges = gt_charges.flatten()
    types = types.flatten()
    MAE = jnp.sum(jnp.abs(init_charges-gt_charges)/len(gt_charges))
    RMSE = jnp.sqrt(jnp.sum(jnp.square(init_charges-gt_charges)/len(gt_charges)))


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
        "preds":np.array(init_charges),
        "ground truth": np.array(gt_charges),
        "types":list(np.array(types.flatten()).astype(int))
        })
    df["atom_types"] = df["types"].map(lambda x : color_dict[x])
    fig = px.scatter(df, x="preds", y="ground truth",color="atom_types", color_discrete_sequence=color_discrete_sequence, width=800, height=800,   title=f"Initialized charges against ground truth for atom charges in {FORMULA}")
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
    return MAE, RMSE, fig




def visualize_steps(result_table, FORMULA, OPTIM_DICT, input_default_dict, y_max = 0.05, y_min=0.0):
    index = ["e_dim","r_switch","r_cut","distance_encoding_type","features","num_passes","activation_fn","n_epochs","lr","wd","eta"]
    all_params = ["E_DIM","R_SWITCH","R_CUT","DISTANCE_ENCODING_TYPE","FEATURES","NUM_PASSES","ACTIVATION","N_EPOCHS","LR","WEIGHT_DECAY","ETA"] 
    index_dict = {
        "E_DIM":"e_dim",
        "R_SWITCH":"r_switch",
        "R_CUT":"r_cut",
        "DISTANCE_ENCODING_TYPE":"distance_encoding_type",
        "FEATURES": "features",
        "NUM_PASSES": "num_passes",
        "ACTIVATION":"activation_fn",
        "N_EPOCHS":"n_epochs",
        "LR":"lr",
        "WEIGHT_DECAY":"wd",
        "ETA":"eta"
    }
    legend_dict = {
        "E_DIM":"no. of edge-dimensions",
        "R_SWITCH":"r_switch",
        "R_CUT":"r_cut",
        "DISTANCE_ENCODING_TYPE":"distance encoding type",
        "FEATURES": "MLP layers",
        "NUM_PASSES": "no. of passing steps",
        "ACTIVATION":"activation function",
        "N_EPOCHS":"no. of epochs",
        "LR":"learning rate",
        "WEIGHT_DECAY":"weight decay",
        "ETA":"eta"
    }
    
    OPTIM_DICT = deepcopy(OPTIM_DICT)
    result_rows = []
    keys = list(OPTIM_DICT.keys())
    values = list(OPTIM_DICT.values())
    keep_in_indices_bool = [False if param in keys else True for param in all_params]
    keep_in_indices = np.arange(len(index))[keep_in_indices_bool]
    keep_in_params = list(compress(all_params,keep_in_indices_bool))
    
    if type(y_max) == list:
        figs = []
        legend = []
        for ym in y_max:
            result_rows = []
            legend = []
            if len(keys) >1:
                        if len(values) != 2:
                            print(len(values))
                            raise ValueError()
                        
                        ase = values[0]
                        bse  = values[1]
                        for a in ase:
                            for b in bse:
                                DEFAULT_DICT = deepcopy(input_default_dict)
                                DEFAULT_DICT[keys[0]] = a
                                DEFAULT_DICT[keys[1]] = b
                                legend.append((a,b))
                                result_rows.append(result_table.loc[tuple(DEFAULT_DICT.values())])
                                keep_in_values = list(compress(tuple(DEFAULT_DICT.values()),keep_in_indices_bool))
                                fig = go.Figure(layout = {"width":800, "height":800,"title":f"<b>Validation RMSEs for each step in {FORMULA}</b><br><sup><br>{get_subtitle(keep_in_params,keep_in_values,legend_dict)}</sup>"})
                                fig.update_layout({"plot_bgcolor":"white","yaxis": {"title":"Validation RMSE","gridcolor":'dimgrey',"minor":{"gridcolor":'rgb(230, 230, 230)'},"range":[y_min,ym]},"margin":{"t":140,"b":20},"title":{"y":0.95},"legend":{"x":0.8,"y":1.15},"xaxis":{"title":"training epoch"}})
                                steps_total = []
                                for i in range(len(result_rows)):
                                    steps_total+=eval(result_rows[i]["steps"])
                                    fig.add_trace(go.Scatter(x=eval(result_rows[i]["steps"]), y=eval(result_rows[i]["val_rmses"]),name=f"{legend_dict[keys[0]]}: {legend[i][0]}<br>{legend_dict[keys[1]]}: {legend[i][1]}"))
                                fig.update_layout({"xaxis":{"range":[0,max(steps_total)]}})
            elif len(values) == 1:
                legend = OPTIM_DICT[keys[0]]
                for a in values[0]:
                    DEFAULT_DICT = deepcopy(input_default_dict)
                    DEFAULT_DICT[keys[0]] = a
                    result_rows.append(result_table.loc[tuple(DEFAULT_DICT.values())])
                    keep_in_values = list(compress(tuple(DEFAULT_DICT.values()),keep_in_indices_bool))
                    fig = go.Figure(layout = {"width":800, "height":800,"title":f"<b>Validation RMSEs for each step in {FORMULA}</b><br><sup><br>{get_subtitle(keep_in_params,keep_in_values,legend_dict)}</sup>"})
                    fig.update_layout({"plot_bgcolor":"white", "yaxis": {"title":"Validation RMSE","gridcolor":'dimgrey',"minor":{"gridcolor":'rgb(230, 230, 230)'},"range":[y_min,ym]},"margin":{"t":140,"b":20},"title":{"y":0.95},"legend":{"x":0.8,"y":1.15},"xaxis":{"title":"training epoch"}})
                    steps_total = []
                    for i in range(len(result_rows)):
                        steps_total+=eval(result_rows[i]["steps"])
                        fig.add_trace(go.Scatter(x=eval(result_rows[i]["steps"]), y=eval(result_rows[i]["val_rmses"]),name=f"{legend_dict[keys[0]]}: {legend[i]}"))
                    fig.update_layout({"xaxis":{"range":[0,max(steps_total)]}})
            figs.append(fig)
        return figs

    else:
        legend = []
        if len(keys) >1:
            
            if len(values) != 2:
                print(len(values))
                raise ValueError()
            ase = values[0]
            bse  = values[1]
            for a in ase:
                for b in bse:
                    DEFAULT_DICT = deepcopy(input_default_dict)
                    DEFAULT_DICT[keys[0]] = a
                    DEFAULT_DICT[keys[1]] = b
                    legend.append((a,b))
                    result_rows.append(result_table.loc[tuple(DEFAULT_DICT.values())])
                    keep_in_values = list(compress(tuple(DEFAULT_DICT.values()),keep_in_indices_bool))
                    fig = go.Figure(layout = {"width":800, "height":800,"title":f"<b>Validation RMSEs for each step in {FORMULA}</b><br><sup><br>{get_subtitle(keep_in_params,keep_in_values,legend_dict)}</sup>"})
                    fig.update_layout({"plot_bgcolor":"white","yaxis": {"title":"Validation RMSE","gridcolor":'dimgrey',"minor":{"gridcolor":'rgb(230, 230, 230)'},"range":[y_min,y_max]},"margin":{"t":140,"b":20},"title":{"y":0.95},"legend":{"x":0.8,"y":1.15},"xaxis":{"title":"training epoch"}})
                    steps_total = []
                    for i in range(len(result_rows)):
                        steps_total+=eval(result_rows[i]["steps"])
                        fig.add_trace(go.Scatter(x=eval(result_rows[i]["steps"]), y=eval(result_rows[i]["val_rmses"]),name=f"{legend_dict[keys[0]]}: {legend[i][0]}<br>{legend_dict[keys[1]]}: {legend[i][1]}"))
                    fig.update_layout({"xaxis":{"range":[0,max(steps_total)]}})
        elif len(values) == 1:
            legend = OPTIM_DICT[keys[0]]
            for a in values[0]:
                DEFAULT_DICT = deepcopy(input_default_dict)
                DEFAULT_DICT[keys[0]] = a
                result_rows.append(result_table.loc[tuple(DEFAULT_DICT.values())])
                keep_in_values = list(compress(tuple(DEFAULT_DICT.values()),keep_in_indices_bool))
                fig = go.Figure(layout = {"width":800, "height":800,"title":f"<b>Validation RMSEs for each step in {FORMULA}</b><br><sup><br>{get_subtitle(keep_in_params,keep_in_values,legend_dict)}</sup>"})
                fig.update_layout({"plot_bgcolor":"white", "yaxis": {"title":"Validation RMSE","gridcolor":'dimgrey',"minor":{"gridcolor":'rgb(230, 230, 230)'},"range":[y_min,y_max]},"margin":{"t":140,"b":20},"title":{"y":0.95},"legend":{"x":0.8,"y":1.15},"xaxis":{"title":"training epoch"}})
                steps_total = []
                for i in range(len(result_rows)):
                    steps_total+=eval(result_rows[i]["steps"])
                    fig.add_trace(go.Scatter(x=eval(result_rows[i]["steps"]), y=eval(result_rows[i]["val_rmses"]),name=f"{legend_dict[keys[0]]}: {legend[i]}"))
                fig.update_layout({"xaxis":{"range":[0,max(steps_total)]}})
        return fig

def get_subtitle(index,values, legend_dict):
    return_list = []
    assert len(index) == len(values)
    i = 0
    while i  < len(index):
        if (i) % 3 == 0 and i!=0:
            return_list.append(f"<br>{legend_dict[index[i]]} = {str(values[i])}")
        else:
            return_list.append(f"{legend_dict[index[i]]} = {str(values[i])}")
        i+=1
    return ", ".join(return_list)

