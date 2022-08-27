from preprocessing_jraph import get_init_crystal_states
import jax
from preprocessing import get_cutoff_mask, get_init_charges, get_gaussian_distance_encodings, v_center_at_atoms_diagonal, type_to_charges_dict, SYMBOL_MAP
from jax import random, numpy as jnp
import optax
import jraph
from typing import Any, Callable, Sequence, Optional, Tuple
import warnings
import numpy as np
import matplotlib.pyplot as plt
import jax.tree_util as tree
import haiku as hk
import pandas as pd
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

def create_implicitly_batched_graphsTuple_with_encoded_distances(descriptors, distances, distances_encoded, init_charges, types, cutoff_mask, cutoff = 3.0):
    batch_size = descriptors.shape[0]
    natom = descriptors.shape[1]
    # Reshaping the descriptors to go over the whole batch
    descriptors = jnp.reshape(descriptors,(batch_size*natom,descriptors.shape[2]))
    # to calculate the number of edges for each individual graph
    distances_flattened_batchwise = jnp.reshape(distances,(batch_size,natom*natom))
    n_edges = jnp.count_nonzero(jnp.logical_and(distances_flattened_batchwise > 0, distances_flattened_batchwise < cutoff),axis=1)
    n_nodes = jnp.repeat(jnp.array([natom]),batch_size)
    # Create a flattened index over all previously diagonal elements to be able to delete them from the flattened arrays.
    flatten_idx = jnp.nonzero(jnp.logical_and(distances.flatten() > 0, distances.flatten() < cutoff))[0]
    idx = jnp.nonzero(jnp.logical_and(distances.flatten() > 0, distances.flatten() < cutoff))[0]
    # Make sure that there are only edges between nodes of the same graph
    # Batch range to add onto the tiled outer products
    batch_range = jnp.reshape(jnp.repeat(jnp.arange(batch_size)*natom,natom*natom),(batch_size,natom,natom))
    # outer product over the atoms
    outer = jnp.tile(jnp.outer(jnp.ones(natom),jnp.arange(natom)).astype(jnp.int32),batch_size).reshape(batch_size,natom,natom)
    # transposed for the other variant
    outer_transposed = jnp.transpose(outer, axes=(0,2,1))
    senders = jnp.add(outer_transposed,batch_range).flatten()[flatten_idx]
    receivers = jnp.add(outer,batch_range).flatten()[flatten_idx]
    sender_descriptors = descriptors[senders,:]
    receiver_descriptors = descriptors[receivers,:]
    # types = jnp.reshape(jnp.array(jnp.transpose(jnp.array([jnp.where(types==x,1,0) for x in jnp.array([0,1,2])]),axes=[1,2,0])),(batch_size*natom,3))
    # sender_types = types[senders,:]
    # receiver_types = types[receivers,:]  
    # Encoded distances are also flattened. Combinations of the same node (diagonal) are deleted
    graph_edges = jnp.reshape(distances_encoded,(distances_encoded.shape[0]*distances_encoded.shape[1]*distances_encoded.shape[2],distances_encoded.shape[3]))[flatten_idx,:]
    # Same for cutoff_mask
    cutoff_mask = cutoff_mask.flatten()[flatten_idx]
    # Nodes contain charges
    # Edges contain concatenation of descriptors, edge_embeddings and cutoff_mask (which will be removed in the Network)
    graph= jraph.GraphsTuple(nodes = init_charges.flatten(),
                            # nodes = jnp.concatenate([descriptors,init_charges],axis=-1), Alternative 
                            senders = senders,
                            receivers = receivers,
                            # edges = jnp.concatenate([receiver_descriptors,receiver_types, sender_descriptors,sender_types, graph_edges, jnp.expand_dims(cutoff_mask,axis=-1)],axis=-1),
                            edges = jnp.concatenate([receiver_descriptors, sender_descriptors, graph_edges, jnp.expand_dims(cutoff_mask,axis=-1)],axis=-1),
                            n_node = n_nodes,
                            n_edge = n_edges,
                            globals = None)
    return graph

#########################################################

def aggregate_edges_for_nodes_fn(edges: jnp.array,
                                receivers: jnp.array,
                                cutoff_mask: jnp.array,
                                n_nodes: int) -> jnp.array:
  # multiply edges with the cutoff mask
  edges = jnp.multiply(edges,cutoff_mask)
  # sum edge embedding values over each receiver node
  return jax.ops.segment_sum(edges,receivers,n_nodes)

def create_model(features, activation):
  if activation == "swish":
    gep_layer = GraphElectronPassing(
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      MLP = lambda n: MLP_haiku_swish(features=features)(n), # use swish activation MLP
    )
  else:
    gep_layer = GraphElectronPassing(
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      MLP = lambda n: MLP_haiku(features=features)(n), # use ReLU activation MLP
    )
  return gep_layer


class MLP_haiku(hk.Module):
  def __init__(self, features: jnp.ndarray):
    super().__init__()
    self.features = features

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    layers = []
    for feat in self.features[:-1]:
      layers.append(hk.Linear(feat))
      layers.append(jax.nn.relu)
    layers.append(hk.Linear(self.features[-1]))
    mlp = hk.Sequential(layers)
    return mlp(x)

class MLP_haiku_swish(hk.Module):
  def __init__(self, features: jnp.ndarray):
    super().__init__()
    self.features = features

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    layers = []
    for feat in self.features[:-1]:
      layers.append(hk.Linear(feat))
      layers.append(jax.nn.swish)
    layers.append(hk.Linear(self.features[-1]))
    mlp = hk.Sequential(layers)
    return mlp(x)

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L506
def GraphElectronPassing(aggregate_edges_for_nodes_fn: Callable,
                        MLP: Callable,
                        h_dim: int = 129) -> Callable:
  """
  Args:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    aggregate_edges_for_nodes_fn: function used to aggregates the sender nodes.

  Returns:
    A method that applies a Graph Convolution layer.
  """

  def _ApplyGEP(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Applies a Graph Convolution layer."""
    nodes, edges, receivers, senders, _, _, _ = graph
    receiver_descriptors = edges[:,:h_dim]
    sender_descriptors=edges[:,h_dim:h_dim*2]
    graph_edges = edges[:,h_dim*2:-1]
    cutoff_mask = jnp.expand_dims(edges[:,-1],axis=-1)
    sender_charges = jnp.expand_dims(nodes[senders],axis=-1)
    receiver_charges = jnp.expand_dims(nodes[receivers],axis=-1)
    # Neural network forward: NN(q_v, q_w, h_v, h_w, e_vw) from the paper
    edges = jnp.concatenate([receiver_charges, sender_charges, receiver_descriptors, sender_descriptors, edges],axis=-1)
    edges_reversed = jnp.concatenate([sender_charges, receiver_charges, sender_descriptors, receiver_descriptors, edges],axis=-1)
    # Subtraction of both outputs to create electron-passing-output for atom v
    MLP_outputs = jnp.subtract(MLP(edges),MLP(edges_reversed))
    # aggregate_edges_for_nodes_fn is the weighting function with the cutoff_mask
    received_attributes = tree.tree_map(
      lambda e: aggregate_edges_for_nodes_fn(e, receivers, cutoff_mask, nodes.shape[0]), MLP_outputs)
    nodes = received_attributes.flatten()
    return graph._replace(nodes=nodes)
  return _ApplyGEP