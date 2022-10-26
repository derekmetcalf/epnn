import numpy as np
import ase.db
import jax
import jax.numpy as jnp
import functools
import tqdm.auto
import typing
import math
from preprocessing_base import get_cutoff_mask, get_init_charges,get_init_charges_single, get_gaussian_distance_encodings, v_center_at_atoms_diagonal
import json
import os

def get_init_crystal_states(distance_encoding_type = "root", # ["log1","root","none"] 
                            r_switch = 1.0,
                            r_cut = 1.5,
                            edge_encoding_dim = 126,
                            eta = 2.0, # gaussian encoding variable
                            SAMPLE_SIZE = None,
                            init_type = "specific",
                            formula = "SrTiO3",
                            ):
    """ Returns preprocessed relevant data from the crystal database.

    Input:
        - distance_encoding_type [“none”, “root”, “log1”]: You can encode the distances between atoms with the following distance encoding types before they are embedded.
                    -	“none”: no distance encoding before embedding the distances
                    -	“root”: square root of the distances before embedding the distances.
                    -	“log1”: logarithmic value of (distance+1) [to avoid negative values]
        - r_switch: float -> The radius at which the function starts differing from 1.
        - r_cut: float -> The radius at which the function becomes exactly 0 and gets cut off
        - edge_encoding_dim: int -> dimension of edge encoding
        - eta: float -> gaussian encoding variable
        - SAMPLE_SIZE: [int] -> How many samples from the database shall be used for training, "None" equals all data.
        - init_type: str -> either 'specific' or 'average'. 
            'specific' means that each element has a specific init charge dependent on the graphics sent from Jésus.
            'average' means that the total charge is just divided by the number of atoms. As the total charge is 0, it would just return an array full of zeroes.
        - formula: [str] Formula of the chemical compound
    Output:
        - Tuple with variables:
            -   descriptors: jnp.array (n_samples, n_atom, h_dim) -> Bessel descriptors
            -   distances: jnp.array (n_samples, n_atom, n_atom)  -> distances between atoms
            -   distances_encoded: jnp.array (n_samples, n_atom, n_atom, e_dim) -> encoded distances between atoms
            -   init_charges: jnp.array (n_samples, n_atom) -> initial charges for all atoms
            -   gt_charges: jnp.array (n_samples, n_atom) -> ground truth charges for all atoms
            -   cutoff_mask: jnp.array (n_samples, n_atom, n_atom)  -> cutoff mask for node & edge effects in message passing
            -   types: jnp.array (n_samples, n_atom) -> integer types for all atoms
    """
    if formula == "C30H120N30O45":
        N_ATOMS_CATION = 11
        N_ATOMS_ANION = 4
        # all_atom_charges = collections.defaultdict(list)
        # all_cation_charges = []
        # all_anion_charges = []
        # Reading relevant data from presets file.
    try:
        with open (os.getcwd()+"/presets.json") as f:
            presets = json.load(f)
            presets = presets[formula]
    except:
        raise ValueError(f"Formula {formula} not found in presets.json.")
    path = presets["path"]
    type_to_charges_dict = {int(k):v for k,v in presets["charge_map"].items()}
    SYMBOL_MAP = presets["symbol_map"]
    
    # Atom type, from 0 to n_types - 1.
    types = []
    # In case we want to use them for the electron-passing NN.
    atomic_numbers = []
    # Positions of each atom in each configuration, in Cartesian coordinates
    # expressed in Å.
    positions = []
    # There are periodic boundary conditions in effect along the X and Y 
    # directions. The length of the simulation box along those directions
    # is stored in the elements [0, 0] and [1, 1] of the 3x3 matrices
    # stored in this array. Although there is a cell matrix for each
    # configuration, they are all the same in this case. The units are also
    # Å. The [2, 2] element is immaterial, and the rest are zero.
    cell_lengths = []
    # Spherical Bessel descriptors for each atom, generated using the following
    # parameters:
    # N_MAX = 5
    # R_CUT = 5.5
    descriptors = []
    # DDEC6 charges that we will try to predict,
    charges = []

    cell_size = np.array([])

    with ase.db.connect(path) as db:
        for idx, row in enumerate(tqdm.auto.tqdm(db.select(), total=db.count())):
            descriptors.append(row["data"]["bessel_descriptors"])
            charges.append(row["data"]["ddec6_charges"])

            atoms = row.toatoms()
            symbols = atoms.get_chemical_symbols()
            if formula == "C30H120N30O45":
                n_pairs = len(symbols) // (N_ATOMS_CATION + N_ATOMS_ANION)
                extended_symbols = []
                for i, s in enumerate(symbols):
                    if s == "N":
                        if i < n_pairs * N_ATOMS_CATION:
                            extended_symbols.append("N_cation")
                        else:
                            extended_symbols.append("N_anion")
                    else:
                        extended_symbols.append(s)
                symbols = extended_symbols

                # for (s, c) in zip(symbols, charges[-1]):
                #     all_atom_charges[s].append(c)
                # cation_charges = (
                #     charges[-1][: N_ATOMS_CATION * n_pairs]
                #     .reshape((-1, N_ATOMS_CATION))
                #     .sum(axis=1)
                # )
                # all_cation_charges.extend(cation_charges.tolist())
                # anion_charges = (
                #     charges[-1][N_ATOMS_CATION * n_pairs :]
                #     .reshape((-1, N_ATOMS_ANION))
                #     .sum(axis=1)
                # )
                # all_anion_charges.extend(anion_charges.tolist())
            types.append([SYMBOL_MAP[s] for s in symbols])
            atomic_numbers.append(atoms.get_atomic_numbers())
            positions.append(atoms.get_positions())
            cell_lengths.append(atoms.cell.lengths())
            cell_size = atoms.cell
            if SAMPLE_SIZE and (SAMPLE_SIZE-1) <= idx:
                break
    if not SAMPLE_SIZE:
        SAMPLE_SIZE = idx+1
    # Descriptor tensors are reshaped to flatten to bessel descriptors
    descriptors = jnp.asarray(descriptors)
    descriptors = descriptors.reshape(*descriptors.shape[:2],-1)
    positions = jnp.asarray(positions)
    types = jnp.asarray(types)
    gt_charges = jnp.asarray(charges)
    total_charges = jnp.repeat(jnp.float32(presets["total_charge"]), SAMPLE_SIZE)
    natom = positions.shape[1]
    
    # This can be changed to "average", so all charges are initialized as 0.0
    init_charges = get_init_charges(types,
                                    init_type,
                                    type_to_charges_dict,
                                    total_charges)
    init_charges = jnp.expand_dims(init_charges,axis=-1)


    cell_size = np.array(cell_size)
    if formula == "SrTiO3":
        # Run this as cell size of z-axis is irrelevant
        cell_size[2,2]=0.0
    cell_size = jnp.array(cell_size)
    distances = v_center_at_atoms_diagonal(positions,jnp.repeat(jnp.diag(cell_size)[jnp.newaxis,:],SAMPLE_SIZE, axis=0))
    cutoff_mask = get_cutoff_mask(batched_distances = distances, R_SWITCH = r_switch, R_CUT = r_cut)
    if distance_encoding_type=="log1":
        distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.log(distances+1.0), ETA = eta, R_CUT = math.log(r_cut+1), dim_encoding = edge_encoding_dim)
    elif distance_encoding_type=="root":
        distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.sqrt(distances), ETA = eta, R_CUT = math.sqrt(r_cut), dim_encoding = edge_encoding_dim)
    else:
        distances_encoded = get_gaussian_distance_encodings(batched_distances = distances, ETA = eta, R_CUT = r_cut, dim_encoding = edge_encoding_dim)
    return descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types


def get_init_crystal_states_for_inference(
                            path,
                            ground_truth_available,
                            distance_encoding_type = "root", # ["log1","root","none"] 
                            r_switch = 1.0,
                            r_cut = 1.5,
                            edge_encoding_dim = 126,
                            eta = 2.0, # gaussian encoding variable
                            SAMPLE_SIZE = None,
                            init_type = "specific",
                            formula = "SrTiO3",
                            ):
    """ Returns preprocessed relevant data from the crystal database.

    Input:
        - path: Filepath to database
        - ground_truth_available: True if we have ground_truth information for charges in the database
        - distance_encoding_type [“none”, “root”, “log1”]: You can encode the distances between atoms with the following distance encoding types before they are embedded.
                    -	“none”: no distance encoding before embedding the distances
                    -	“root”: square root of the distances before embedding the distances.
                    -	“log1”: logarithmic value of (distance+1) [to avoid negative values]
        - r_switch: The radius at which the function starts differing from 1.
        - r_cut: The radius at which the function becomes exactly 0.
        - edge_encoding_dim: int -> dimension of edge encoding
        - eta: float -> gaussian encoding variable
        - SAMPLE_SIZE: [int] -> How many samples from the database shall be used for training, "None" equals all data.
        - init_type: str -> either 'specific' or 'average'. 
            'specific' means that each element has a specific init charge dependent on the graphics sent from Jésus.
            'average' means that the total charge is just divided by the number of atoms. As the total charge is 0, it would just return an array full of zeroes.
        - formula: [str] Formula of the chemical compound
    Output:
        - Tuple with variables:
            -   descriptors: jnp.array (n_samples, n_atom, h_dim) -> Bessel descriptors
            -   distances: jnp.array (n_samples, n_atom, n_atom)  -> distances between atoms
            -   distances_encoded: jnp.array (n_samples, n_atom, n_atom, e_dim) -> encoded distances between atoms
            -   init_charges: jnp.array (n_samples, n_atom) -> initial charges for all atoms
            -   gt_charges [optional]: jnp.array (n_samples, n_atom) -> ground truth charges for all atoms
            -   cutoff_mask: jnp.array (n_samples, n_atom, n_atom)  -> cutoff mask for node & edge effects in message passing
            -   types: jnp.array (n_samples, n_atom) -> integer types for all atoms
    """
    if formula == "C30H120N30O45":
        N_ATOMS_CATION = 11
        N_ATOMS_ANION = 4
        # all_atom_charges = collections.defaultdict(list)
        # all_cation_charges = []
        # all_anion_charges = []
        # Reading relevant data from presets file.
    try:
        with open (os.getcwd()+"/presets.json") as f:
            presets = json.load(f)
            presets = presets[formula]
    except:
        raise ValueError(f"Formula {formula} not found in presets.json.")
    type_to_charges_dict = {int(k):v for k,v in presets["charge_map"].items()}
    SYMBOL_MAP = presets["symbol_map"]
    
    # Atom type, from 0 to n_types - 1.
    types = []
    # In case we want to use them for the electron-passing NN.
    atomic_numbers = []
    # Positions of each atom in each configuration, in Cartesian coordinates
    # expressed in Å.
    positions = []
    # There are periodic boundary conditions in effect along the X and Y 
    # directions. The length of the simulation box along those directions
    # is stored in the elements [0, 0] and [1, 1] of the 3x3 matrices
    # stored in this array. Although there is a cell matrix for each
    # configuration, they are all the same in this case. The units are also
    # Å. The [2, 2] element is immaterial, and the rest are zero.
    cell_lengths = []
    # Spherical Bessel descriptors for each atom, generated using the following
    # parameters:
    # N_MAX = 5
    # R_CUT = 5.5
    descriptors = []
    # DDEC6 charges that we will try to predict,
    charges = []

    cell_size = np.array([])

    with ase.db.connect(path) as db:
        for idx, row in enumerate(tqdm.auto.tqdm(db.select(), total=db.count())):
            descriptors.append(row["data"]["bessel_descriptors"])
            if ground_truth_available:
                charges.append(row["data"]["ddec6_charges"])

            atoms = row.toatoms()
            symbols = atoms.get_chemical_symbols()
            if formula == "C30H120N30O45":
                n_pairs = len(symbols) // (N_ATOMS_CATION + N_ATOMS_ANION)
                extended_symbols = []
                for i, s in enumerate(symbols):
                    if s == "N":
                        if i < n_pairs * N_ATOMS_CATION:
                            extended_symbols.append("N_cation")
                        else:
                            extended_symbols.append("N_anion")
                    else:
                        extended_symbols.append(s)
                symbols = extended_symbols

                # for (s, c) in zip(symbols, charges[-1]):
                #     all_atom_charges[s].append(c)
                # cation_charges = (
                #     charges[-1][: N_ATOMS_CATION * n_pairs]
                #     .reshape((-1, N_ATOMS_CATION))
                #     .sum(axis=1)
                # )
                # all_cation_charges.extend(cation_charges.tolist())
                # anion_charges = (
                #     charges[-1][N_ATOMS_CATION * n_pairs :]
                #     .reshape((-1, N_ATOMS_ANION))
                #     .sum(axis=1)
                # )
                # all_anion_charges.extend(anion_charges.tolist())
            types.append([SYMBOL_MAP[s] for s in symbols])
            atomic_numbers.append(atoms.get_atomic_numbers())
            positions.append(atoms.get_positions())
            cell_lengths.append(atoms.cell.lengths())
            cell_size = atoms.cell
            if SAMPLE_SIZE and (SAMPLE_SIZE-1) <= idx:
                break
    if not SAMPLE_SIZE:
        SAMPLE_SIZE = idx+1
    # Descriptor tensors are reshaped to flatten to bessel descriptors
    descriptors = jnp.asarray(descriptors)
    descriptors = descriptors.reshape(*descriptors.shape[:2],-1)
    positions = jnp.asarray(positions)
    types = jnp.asarray(types)
    if ground_truth_available:
        gt_charges = jnp.asarray(charges)
    total_charges = jnp.repeat(jnp.float32(presets["total_charge"]), SAMPLE_SIZE)
    natom = positions.shape[1]
    
    # This can be changed to "average", so all charges are initialized as 0.0
    init_charges = get_init_charges(types,
                                    init_type,
                                    type_to_charges_dict,
                                    total_charges)
    init_charges = jnp.expand_dims(init_charges,axis=-1)


    cell_size = np.array(cell_size)
    if formula == "SrTiO3":
        # Run this as cell size of z-axis is irrelevant
        cell_size[2,2]=0.0
    cell_size = jnp.array(cell_size)
    distances = v_center_at_atoms_diagonal(positions,jnp.repeat(jnp.diag(cell_size)[jnp.newaxis,:],SAMPLE_SIZE, axis=0))
    cutoff_mask = get_cutoff_mask(batched_distances = distances, R_SWITCH = r_switch, R_CUT = r_cut)
    if distance_encoding_type=="log1":
        distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.log(distances+1.0), ETA = eta, R_CUT = math.log(r_cut+1), dim_encoding = edge_encoding_dim)
    elif distance_encoding_type=="root":
        distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.sqrt(distances), ETA = eta, R_CUT = math.sqrt(r_cut), dim_encoding = edge_encoding_dim)
    else:
        distances_encoded = get_gaussian_distance_encodings(batched_distances = distances, ETA = eta, R_CUT = r_cut, dim_encoding = edge_encoding_dim)
    if ground_truth_available:
        return descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types
    else:
        return descriptors, distances, distances_encoded, init_charges, cutoff_mask, types



# def get_init_crystal_states_single(
#                             descriptors : jnp.array,
#                             positions : jnp.array,
#                             gt_charges : jnp.array,
#                             types: jnp.array,
#                             type_to_charges_dict,
#                             cell_size : jnp.array,
#                             distance_encoding_type = "root", # ["log1","root","none"] 
#                             r_switch = 1.0,
#                             r_cut = 1.5,
#                             edge_encoding_dim = 24,
#                             eta = 2.0, # gaussian encoding variable
#                             SAMPLE_SIZE = None,
#                             init_type = "specific",
#                             formula = "SrTiO3",
#                             ):
#     """ Returns preprocessed important data from the crystal database.

#     Input:
#         - descriptors : jnp.array -> single descriptor of a sample.
#         - positions : jnp.array -> single positions-array of a sample.
#         - gt_charges : jnp.array -> single gt_charges-array of a sample.
#         - types: jnp.array -> single types-array of a sample.
#         - type_to_charges_dict -> dict for type_to_charges
#         - cell_size : jnp.array -> single descriptor of a sample.
#         - distance_encoding_type: str -> Type of distance encoding (root/log/none)
#         - r_switch: The radius at which the function starts differing from 1.
#         - r_cut: The radius at which the function becomes exactly 0.
#     Output:
#         - Dictionary with keys:
#             - "charges": ground truth charges for each atom in each slab (batchsize x n_atom)
#             - "types": element info of each atom encoded (batchsize x n_atom) (0 = Oxygen, 1 = Strontium, 2 = Titanium)
#             - "atomic_numbers": atomic number of each atom (batchsize x n_atom) (8 = Oxygen, 38 = Strontium, 22 = Titanium)
#             - "positions": positions of all atoms (batchsize x n_atom x 3)
#             - "distances": pairwise distances between all atoms (batchsize x n_atom x n_atom)
#     """
#     #####################################
#     #####################################
#     ####### Stuff to run before function!
#     #####################################
#     #####################################
#     # Reading relevant data from presets file.
#     try:
#         with open (os.getcwd()+"/presets.json") as f:
#             presets = json.load(f)
#             presets = presets[formula]
#     except:
#         raise ValueError(f"Formula {formula} not found in presets.json.")
#     path = presets["path"]
#     type_to_charges_dict = {int(k):v for k,v in presets["charge_map"].items()}
#     SYMBOL_MAP = presets["symbol_map"]
#     total_charge = float(presets["total_charge"])



#     cell_size = np.array(cell_size)
#     # Run this as cell size of z-axis is irrelevant
#     cell_size[2,2]=0.0
#     cell_size = jnp.array(cell_size)

#     # Descriptor tensors are reshaped to flatten to bessel descriptors
#     descriptors = descriptors.reshape(*descriptors.shape[:2],-1)
#     types = jnp.asarray(types)
#     natom = positions.shape[1]
    
#     # This can be changed to "average", so all charges are initialized as 0.0
#     init_charges = get_init_charges_single(types,
#                                     init_type,
#                                     type_to_charges_dict,
#                                     total_charge)
#     init_charges = jnp.expand_dims(init_charges,axis=-1)

    
#     distances = v_center_at_atoms_diagonal(positions,jnp.repeat(jnp.diag(cell_size)[jnp.newaxis,:],SAMPLE_SIZE, axis=0))
#     cutoff_mask = get_cutoff_mask(batched_distances = distances, R_SWITCH = r_switch, R_CUT = r_cut)
#     if distance_encoding_type=="log1":
#         distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.log(distances+1.0), ETA = eta, R_CUT = math.log(r_cut+1), dim_encoding = edge_encoding_dim)
#     elif distance_encoding_type=="root":
#         distances_encoded = get_gaussian_distance_encodings(batched_distances = jnp.sqrt(distances), ETA = eta, R_CUT = math.sqrt(r_cut), dim_encoding = edge_encoding_dim)
#     else:
#         distances_encoded = get_gaussian_distance_encodings(batched_distances = distances, ETA = eta, R_CUT = r_cut, dim_encoding = edge_encoding_dim)
#     return descriptors, distances, distances_encoded, init_charges, gt_charges, cutoff_mask, types



if __name__ == "__main__":
    pass
    # h_dim = 126
    # e_dim = 126
    # path = "data/SrTiO3_500.db"
    # n_elems = 3
    # preprocessed_dict = get_init_crystal_states(path, SAMPLE_SIZE = 100)
    # natom=105
    # descriptors = preprocessed_dict["descriptors"]
    # test = jnp.tile(jnp.expand_dims(descriptors,axis=2),(1,1,natom,1))
    # print(preprocessed_dict["cutoff_mask"]-np.transpose(preprocessed_dict["cutoff_mask"], axes=[0,2,1])==np.zeros(preprocessed_dict["cutoff_mask"].shape).all())
    # print("Something")