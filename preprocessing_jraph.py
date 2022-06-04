import numpy as np
import ase.db
import jax
import jax.numpy as jnp
import functools
import tqdm.auto
import typing
import math
from preprocessing import get_cutoff_mask, get_init_charges, get_gaussian_distance_encodings, v_center_at_atoms_diagonal, type_to_charges_dict, SYMBOL_MAP

def get_init_crystal_states(path: str = "data/SrTiO3_500.db",
                            distance_encoding_type = "root", # ["log1","root","none"] 
                            r_switch = 1.0,
                            r_cut = 1.5,
                            edge_encoding_dim = 126,
                            eta = 2.0, # gaussian encoding variable
                            SAMPLE_SIZE = None,
                            ):
    """ Returns preprocessed important data from the crystal database.

    Input:
        - filepath: Filepath to database
        - distance_encoding_type: str -> Type of distance encoding
            gaussian: Gaussian encoding over a number of dims. 
        - r_switch: The radius at which the function starts differing from 1.
        - r_cut: The radius at which the function becomes exactly 0.
    Output:
        - Dictionary with keys:
            - "charges": ground truth charges for each atom in each slab (batchsize x n_atom)
            - "types": element info of each atom encoded (batchsize x n_atom) (0 = Oxygen, 1 = Strontium, 2 = Titanium)
            - "atomic_numbers": atomic number of each atom (batchsize x n_atom) (8 = Oxygen, 38 = Strontium, 22 = Titanium)
            - "positions": positions of all atoms (batchsize x n_atom x 3)
            - "distances": pairwise distances between all atoms (batchsize x n_atom x n_atom)
    """
    # These labels identify each individual configuration in the larger
    # database this small subsample was extracted from.
    ind_labels = []
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
            ind_labels.append(row["ind_label"])
            descriptors.append(row["data"]["bessel_descriptors"])
            charges.append(row["data"]["ddec6_charges"])

            atoms = row.toatoms()
            symbols = atoms.get_chemical_symbols()
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
    total_charges = jnp.zeros(SAMPLE_SIZE)
    natom = positions.shape[1]
    
    # This can be changed to "average", so all charges are initialized as 0.0
    init_charges = get_init_charges(types,
                                    "specific",
                                    type_to_charges_dict,
                                    total_charges)
    init_charges = jnp.expand_dims(init_charges,axis=-1)


    cell_size = np.array(cell_size)
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



if __name__ == "__main__":
    h_dim = 126
    e_dim = 126
    path = "data/SrTiO3_500.db"
    n_elems = 3
    preprocessed_dict = get_init_crystal_states(path, SAMPLE_SIZE = 100)
    natom=105
    descriptors = preprocessed_dict["descriptors"]
    test = jnp.tile(jnp.expand_dims(descriptors,axis=2),(1,1,natom,1))
    print(preprocessed_dict["cutoff_mask"]-np.transpose(preprocessed_dict["cutoff_mask"], axes=[0,2,1])==np.zeros(preprocessed_dict["cutoff_mask"].shape).all())
    print("Something")