import numpy as np
import ase.db
import jax
import jax.numpy as jnp
import functools
import tqdm.auto
import typing

SORTED_ELEMENTS = sorted(["Sr", "Ti", "O"])

# This is the element-to-type map used for generating the descriptors, which
# determines the order of the descriptor vector for each atom.
SYMBOL_MAP = {s: i for i, s in enumerate(SORTED_ELEMENTS)}

type_to_AN_dict = {
    0: 8,
    1: 38,
    2: 22
}

# 65 Oxygen atoms
# 15 Strontium atoms
# 25 Titanium atoms
type_to_charges_dict = {
    0: np.float64(-1.15),
    1: np.float64(1.83),
    2: np.float64(1.892)
}

##############################################################################

def center_at_atoms_diagonal(coordinates: jnp.ndarray, cell_size_diagonal: jnp.ndarray):
    """ Vectorized version: v_center_at_atoms_diagonal -> Calculate the distances towards each atom inside of the batched coordinate array

    Input:
        - coordinates (batchsize x n_atom x 3 (dimensions)): Coordinates of each atom.
        - cell_size_diagonal (batchsize x 3): Diagonal values of the cell_size matrix. [2,2] = 0 as it is irrelevant.

    Output:
        - batchwise distances: (batchsize x n_atom x n_atom)

    """
    delta = coordinates - coordinates[:, jnp.newaxis, :]
    zero_indices = cell_size_diagonal == 0.
    icell_size_diagonal = jnp.where(zero_indices, 0., 1. / jnp.where(zero_indices, 1., cell_size_diagonal))
    delta -= cell_size_diagonal * jnp.round(delta * icell_size_diagonal)
    return jnp.sqrt(jnp.sum(delta**2, axis=-1))
v_center_at_atoms_diagonal = jax.jit(jax.vmap(center_at_atoms_diagonal))

###############################################################################    

def _aux_function_f(t):
    "First auxiliary function used in the definition of the smooth bump."
    return jnp.where(t > 0., jnp.exp(-1. / jnp.where(t > 0., t, 1.)), 0.)


def _aux_function_g(t):
    "Second auxiliary function used in the definition of the smooth bump."
    f_of_t = _aux_function_f(t)
    return f_of_t / (f_of_t + _aux_function_f(1. - t))


def smooth_cutoff(r, r_switch, r_cut):
    """One-dimensional smooth cutoff function based on a smooth bump.

    This function follows the prescription given by Loring W. Tu in
    "An Introduction to Manifolds", 2nd Edition, Springer

    Args:
        r: The radii at which the function must be evaluated.
        r_switch: The radius at which the function starts differing from 1.
        r_cut: The radius at which the function becomes exactly 0.
    """
    r_switch2 = r_switch * r_switch
    r_cut2 = r_cut * r_cut

    return 1. - _aux_function_g((r * r - r_switch2) / (r_cut2 - r_switch2))

def get_cutoff_mask(batched_distances, R_SWITCH = 1.0, R_CUT = 1.5):
    partial_cutoff = functools.partial(smooth_cutoff,
                                r_switch=R_SWITCH,
                                r_cut=R_CUT)
    return jax.jit(jax.vmap(partial_cutoff))(batched_distances)
###############################################################################

def get_gaussian_distance_encodings(batched_distances, ETA = 2.0, R_CUT = 1.5, dim_encoding = 48):
    """ Get the gaussian encodings of pair-wise distances.
    Input:
        - batched_distances: jnp.array -> Batched pair-wise distances between all atoms (batchsize x natom x natom)
        - ETA: float-> Custom variable for gaussian encoding (can be varied or trained) -> Larger ETA = sharper peaks
        - R_CUT: float -> Cutoff variable and upper border for mu. Makes sure that the gaussian encodings are well distributed between 0.1 and cutoff.
        - dim_encoding: int -> number of dimensions for gaussian encoding
    
    Output:
        - gaussian-encoded batched distances: jnp.array -> (batchsize x natom x natom x dim_encoding)
    """
    mu = jnp.linspace(0.1, R_CUT, num=dim_encoding)
    mu = jnp.expand_dims(mu, 0)
    mu = jnp.expand_dims(mu, 0)
    mu = jnp.expand_dims(mu, 0)
    mu = jnp.tile(mu, [batched_distances.shape[0], batched_distances.shape[1], batched_distances.shape[2], 1])
    # print("Mu-Shape",mu.shape)
    batched_distances = jnp.expand_dims(batched_distances, -1)
    batched_distances = jnp.tile(batched_distances, [1, 1, 1, dim_encoding])
    # print("BD-Shape",mu.shape)
    # Question: Should we include the cutoff function already here (e.g. multiply distances with cutoff mask)
    e_encodings = jnp.exp(-ETA * (batched_distances-mu)**2)
    # print("e-enc shape",e_encodings.shape)
    return e_encodings

###############################################################################

def get_init_charges(element_array: jnp.array,
                    init_method: str, # ["specific","average"]
                    charge_dict: dict = None,
                    total_charge: jnp.array = None):
    """ Get the charges for an array of element (encoded with type numbers, not atomic numbers).
    Input:
        - element_array (batchsize x natom)
        - init_method: str -> either 'specific' or 'average'. 
            'specific' means that each element has a specific init charge dependent on the graphics sent from Jésus.
            'average' means that the total charge is just divided by the number of atoms. As the total charge is 0, it would just return an array full of zeroes.
        - charge_dict: Dictionary that connects element encoding (e.g. 0, 1 or 2) with specific charge.
        - total_charge: total charge to average on or to check if specific charges are correct (sum of single charges == total_charge)
    
    Output:
        - return_array of single charges for each atom over all batches.
    """
    if init_method == "average":
        average_charge = total_charge.astype(jnp.float32)/float(element_array.shape[1]) # for a 2D array (batchsize x natom)
        return jnp.ones_like(element_array)*np.expand_dims(average_charge,1) # broadcasting the result even though dimensions are different.
    if init_method == "specific":
        return_array = np.vectorize(charge_dict.get)(element_array)
        assert (jnp.round(jnp.sum(return_array,axis=1),3) == jnp.round(total_charge.astype(jnp.float32),3)).all()
        return return_array 
    else:
        raise Exception("Not a viable init method in function get_init_charges().")
###############################################################################

def get_init_crystal_states(path: str = "data/SrTiO3_500.db",
                            distance_encoding_type = "gaussian", # ["gaussian","logarithmic","root","none"] TODO
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
    # Other Variables
    ETA = 2.0 # Gaussian Variable
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
    descriptors = jnp.asarray(descriptors)
    positions = jnp.asarray(positions)
    atomic_numbers = jnp.asarray(atomic_numbers)
    types = jnp.asarray(types)
    gt_charges = jnp.asarray(charges)
    return_dict = dict()
    return_dict["types"] = types
    return_dict["natom"] = positions.shape[1]
    return_dict["atomic_numbers"] = atomic_numbers
    return_dict["ohe_types"] = jnp.array(jnp.transpose(jnp.array([jnp.where(types==x,1,0) for x in jnp.array([0,1,2])]),axes=[1,2,0]))
    return_dict["gt_charges"] = gt_charges
    return_dict["total_charges"] = jnp.zeros(SAMPLE_SIZE)
    # This can be changed to "average", so all charges are initialized as 0.0
    return_dict["init_charges"] = get_init_charges(types,
                                                    "specific",
                                                    type_to_charges_dict,
                                                    return_dict["total_charges"])
    return_dict["positions"] = positions
    # Descriptor tensors are reshaped to flatten to bessel descriptors
    descriptors = descriptors.reshape(*descriptors.shape[:2],-1)
    return_dict["descriptors"] = descriptors
    cell_size = np.array(cell_size)
    # Run this as cell size of z-axis is irrelevant
    cell_size[2,2]=0.0
    cell_size = jnp.array(cell_size)
    return_dict["distances"] = v_center_at_atoms_diagonal(return_dict["positions"],jnp.repeat(jnp.diag(cell_size)[jnp.newaxis,:],SAMPLE_SIZE, axis=0))
    return_dict["cutoff_mask"] = get_cutoff_mask(batched_distances = return_dict["distances"], R_SWITCH = r_switch, R_CUT = r_cut)
    if distance_encoding_type=="gaussian":
        return_dict["distances_encoded"] = get_gaussian_distance_encodings(batched_distances = return_dict["distances"], ETA = eta, R_CUT = r_cut, dim_encoding = edge_encoding_dim)
    return return_dict


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