#!/usr/bin/env python

import ase.db
import jax.numpy as jnp
import tqdm.auto

DB_FILE_NAME = "SrTiO3_3x1_500_configurations_and_charges.db"
# This is the element-to-type map used for generating the descriptors, which
# determines the order of the descriptor vector for each atom.
SORTED_ELEMENTS = sorted(["Sr", "Ti", "O"])
SYMBOL_MAP = {s: i for i, s in enumerate(SORTED_ELEMENTS)}

if __name__ == "__main__":
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
    cells = []
    # Spherical Bessel descriptors for each atom, generated using the following
    # parameters:
    # N_MAX = 5
    # R_CUT = 5.5
    descriptors = []
    # DDEC6 charges that we will try to predict,
    charges = []

    with ase.db.connect(DB_FILE_NAME) as db:
        for row in tqdm.auto.tqdm(db.select(), total=db.count()):
            ind_labels.append(row["ind_label"])
            descriptors.append(row["data"]["bessel_descriptors"])
            charges.append(row["data"]["ddec6_charges"])

            atoms = row.toatoms()
            symbols = atoms.get_chemical_symbols()
            types.append([SYMBOL_MAP[s] for s in symbols])
            atomic_numbers.append(atoms.get_atomic_numbers())
            positions.append(atoms.get_positions())
            cells.append(atoms.cell[...])

    descriptors = jnp.asarray(descriptors)
    charges = jnp.asarray(charges)
    types = jnp.asarray(types)
    atomic_numbers = jnp.asarray(atomic_numbers)
    positions = jnp.asarray(positions)
    cells = cells.append(cells)

