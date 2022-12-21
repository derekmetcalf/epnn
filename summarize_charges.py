#!/usr/bin/env python

import enum
import json
import collections

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ase
import ase.io
import ase.db

import tqdm.auto

np.set_printoptions(linewidth=np.inf)

DB_FILE_NAME = "data/IL_charges.db"
N_ATOMS_CATION = 11
N_ATOMS_ANION = 4

if __name__ == "__main__":
    all_atom_charges = collections.defaultdict(list)
    all_cation_charges = []
    all_anion_charges = []

    with ase.db.connect(DB_FILE_NAME) as db:
        for row in tqdm.auto.tqdm(db.select(), total=db.count()):
            charges = np.asarray(row["data"]["ddec6_charges"])
            atoms = row.toatoms()
            symbols = atoms.get_chemical_symbols()
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

            for (s, c) in zip(symbols, charges):
                all_atom_charges[s].append(c)
            cation_charges = (
                charges[: N_ATOMS_CATION * n_pairs]
                .reshape((-1, N_ATOMS_CATION))
                .sum(axis=1)
            )
            all_cation_charges.extend(cation_charges.tolist())
            anion_charges = (
                charges[N_ATOMS_CATION * n_pairs :]
                .reshape((-1, N_ATOMS_ANION))
                .sum(axis=1)
            )
            all_anion_charges.extend(anion_charges.tolist())

    def melt_dict(dictionary):
        frame = pd.DataFrame.from_dict(dictionary, orient="index")
        return (
            frame.T.melt()
            .dropna()
            .rename(columns={"variable": "type", "value": "charge / e"})
        )

    plt.figure()
    sns.boxplot(x="type", y="charge / e", data=melt_dict(all_atom_charges))
    plt.title(f"Atomic charges")
    plt.tight_layout()

    plt.figure()
    sns.boxplot(
        x="type",
        y="charge / e",
        data=melt_dict(
            dict(anion=all_anion_charges, cation=all_cation_charges)
        ),
    )
    plt.title(f"Ionic charges")
    plt.tight_layout()

    plt.show()
