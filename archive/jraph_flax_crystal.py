import os
from re import S
import scipy
import numpy as np
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from preprocessing_jraph import get_init_crystal_states
import optax
from functools import partial

atom_num_dict = {'O': 6, 'Sr': 38, 'Ti': 22}
elem_dict = {'O': 0, 'Sr': 1, 'Ti': 2}


class MLP_flax(nn.Module):
    # We need the output
    features: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


# class EPN_flax(nn.Module):
#     layers: Sequence[int]
#     T: int
#     def setup(self):
#         self.pass_fns = [MLP_flax(self.layers) for i in range(self.T)]

#     def __call__(self, h, e, q, cutoff_mask, natom):
#         one_timestep_partial = partial(self.one_timestep, h = h, e = e, cutoff_mask = cutoff_mask,natom = natom)
#         return lax.fori_loop(0, self.T, one_timestep_partial , q)
#     # @jax.jit
#     def one_timestep(self, i, q, h, e, cutoff_mask, natom):
#         # print("h-Shape:",h.shape)
#             # print("q-Shape:",q.shape)
#             inp_atom_i = jnp.concatenate((h, q), axis=-1)  # nbatch x natom x 126+1
#             inp_i = jnp.tile(jnp.expand_dims(inp_atom_i, axis=1), [1, natom, 1]) # nbatch x natom x natom x 126+1
#             inp_j = jnp.transpose(inp_i, [1, 0, 2]) #nbatch x natom x natom x 126+1

#             # This should create the inputs for the neural networks.
#             # Transposing axis 1 and 2 leads to every single combination of embedding vectors (e.g. for inp_ij:
#             # [e_a1, e_a1], [e_a1, e_a2], [e_a1, e_a3], [e_a1, e_a4], ...
#             # [e_a2, e_a1], [e_a2, e_a2], [e_a2, e_a3], [e_a2, e_a4], ...
#             # [e_a3, e_a1], [e_a3, e_a2], [e_a3, e_a3], [e_a3, e_a4], ...
#             # ...
#             # and for inp_ji
#             # [e_a1, e_a1], [e_a2, e_a1], [e_a3, e_a1], [e_a4, e_a1], ...
#             # [e_a1, e_a2], [e_a2, e_a2], [e_a3, e_a2], [e_a4, e_a2], ...
#             # [e_a1, e_a3], [e_a2, e_a3], [e_a3, e_a3], [e_a4, e_a3], ...
#             # ...
#             #
#             inp_ij = jnp.concatenate((inp_i, inp_j, e), axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
#             inp_ji = jnp.concatenate((inp_j, inp_i, e), axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
#             # print("inp-ji-Shape:",inp_ji.shape, "Expected: (105,105,302)")

#             # By flattening the inputs to shape (None, 302), we easily pass it as a large batch with each embedding-embedding-concatenation
#             # towards the pass_fn, which is a dense network.
#             # In the paper, those dense networks are called NN_s 
#             # QUESTION: Will the neural network process each combination (h_v, q_v, h_w, q_w, e_wv) individually, if we flatten it beforehand?
#             # This would make sense, as the paper also states no influence from other atoms for each specific message between two atoms.
#             flat_inp_ij = jnp.reshape(inp_ij, [-1, inp_ij.shape[-1]]) # flatten everything except last axis (nbatch*natom*natom x (126+1)x2 + 126)
#             flat_inp_ji = jnp.reshape(inp_ji, [-1, inp_ji.shape[-1]]) # (nbatch*natom*natom x (126+1)x2 + 126)
#             # print("flat_inp_ij -Shape:",flat_inp_ij.shape, "Expected: (11025, 302)")
#             print(self.pass_fns[i])
#             elec_ij_flat = self.pass_fns[i](flat_inp_ij)
#             elec_ji_flat = self.pass_fns[i](flat_inp_ji)
#             # print("elec_ij_flat-Shape:",elec_ij_flat.shape, "Expected: (11025, 1)")
            
#             # Now we reshape both arrays back to shape (None (1), 105, 105) which should be both parts of the total message.
#             elec_ij = jnp.reshape(elec_ij_flat, [natom, natom]) #reshape back?
#             elec_ji = jnp.reshape(elec_ji_flat, [natom, natom]) #reshape back?
#             # print("elec_ij-Shape:",elec_ij.shape, "Expected: (natom, natom)")

#             # The subtraction of both arrays will lead to the following array:
#             # [s1-1 - s1-1], [s1-2 - s2-1], [s1-3 - s3-1], ...
#             # [s2-1 - s1-2], [s2-2 - s2-2], [s2-3 - s3-2], ...
#             # ...
#             # The array is 0 on the diagonal.
#             # [s1-2 - s2-1] is the message from atom 2 to atom 1.
#             # Applying the symmetric cutoff_mask leads to a cutoff for those weights at the respective cutoff point.
#             antisym_pass = jnp.multiply(jnp.subtract(elec_ij,elec_ji),cutoff_mask) # possibly * 0.5
#             # print("antisym_pass-Shape:",antisym_pass.shape, "Expected: (natom, natom)")
#             # print("antisym_pass-reduced-Shape:",jnp.sum(antisym_pass, axis=1).shape, "Expected: (105,)")
#             # print("antisym_pass-reduced, expanded-Shape:",jnp.expand_dims(jnp.sum(antisym_pass, axis=1), axis=-1).shape, "Expected: (105,1)")
#             # Summing up over the second axis leads to the charge adaptions in the first row to be for atom 1, in the second row for atom 2 etc.
#             # This is exactly what we want.
#             q += jnp.expand_dims(jnp.sum(antisym_pass, axis=1), axis=-1)
#             return q


class EPN_flax(nn.Module):
    layers: Sequence[int]
    T: int
    def setup(self):
        self.pass_fns = [MLP_flax(self.layers) for i in range(self.T)]

    def __call__(self, h, e, q, cutoff_mask, natom):
        one_timestep_partial = partial(self.one_timestep, h = h, e = e, cutoff_mask = cutoff_mask,natom = natom)
        q = one_timestep_partial(q=q, pass_fn=self.pass_fns[0])
        q = one_timestep_partial(q=q, pass_fn=self.pass_fns[1])
        q = one_timestep_partial(q=q, pass_fn=self.pass_fns[2])
        return q
    # v_one_timestep = jax.vmap(one_timestep)
    def one_timestep(self, q, h, e, cutoff_mask, natom, pass_fn):
            # print("h-Shape:",h.shape)
            # print("q-Shape:",q.shape)
            inp_atom_i = jnp.concatenate((h, q), axis=-1)  # nbatch x natom x 126+1
            inp_i = jnp.tile(jnp.expand_dims(inp_atom_i, axis=1), [1, natom, 1]) # nbatch x natom x natom x 126+1
            inp_j = jnp.transpose(inp_i, [1, 0, 2]) #nbatch x natom x natom x 126+1

            # This should create the inputs for the neural networks.
            # Transposing axis 1 and 2 leads to every single combination of embedding vectors (e.g. for inp_ij:
            # [e_a1, e_a1], [e_a1, e_a2], [e_a1, e_a3], [e_a1, e_a4], ...
            # [e_a2, e_a1], [e_a2, e_a2], [e_a2, e_a3], [e_a2, e_a4], ...
            # [e_a3, e_a1], [e_a3, e_a2], [e_a3, e_a3], [e_a3, e_a4], ...
            # ...
            # and for inp_ji
            # [e_a1, e_a1], [e_a2, e_a1], [e_a3, e_a1], [e_a4, e_a1], ...
            # [e_a1, e_a2], [e_a2, e_a2], [e_a3, e_a2], [e_a4, e_a2], ...
            # [e_a1, e_a3], [e_a2, e_a3], [e_a3, e_a3], [e_a4, e_a3], ...
            # ...
            #
            inp_ij = jnp.concatenate((inp_i, inp_j, e), axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
            inp_ji = jnp.concatenate((inp_j, inp_i, e), axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
            # print("inp-ji-Shape:",inp_ji.shape, "Expected: (105,105,302)")

            # By flattening the inputs to shape (None, 302), we easily pass it as a large batch with each embedding-embedding-concatenation
            # towards the pass_fn, which is a dense network.
            # In the paper, those dense networks are called NN_s 
            # QUESTION: Will the neural network process each combination (h_v, q_v, h_w, q_w, e_wv) individually, if we flatten it beforehand?
            # This would make sense, as the paper also states no influence from other atoms for each specific message between two atoms.
            flat_inp_ij = jnp.reshape(inp_ij, [-1, inp_ij.shape[-1]]) # flatten everything except last axis (nbatch*natom*natom x (126+1)x2 + 126)
            flat_inp_ji = jnp.reshape(inp_ji, [-1, inp_ji.shape[-1]]) # (nbatch*natom*natom x (126+1)x2 + 126)
            # print("flat_inp_ij -Shape:",flat_inp_ij.shape, "Expected: (11025, 302)")
            elec_ij_flat = pass_fn(flat_inp_ij)
            elec_ji_flat = pass_fn(flat_inp_ji)
            # print("elec_ij_flat-Shape:",elec_ij_flat.shape, "Expected: (11025, 1)")
            
            # Now we reshape both arrays back to shape (None (1), 105, 105) which should be both parts of the total message.
            elec_ij = jnp.reshape(elec_ij_flat, [natom, natom]) #reshape back?
            elec_ji = jnp.reshape(elec_ji_flat, [natom, natom]) #reshape back?
            # print("elec_ij-Shape:",elec_ij.shape, "Expected: (natom, natom)")

            # The subtraction of both arrays will lead to the following array:
            # [s1-1 - s1-1], [s1-2 - s2-1], [s1-3 - s3-1], ...
            # [s2-1 - s1-2], [s2-2 - s2-2], [s2-3 - s3-2], ...
            # ...
            # The array is 0 on the diagonal.
            # [s1-2 - s2-1] is the message from atom 2 to atom 1.
            # Applying the symmetric cutoff_mask leads to a cutoff for those weights at the respective cutoff point.
            antisym_pass = jnp.multiply(jnp.subtract(elec_ij,elec_ji),cutoff_mask) # possibly * 0.5
            # print("antisym_pass-Shape:",antisym_pass.shape, "Expected: (natom, natom)")
            # print("antisym_pass-reduced-Shape:",jnp.sum(antisym_pass, axis=1).shape, "Expected: (105,)")
            # print("antisym_pass-reduced, expanded-Shape:",jnp.expand_dims(jnp.sum(antisym_pass, axis=1), axis=-1).shape, "Expected: (105,1)")
            # Summing up over the second axis leads to the charge adaptions in the first row to be for atom 1, in the second row for atom 2 etc.
            # This is exactly what we want.
            q += jnp.expand_dims(jnp.sum(antisym_pass, axis=1), axis=-1)
            return q
        
            

# @jax.jit
def mse_loss(params: optax.Params,
            descriptors : jnp.ndarray,
            distances_encoded : jnp.ndarray,
            init_charges : jnp.ndarray,
            cutoff_mask : jnp.ndarray,
            gt_charges : jnp.ndarray,
            natom : int,
            ) -> jnp.ndarray:
    y_hat = MODEL.apply(params, descriptors, distances_encoded, init_charges, cutoff_mask, natom)
    # optax also provides a number of common loss functions.
    loss_value = optax.l2_loss(y_hat, gt_charges).sum(axis=-1)

    return loss_value.mean()


def fit(params: optax.Params,
        optimizer: optax.GradientTransformation,
        descriptors_train : jnp.ndarray,
        distances_encoded_train : jnp.ndarray,
        init_charges_train : jnp.ndarray,
        cutoff_mask_train : jnp.ndarray,  
        gt_charges_train : jnp.ndarray,
        descriptors_test : jnp.ndarray,
        distances_encoded_test : jnp.ndarray,
        init_charges_test : jnp.ndarray,
        cutoff_mask_test : jnp.ndarray,  
        gt_charges_test : jnp.ndarray,
        n_epochs : int,
        natom : int,
        ) -> optax.Params:
    opt_state = optimizer.init(params)
    train_loss_values = []
    test_loss_values = []


    # @jax.jit
    def step(params, opt_state, descriptors, distances_encoded, init_charges, cutoff_mask, gt_charges, natom):
        train_loss_value, grads = jax.value_and_grad(mse_loss)(params, descriptors, distances_encoded, init_charges, cutoff_mask, gt_charges,natom)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, train_loss_value

    
    for i in range(n_epochs):
        for j in range(descriptors_train.shape[0]):
            params, opt_state, train_loss_value = step(params, opt_state, descriptors_train[j], distances_encoded_train[j], init_charges_train[j], cutoff_mask_train[j], gt_charges_train[j],natom)
            train_loss_values.append(train_loss_value)
        for j in range(descriptors_test.shape[0]):
            test_loss_value = mse_loss(params, descriptors_test[j], distances_encoded_test[j], init_charges_test[j], cutoff_mask_test[j], gt_charges_test[j], natom)
            test_loss_values.append(test_loss_value)
        # if i % 10 == 0:
        print(f'step {i}, train-loss: {train_loss_value}, test-loss: {test_loss_value}')

    return params

if __name__ == "__main__":
    key, subkey = random.split(random.PRNGKey(0))
    h_dim = 126
    e_dim = 48
    layers = [32, 32, 1] # hidden layers
    T = 3
    path = "data/SrTiO3_500.db"
    EPOCHS = 30
    SAMPLE_SIZE = 100
    # 500 crystals in Dataset
    ###############################
    # descriptors: bessel descriptors for atoms (nbatch x natom x 126)
    # distances_encoded: Distance encodings (nbatch x natom x natom x edim)
    # init_charges: initial charge encodings (nbatch x natom)
    # gt_charges: ground truth charge encodings (nbatch x natom)
    # cutoff_mask: mask with cutoff for specific distances (nbatch x natom x natom)

    descriptors, distances_encoded, init_charges, gt_charges, cutoff_mask = get_init_crystal_states(path, edge_encoding_dim = e_dim, SAMPLE_SIZE = SAMPLE_SIZE) # Change sample size to None if all samples should be read.
    natom = descriptors.shape[1]
    MODEL = EPN_flax(layers, T)
    key, subkey = random.split(subkey)
    descriptors_train, descriptors_test, distances_encoded_train, distances_encoded_test, init_charges_train, init_charges_test, gt_charges_train, gt_charges_test, cutoff_mask_train, cutoff_mask_test = train_test_split(descriptors, distances_encoded, init_charges, gt_charges, cutoff_mask, test_size=0.2, random_state=42)
    params = MODEL.init(key, descriptors_train[0], distances_encoded_train[0], init_charges_train[0], cutoff_mask_train[0], natom)
    optimizer = optax.adam(learning_rate=1e-2)

    params = fit(params,optimizer,descriptors_train,
        distances_encoded_train,
        init_charges_train,
        cutoff_mask_train,  
        gt_charges_train,
        descriptors_test,
        distances_encoded_test,
        init_charges_test,
        cutoff_mask_test,  
        gt_charges_test,
        n_epochs=EPOCHS,
        natom=natom)

    print("Something.")

    





    # for epoch in range(EPOCHS):
    #     train_loss.reset_states()
    #     train_acc.reset_states()
    #     test_loss.reset_states()
    #     test_acc.reset_states()
    #     train_preds = []
    #     test_preds = []
    #     # for each molecule in train set
    #     for i in range(len(yt)):
    #         hb = np.array(np.expand_dims(ht[i], axis=0))
    #         eb = np.array(np.expand_dims(et[i], axis=0))
    #         qb = np.array(np.expand_dims(qt[i], axis=0))
    #         yb = np.array(np.expand_dims(yt[i], axis=0))
    #         cutoff_maskb = np.array(np.expand_dims(cutoff_maskt[i], axis=0))
    #         train_preds.append(train_step(hb, eb, qb, yb, cutoff_maskb))
    #     # for each molecule in test set
    #     for i in range(len(ye)):
    #         hb = np.array(np.expand_dims(he[i], axis=0))
    #         eb = np.array(np.expand_dims(ee[i], axis=0))
    #         qb = np.array(np.expand_dims(qe[i], axis=0))
    #         yb = np.array(np.expand_dims(ye[i], axis=0))
    #         cutoff_maskb = np.array(np.expand_dims(cutoff_maske[i], axis=0))
    #         test_preds.append(test_step(hb, eb, qb, yb, cutoff_maskb))
    #     if test_acc.result() < best_test_acc:
    #         best_test_acc = test_acc.result()
    #         model.save_weights('models/model_weights')
    #     #    model.save(f'models/model')
    #         np.save("train_pred_charges.npy", np.array(np.squeeze(train_preds)))
    #         np.save("train_lab_charges.npy", np.squeeze(yt))
    #         np.save("test_pred_charges.npy", np.array(np.squeeze(test_preds)))
    #         np.save("test_lab_charges.npy", np.squeeze(ye))

    #     template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
    #     print(template.format(epoch, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()))