import os
from re import S
import scipy
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import preprocessing

atom_num_dict = {'O': 6, 'Sr': 38, 'Ti': 22}
elem_dict = {'O': 0, 'Sr': 1, 'Ti': 2}

class MLP_layer(tf.keras.layers.Layer):
    def __init__(self, nodes, out_dim=1, activation='relu'):
        ''' Simple Multi-Level perceptron to create DNN.

            nodes: array of integers for output sizes of dense layers
            out_dim: final output dimension
            activation: activation function string
        '''
        super(MLP_layer, self).__init__()
        self.nodes = nodes
        self.layer_set = []
        self.out_dim = out_dim
        self.activation = activation
        for num in nodes:
            self.layer_set.append(Dense(num, activation=activation))
        self.layer_set.append(Dense(out_dim, activation=None))

    @tf.function(experimental_relax_shapes=True)
    def call(self, x):
        ''' Calls each layer onto the input.'''
        for layer in self.layer_set:
            x = layer(x)
        return x

class EPN_layer(tf.keras.layers.Layer):
    """Special 'Electron Passing Network,' which retains conservation of electrons but allows non-local passing."""

    def __init__(self, pass_fn,layers, T=1):
        '''
        Inputs:
            pass_fn (MLP_layer): electron_model, each time step has its own neural network.
            T (integer): number of time steps to run this model. Hyperparameter.
        '''
        super(EPN_layer, self).__init__()
        self.pass_fns = []
        for t in range(T):
            self.pass_fns.append(pass_fn([32,32]))
        self.T = T

    @tf.function(experimental_relax_shapes=True)
    def call(self, h, e, q, cutoff_mask):
        """
        Inputs:
            h: Node descriptors - # nmol (Batch size) x natom x h_dim (Dimensions of Bessel descriptors)
            e: Edge distances # nmol (Batch size) x natom x natom x e_dim (Dimensions of Bessel descriptors)
            q: Atomic charges - # nmol x natom x 1
            cutoff_mask: # nmol x natom x natom
        """
        natom = e.shape[1]

        for t in range(self.T):
            self.pass_fn = self.pass_fns[t]
            # print("h-Shape:",h.shape)
            # print("q-Shape:",q.shape)
            inp_atom_i = tf.concat([h, q], axis=-1)  # nbatch x natom x 126+1
            inp_i = tf.tile(tf.expand_dims(inp_atom_i, axis=2), [1, 1, natom, 1]) # nbatch x natom x natom x 126+1
            inp_j = tf.transpose(inp_i, [0, 2, 1, 3]) #nbatch x natom x natom x 126+1

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
            inp_ij = tf.concat([inp_i, inp_j, e], axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
            inp_ji = tf.concat([inp_j, inp_i, e], axis=-1) #nbatch x natom x natom x (126+1)x2 + 48
            # print("inp-ji-Shape:",inp_ji.shape)

            # By flattening the inputs to shape (None, 302), we easily pass it as a large batch with each embedding-embedding-concatenation
            # towards the pass_fn, which is a dense network.
            # In the paper, those dense networks are called NN_s 
            # QUESTION: Will the neural network process each combination (h_v, q_v, h_w, q_w, e_wv) individually, if we flatten it beforehand?
            # This would make sense, as the paper also states no influence from other atoms for each specific message between two atoms.
            flat_inp_ij = tf.reshape(inp_ij, [-1, inp_ij.shape[-1]]) # flatten everything except last axis (nbatch*natom*natom x (126+1)x2 + 126)
            flat_inp_ji = tf.reshape(inp_ji, [-1, inp_ji.shape[-1]]) # (nbatch*natom*natom x (126+1)x2 + 126)
            # print("flat_inp_ij -Shape:",flat_inp_ij.shape, "Expected: (None, 302)")

            elec_ij_flat = self.pass_fn(flat_inp_ij)
            elec_ji_flat = self.pass_fn(flat_inp_ji)
            # print("elec_ij_flat-Shape:",elec_ij_flat.shape, "Expected: (11025, 1)")
            
            # Now we reshape both arrays back to shape (None (1), 105, 105) which should be both parts of the total message.
            elec_ij = tf.reshape(elec_ij_flat, [-1, natom, natom]) #reshape back?
            elec_ji = tf.reshape(elec_ji_flat, [-1, natom, natom]) #reshape back?
            # print("elec_ij-Shape:",elec_ij.shape, "Expected: (None, natom, natom)")
            # print("elec_ji-Shape:",elec_ji.shape, "Expected: (None, natom, natom)")
            # print("elec_subtract-Shape:",(elec_ij - elec_ji).shape, "Expected: (None, natom, natom)")



            # The subtraction of both arrays will lead to the following array:
            # [s1-1 - s1-1], [s1-2 - s2-1], [s1-3 - s3-1], ...
            # [s2-1 - s1-2], [s2-2 - s2-2], [s2-3 - s3-2], ...
            # ...
            # The array is 0 on the diagonal.
            # [s1-2 - s2-1] is the message from atom 2 to atom 1.
            # Applying the symmetric cutoff_mask leads to a cutoff for those weights at the respective cutoff point.
            antisym_pass = (elec_ij - elec_ji) * cutoff_mask # possibly * 0.5
            # print("antisym_pass-Shape:",antisym_pass.shape, "Expected: (None, natom, natom, 1)")
            # print("antisym_pass-reduced-Shape:",tf.reduce_sum(antisym_pass, axis=2).shape, "Expected: (None, natom, natom, 1)")
            # print("antisym_pass-reduced, expanded-Shape:",tf.expand_dims(tf.reduce_sum(antisym_pass, axis=2), axis=-1).shape, "Expected: (None, natom, natom, 1)")
            # print("Reduced_sum:",tf.get_static_value(tf.math.reduce_sum(antisym_pass-tf.transpose(antisym_pass,[0, 2, 1]))))

            # Summing up over the second axis leads to the charge adaptions in the first row to be for atom 1, in the second row for atom 2 etc.
            # This is exactly what we want.
            q += tf.expand_dims(tf.reduce_sum(antisym_pass, axis=2), axis=-1)
        return q

def make_model(layers, h_dim, e_dim, T, natom):#mask, natom):                    # mask: nmol x natom
    electron_model = MLP_layer # layers = [32,32] but passed inside of EPN_layer
    electron_net = EPN_layer(electron_model, layers = layers, T=T)
    # h_dim = embedding size of nodes (Bessel descriptors)
    # e_dim = embedding size of distances (Gaussian?)
    h_inp = tf.keras.Input(shape=(natom, h_dim), dtype='float32', name='h_inp')                 # nbatch x natom x h_dim -> Flattened Bessel descriptors
    e_inp = tf.keras.Input(shape=(natom, natom, e_dim), dtype='float32', name='e_inp')          # nbatch x natom x natom x e_dim -> Dim of distance embedding
    q_inp = tf.keras.Input(shape=(natom, 1), dtype='float32', name='q_inp')                     # nbatch x natom x  1
    cutoff_mask_inp = tf.keras.Input(shape=(natom, natom), dtype='float32', name='mask_inp') # nbatch x natom x natom x 1

    # Prediction from electron passing neural network.
    q_pred = electron_net(h_inp, e_inp, q_inp, cutoff_mask_inp)                                 # nbatch x natom x 1
    model = tf.keras.Model(inputs=[h_inp, e_inp, q_inp, cutoff_mask_inp], outputs=q_pred)

    return model

@tf.function(experimental_relax_shapes=True)
def train_step(h, e, q, y, cutoff_mask):
    with tf.GradientTape() as tape:
        predictions = model([h, e, q, cutoff_mask])
        loss = tf.keras.losses.MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(predictions, y)
    return predictions

@tf.function(experimental_relax_shapes=True)
def test_step(h, e, q, y, cutoff_mask):
    predictions = model([h, e, q, cutoff_mask])
    t_loss = tf.keras.losses.MSE(y, predictions)
    test_loss(t_loss)
    test_acc(predictions, y)
    return predictions

if __name__ == "__main__":
    h_dim = 126
    e_dim = 48
    layers = [32, 32] # hidden layers
    T = 5
    path = "data/SrTiO3_500.db"
    n_elems = 3
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.MeanAbsoluteError(name='train_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.MeanAbsoluteError(name='test_acc')
    EPOCHS = 100
    best_test_acc = np.inf

    # 500 crystals in Dataset
    ###############################
    # x: element encodings for nodes (nbatch x natom x natom x nelem)
    # h: node encodings (nbatch x natom x natom x hdim)
    # q: charge encodings (nbatch x natom x natom x 1)
    # e: Distance encodings (nbatch x natom x natom x hdim)
    # Q: Total charge of system (nbatch x 1)
    # y: Ground truth: labels to predict (charges for each atom) (nbatch x natom x 1)
    # mask: 
    # x, h, q, e, Q, y, mask, names = get_init_crystal_states(path)
    preprocessed_dict = preprocessing.get_init_crystal_states(path, edge_encoding_dim = e_dim, SAMPLE_SIZE = None) # Change sample size to None if all samples should be read.
    natom = preprocessed_dict["natom"]
    # # x is a concatenation of element numbers and one-hot-encoded element type
    # x = jnp.tile(jnp.expand_dims(jnp.concatenate([jnp.expand_dims(preprocessed_dict["atomic_numbers"],axis=-1),preprocessed_dict["ohe_types"]],axis=2),axis=2),(1,1,natom,1))
    # h = jnp.tile(jnp.expand_dims(preprocessed_dict["descriptors"],axis=2),(1,1,natom,1))
    # q = jnp.expand_dims(jnp.tile(jnp.expand_dims(preprocessed_dict["init_charges"],axis=-1),(1,1,natom)),axis=-1)
    # e = preprocessed_dict["distances_encoded"]
    # Q = preprocessed_dict["total_charges"]
    # y = jnp.expand_dims(preprocessed_dict["gt_charges"],axis=-1)

    
    h = preprocessed_dict["descriptors"]
    q = jnp.expand_dims(preprocessed_dict["init_charges"], axis=-1)
    e = preprocessed_dict["distances_encoded"]
    Q = preprocessed_dict["total_charges"]
    y = jnp.expand_dims(preprocessed_dict["gt_charges"],axis=-1)
    cutoff_mask = preprocessed_dict["cutoff_mask"]
    # cutoff_mask = jnp.expand_dims(preprocessed_dict["cutoff_mask"],axis=-1)
    
    model = make_model(layers, h_dim, e_dim, T, h.shape[1])

    ht, he, qt, qe, et, ee, Qt, Qe, yt, ye, cutoff_maskt, cutoff_maske = train_test_split(h,q,e,Q,y,cutoff_mask, test_size=0.2, random_state=42)
    
    # np.save("train_names.npy", namest, allow_pickle=True)
    # np.save("val_names.npy", namese, allow_pickle=True)

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()
        train_preds = []
        test_preds = []
        # for each molecule in train set
        for i in range(len(yt)):
            hb = np.array(np.expand_dims(ht[i], axis=0))
            eb = np.array(np.expand_dims(et[i], axis=0))
            qb = np.array(np.expand_dims(qt[i], axis=0))
            yb = np.array(np.expand_dims(yt[i], axis=0))
            cutoff_maskb = np.array(np.expand_dims(cutoff_maskt[i], axis=0))
            train_preds.append(train_step(hb, eb, qb, yb, cutoff_maskb))
        # for each molecule in test set
        for i in range(len(ye)):
            hb = np.array(np.expand_dims(he[i], axis=0))
            eb = np.array(np.expand_dims(ee[i], axis=0))
            qb = np.array(np.expand_dims(qe[i], axis=0))
            yb = np.array(np.expand_dims(ye[i], axis=0))
            cutoff_maskb = np.array(np.expand_dims(cutoff_maske[i], axis=0))
            test_preds.append(test_step(hb, eb, qb, yb, cutoff_maskb))
        if test_acc.result() < best_test_acc:
            best_test_acc = test_acc.result()
            model.save_weights('models/model_weights')
        #    model.save(f'models/model')
        #    #tf.saved_model.save(model(he[-1], ee[-1], xe[-1], qe[-1]), 'models/model')
            np.save("train_pred_charges.npy", np.array(np.squeeze(train_preds)))
            np.save("train_lab_charges.npy", np.squeeze(yt))
            np.save("test_pred_charges.npy", np.array(np.squeeze(test_preds)))
            np.save("test_lab_charges.npy", np.squeeze(ye))

        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()))



######################################################################################################
######################################################################################################
########################## ARCHIVED FUNCTIONS WITH COMMENTS ##########################################
######################################################################################################
######################################################################################################


# def get_init_edges(xyz, molecular_splits, num=32, cutoff=3.0, eta=2.0):
#     """
#     Creates multi-dimensionally encoded edges and soft-mask from coordinates of atoms, number of molecular splits, cutoff etc.
#     Input:
#         - xyz: np.array (n_atom x 3) -> Three-dimensional corrdinates.
#         - molecular_splits: 
#         - num: int -> number of dimensions for distance encoding
#     """
#     # get (num) evenly distributed numbers between 0.1 & cutoff to create different gaussian distributions.
#     mu = np.linspace(0.1, cutoff, num=num)
#     # Distance matrix of all atoms to all other atoms 
#     D = scipy.spatial.distance_matrix(xyz,xyz)

#     if molecular_splits.shape == (0,):
#         adj = np.ones(D.shape)
#     elif molecular_splits.shape == ():
#         # Make outer product of two vectors e.g. [1,1,1,0,0,0]x[0,0,0,1,1,1]
#         molec_vecA = np.zeros(D.shape[0])
#         molec_vecA[:molecular_splits] = 1
#         molec_vecB = np.zeros(D.shape[0])
#         molec_vecB[molecular_splits:] = 1
#         # adj = adjacency matrix. (Not needed in Crystal graphs as all elements are deemed to be adjacent to each other)
#         adj = np.outer(molec_vecA, molec_vecA.T) + np.outer(molec_vecB, molec_vecB.T)
#     else:
#         adj = np.zeros(D.shape)
#         prev_split = 0
#         for i, split in enumerate(molecular_splits):
#             molec_vec = np.zeros(D.shape[0])
#             molec_vec[prev_split:split] = 1
#             # print(molec_vec)
#             molec_mat = np.outer(molec_vec, molec_vec.T)
#             # print(molec_mat)
#             adj += molec_mat
#         # print(adj)
#         exit()
#     # Insert a new axis at the end
#     adj = np.expand_dims(adj, -1)
#     # Calculate C (Soft-Mask) out of Distance Matrix
#     C = (np.cos(np.pi * (D - 0.0) / cutoff) + 1.0) / 2.0
#     C[D >= cutoff] = 0.0
#     C[D <= 0.0] = 1.0
#     np.fill_diagonal(C, 0.0)
#     D = np.expand_dims(D, -1)
#     D = np.tile(D, [1, 1, num])
#     C = np.expand_dims(C, -1)
#     C = np.tile(C, [1, 1, num])
#     mu = np.expand_dims(mu, 0)
#     mu = np.expand_dims(mu, 0)
#     mu = np.tile(mu, [D.shape[0], D.shape[1], 1])
#     # This is where the gaussian encodings for the distances are calculated
#     e = C * np.exp(-eta * (D-mu)**2)
#     e = np.array(e, dtype=np.float32)

#     return e, C

# def gen_init_state(path,  h_dim, e_dim):
#     x = []
#     h = []
#     q = []
#     Q = []
#     e = []
#     y = []
#     for filename in os.listdir(path):
#         if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
#             splits_path = path + filename[:-4] + "splits.npy"
#             if os.path.exists(splits_path):
#                 splits = np.load(splits_path)
#             else: splits = np.array([])
#             xyzfile = open(path + filename, 'r')
#             lines = xyzfile.readlines()
#             y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float32))
#             Q.append(np.array(lines[1].strip().split()[0], dtype=np.float32))
#             xyz = []
#             this_x = []
#             for line in lines[2:]:
#                 data = line.split()
#                 elem_name = data[0]
#                 xyz.append([data[1], data[2], data[3]])
#                 ohe = np.zeros(len(elem_dict)+1)
#                 ohe[0] = atom_num_dict[elem_name]
#                 ohe[elem_dict[elem_name] + 1] = 1
#                 this_x.append(ohe)
#             this_x = np.array(this_x)
#             xyz = np.array(xyz, dtype=np.float32)
#             x.append(np.array(this_x, dtype=np.float32))
#             h.append(np.zeros((this_x.shape[0], h_dim), dtype=np.float32))
#             avg_q = Q[-1] / len(this_x)
#             q.append(np.array(np.ones(len(this_x)) * avg_q, dtype=np.float32))
#             e.append(get_init_edges(xyz, splits, num=e_dim))
#     return x, h, q, e, Q, y

# def gen_flat_init_state(path,  h_dim, e_dim):
#     x = []
#     h = []
#     q = []
#     Q = []
#     e = []
#     y = []
#     for filename in os.listdir(path):
#         if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
#             xyzfile = open(path + filename, 'r')
#             lines = xyzfile.readlines()
#             y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float32))
#             Q.append(np.array(lines[1].strip().split()[0], dtype=np.float32))
#             xyz = []
#             this_x = []
#             for line in lines[2:]:
#                 data = line.split()
#                 elem_name = data[0]
#                 xyz.append([data[1], data[2], data[3]])
#                 ohe = np.zeros(len(elem_dict)+1)
#                 ohe[0] = atom_num_dict[elem_name]
#                 ohe[elem_dict[elem_name] + 1] = 1
#                 this_x.append(ohe)
#             this_x = np.array(this_x)
#             xyz = np.array(xyz, dtype=np.float32)
#             these_edges = get_init_edges(xyz, num=e_dim)
#             e.append(these_edges.reshape((-1, these_edges.shape[-1])))

#             x.append(np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1)))
#             h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float32), (these_edges.shape[0], 1)))
#             avg_q = Q[-1] / len(this_x)
#             q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float32), (these_edges.shape[0], 1)))

#     return x, h, q, e, Q, y

# def gen_flat_padded_init_state(path,  h_dim, e_dim):
#     x = []
#     h = []
#     q = []
#     Q = []
#     e = []
#     y = []
#     for filename in os.listdir(path):
#         if os.path.exists(path + filename[:-4] + ".npy") and filename.endswith(".xyz"):
#             xyzfile = open(path + filename, 'r')
#             lines = xyzfile.readlines()
#             y.append(np.array(np.load(path + filename[:-4] + '.npy'), dtype=np.float32))
#             Q.append(np.array(lines[1].strip().split()[0], dtype=np.float32))
#             xyz = []
#             this_x = []
#             for line in lines[2:]:
#                 data = line.split()
#                 elem_name = data[0]
#                 xyz.append([data[1], data[2], data[3]])
#                 ohe = np.zeros(len(elem_dict)+1)
#                 ohe[0] = atom_num_dict[elem_name]
#                 ohe[elem_dict[elem_name] + 1] = 1
#                 this_x.append(ohe)
#             this_x = np.array(this_x)
#             xyz = np.array(xyz, dtype=np.float32)
#             these_edges = get_init_edges(xyz, num=e_dim)
#             e.append(these_edges.reshape((-1, these_edges.shape[-1])))

#             x.append(np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1)))
#             h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float32), (these_edges.shape[0], 1)))
#             avg_q = Q[-1] / len(this_x)
#             q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float32), (these_edges.shape[0], 1)))
    
#     largest_system = np.max([y[i].shape[0] for i in range(len(y))])
#     x_padded = np.zeros((len(Q), largest_system**2, x[0].shape[1]))
#     h_padded = np.zeros((len(Q), largest_system**2, h[0].shape[1]))
#     q_padded = np.zeros((len(Q), largest_system**2, q[0].shape[1]))
#     e_padded = np.zeros((len(Q), largest_system**2, e[0].shape[1]))
#     y_padded = np.zeros((len(Q), largest_system, y[0].shape[1]))
#     pad = np.zeros((len(Q)))

#     for i in range(x_padded.shape[0]):
#         for j in range(y[i].shape[0]):
#             y_padded[i][j] = y[i][j]
#             pad[i] = j
#         for j in range(x[i].shape[0]):
#             for k in range(x[i][j].shape[0]):
#                 x_padded[i][j][k] = x[i][j][k]
#             for k in range(h[i][j].shape[0]):
#                 h_padded[i][j][k] = h[i][j][k]
#             for k in range(q[i][j].shape[0]):
#                 q_padded[i][j][k] = q[i][j][k]
#             for k in range(e[i][j].shape[0]):
#                 e_padded[i][j][k] = e[i][j][k]
#     return x_padded, h_padded, q_padded, e_padded, Q, y_padded, pad

# def gen_padded_init_state(path,  h_dim, e_dim):
#     """
#     Create a padded initial state. 
#     Input:
#         - path: PATH to all the molecules in xyz format
#         - h_dim: Dimension for node encoding
#         - e_dim: Dimension for distance encoding (Gaussian encoding over different dimensions)
#     """
#     # First molecule has 16 atoms!
#     # e_dim = 48
#     # h_dim = 48
#     # 4379 molecules in Dataset
#     x = [] # element encodings for nodes
#     h = [] # node encodings
#     q = [] # charge encodings
#     Q = [] # Total charge of system
#     e = [] # Distance encodings
#     y = [] # Ground truth: labels to predict (charges for each atom)
#     soft_mask = []
#     names = []
#     for filename in os.listdir(path):
#         label_file = path + filename[:-4] + '.npy'
#         #if os.path.exists(label_file) and filename.endswith(".xyz"):
#         if filename.endswith(".xyz"):
#             splits_path = path + filename[:-4] + "splits.npy"
#             if os.path.exists(splits_path):
#                 splits = np.load(splits_path)
#             else: splits = np.array([])
#             xyzfile = open(path + filename, 'r')
#             lines = xyzfile.readlines()
#             label_file = path + filename[:-4] + '.npy'
#             if os.path.exists(label_file):
#                 y.append(np.array(np.load(label_file), dtype=np.float32))
#             else:
#                 print('No labels provided, y set to 0')
#                 y.append(np.zeros(len(lines)-2))

#             # Q = list of global charges for each molecule in directory (?)
#             Q.append(np.array(lines[1].strip().split()[0], dtype=np.float32))
#             # names = names of all molecules in directory (?)
#             names.append(filename[:-4])

#             # xyz are coordinates for each atom in the molecule
#             xyz = []
#             # this_x = list of one-hot-encoded lists for atom numbers
#             this_x = []

#             for line in lines[2:]:
#                 data = line.split()
#                 elem_name = data[0]
#                 xyz.append([data[1], data[2], data[3]])

#                 # ohe (=) one_hot_encoded in structure [atom number, (one-hot-encoding)]
#                 ohe = np.zeros(len(elem_dict)+1)
#                 ohe[0] = atom_num_dict[elem_name]
#                 ohe[elem_dict[elem_name] + 1] = 1
#                 this_x.append(ohe)

#             this_x = np.array(this_x)
#             xyz = np.array(xyz, dtype=np.float32)
#             # these_edges = edge encodings
#             # this_soft_mask = mask with 0 for cutoff and low values for 
#             these_edges, this_soft_mask = get_init_edges(xyz, splits, num=e_dim)
#             e.append(these_edges.reshape((-1, these_edges.shape[-1])))
#             soft_mask.append(this_soft_mask)

#             # Element encodings are repeated to have the same dimension as e (256, 10)
#             x.append(np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1)))
#             # Node embeddings start as zeroes and are repeated to have the same dimension as e (256, 48)
#             h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float32), (these_edges.shape[0], 1)))
#             # average charge for each atom as initialisation
#             avg_q = Q[-1] / len(this_x)
#             q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float32), (these_edges.shape[0], 1)))

#     # Largest system is size of system with most atoms: 41
#     largest_system = np.max([y[i].shape[0] for i in range(len(y))])
#     # Create padding vectors based on this size to have same-sized vectors to be fed into the model
#     x_padded = np.zeros((len(Q), largest_system, largest_system,  x[0].shape[1]))
#     h_padded = np.zeros((len(Q), largest_system, largest_system, h[0].shape[1]))
#     q_padded = np.zeros((len(Q), largest_system, largest_system, q[0].shape[1]))
#     e_padded = np.zeros((len(Q), largest_system, largest_system, e[0].shape[1]))
#     soft_mask_padded = np.zeros((len(Q), largest_system, largest_system, 1))
#     y_padded = np.zeros((len(Q), largest_system, 1))
#     pad_n = np.zeros((len(Q)))
#     mask = np.zeros((len(Q), largest_system, largest_system))

#     # For each molecule
#     for i in range(x_padded.shape[0]):
#         # Actual molecule size
#         molec_size = np.sqrt(x[i].shape[0]).astype(np.int32)
#         # For each atom index in the molecule, fill in the ground truth charges
#         for j in range(y[i].shape[0]):
#             y_padded[i][j] = y[i][j]
#             #soft_mask_padded[i][j] = soft_mask[i][j]
#             pad_n[i] = j
#         # Fill in the values for the other vectors
#         for j in range(molec_size):
#             for k in range(molec_size):
#                 x_padded[i][j][k] = x[i][j*molec_size + k]
#                 h_padded[i][j][k] = h[i][j*molec_size + k]
#                 q_padded[i][j][k] = q[i][j*molec_size + k]
#                 e_padded[i][j][k] = e[i][j*molec_size + k]
#                 # Mask is just the padding mask
#                 mask[i][j][k] = 1

#     return x_padded, h_padded, q_padded, e_padded, Q, y_padded, mask, np.array(names)


# class GNN_layer(tf.keras.layers.Layer):
#     def __init__(self, message_fn, update_fn, T):
#         ''' Graph Neural Network Layer

#             message_fn: TODO ???
#             update_fn: TODO ???
#             T: Number of timesteps to pass the messages
#         '''
#         super(GNN_layer, self).__init__()
#         self.message_fns = []
#         for i in range(T):
#             self.message_fns.append(message_fn([32,32], out_dim=32))
#         self.update_fn = update_fn
#         self.T = T

#     @tf.function(experimental_relax_shapes=True)
#     def call(self, h, e, x, q, mask):
#         '''
#         '''
#         # Number of atoms
#         natom = e.shape[1]
#         node_mask = tf.clip_by_value(tf.reduce_sum(mask, axis=1), clip_value_min=0, clip_value_max=1)
#         for t in range(self.T):
#             self.message_fn = self.message_fns[t]
#             inp_atom_i = tf.concat([x, h, q], axis=-1)  # nmolec x natom x 9+32+1
#             inp_i = tf.tile(tf.expand_dims(inp_atom_i, axis=2), [1, 1, natom, 1]) # nmolec x natom x natom x 9+32+1
#             inp_j = tf.transpose(inp_i, [0, 2, 1, 3]) #nmolec x natom x natom x 9+32+1
#             inp_ij = tf.concat([inp_i, inp_j, e], axis=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
#             flat_inp_ij = tf.reshape(inp_ij, (-1, inp_ij.shape[-1]))#tf.shape(inp_ij)[-1])) #nmolec * natom**2 x 9*2+32*2+1*2+32

#             flat_pair_messages = self.message_fn(flat_inp_ij)
#             pair_messages = tf.reshape(flat_pair_messages, (-1, natom, natom, 32))
#             messages = tf.reduce_sum(pair_messages, axis=2)
#             update_input = tf.concat([h, messages], axis=2)
#             masked_input = update_input * node_mask
#             h = self.update_fn(masked_input)
#             h = h * node_mask
#         return h