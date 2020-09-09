import os
import scipy
import numpy as np
import tensorflow as tf
import charge_gn
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

atom_num_dict = {'H' : 1,
             'C' : 6,
             'N' : 7,
             'O' : 8,
             'F' : 9,
             'S' : 16,
             'Cl': 17,
             'Br': 35,
             }
elem_dict = {'H' : 0,
             'C' : 1,
             'N' : 2,
             'O' : 3,
             'F' : 4,
             'S' : 5,
             'Cl': 6,
             'Br': 7,
             }

@tf.function(experimental_relax_shapes=True)
def test_step(h, e, x, q, y, mask):
    predictions = model([h, e, x, q, mask])
    return predictions

if __name__ == "__main__":
    h_dim = 48
    e_dim = 48
    layers = [32, 32]
    T = 5
    path = '' # path of desired model inputs
    n_elems = 9
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.MeanAbsoluteError(name='train_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.MeanAbsoluteError(name='test_acc')
    EPOCHS = 200
    best_test_acc = np.inf
    
    timeA = time.time()
    x, h, q, e, Q, y, mask, names = charge_gn.gen_padded_init_state(path, h_dim, e_dim)
    timeB = time.time()

    model = charge_gn.make_model(layers, h_dim, T, n_elems, x.shape[1])
    model.load_weights('./models/decay_model_weights')

    np.save("test_names.npy", names, allow_pickle=True)
    
    test_preds = []
    for i in range(len(x)):
        hb = np.array(np.expand_dims(h[i], axis=0))
        eb = np.array(np.expand_dims(e[i], axis=0))
        xb = np.array(np.expand_dims(x[i], axis=0))
        qb = np.array(np.expand_dims(q[i], axis=0))
        yb = np.array(np.expand_dims(y[i], axis=0))
        maskb = np.array(np.expand_dims(mask[i], axis=0))
        #train_preds.append(train_step(ht[i], et[i], xt[i], qt[i], yt[i], maskt[i]))
        timeC = time.time()
        for j in range(repeats):
            inf_1 = time.time()
            test_preds.append(test_step(hb, eb, xb, qb, yb, maskb))
            inf_2 = time.time()
            print(inf_2-inf_1)
        timeD = time.time()

    print(f"avg inference time: {(timeD-timeC)/repeats}")
    print(f"avg feature time:{(timeB-timeA)}")
