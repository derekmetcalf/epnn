# Electron-Passing Neural Networks with jraph


### Current Status:
- runs the EPNN Algorithm with jraph, haiku (instead of flax) and jax functionalities.
- efficiency improvements in the batch function (2.2s for all 500 samples).
- comparison of model performance during training for training and val set.

### ToDo
- End-2-End Training-Pipeline with batch-loading to use whole training-set
- Change activation function? Swish-1?
- Other possibilities for Edge features: normal distances / logarithmic distances / (1/d)^x as variants
- Play around with network size, depth, number of timesteps, larger training size
 master
 - Write user guide
 - Write report
