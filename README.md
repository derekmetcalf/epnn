# Electron-Passing Neural Networks
Partner code to "Electron-Passing Neural Networks for Atomic Charge Prediction in Systems with Arbitrary Molecular Charge"
![protein_image](/assets/protein_fig.png)

**(A.)** A comparison between reference MBIS charges and predicted EPNN charges on an anionic fragment of Galectin 3C. Blue indicates negatively charged atoms and red indicates positively charged. **(B.)** EPNN charge predictions on the entire 2220-atom Galectin 3C protein, net charge +2. The fragment from **A** is backlit to show similarity in charge prediction despite system size and charge changes.

EPNNs are an extensible framework to encode graph-level conservation properties (such as conservation of charge in molecular systems) while operating only locally on portions of the graph. Use EPNNs for cheminformatics, force field charges, or as input to neural network potentials. Train and test your own EPNNs on your systems. Requires TensorFlow 2.x, numpy, scipy, and sklearn.
