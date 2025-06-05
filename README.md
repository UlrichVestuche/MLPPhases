Studying phases of q-state clock model with MLP:

![image](https://github.com/user-attachments/assets/316d79eb-4c9d-47f0-95c9-b4528d0e2c1e)


The 2-state Potts model is also known as Ising model has a well-defined phase transition (Kramers-Wannier dual point).

The 6-state Potts model ...

The training data is unusual and knows nothing about the actual model. We feed two 200 copies of two vectors with a collinear 1d array of vectors, i.e. [11111111...] and [000000...] with
the one-hot encoding defined as [0] -> [1 0] and [1] -> [0 1].

The first layer is a 400-node layer, and there are also two hidden layers, one with ReLu and the other with SoftMax activation.

The model gives predictions for the Wollf algorithm simulated for L number of sites. 




Prediction from the linear classifier based on number of 1 and 0 in the string:
![Figure_1](https://github.com/user-attachments/assets/5451e407-7a00-44ed-bab9-67702392510f)

Prediction of the MLP model:

![MagTem100draws](https://github.com/user-attachments/assets/2e18e745-55ad-43a6-95a6-c37fae753788)


Compare these predictions with unsupervised learning PCA + k-clustering.

[paper](https://arxiv.org/abs/2112.06735)
