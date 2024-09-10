Studying phase of q-state clock model with MLP:

![image](https://github.com/user-attachments/assets/316d79eb-4c9d-47f0-95c9-b4528d0e2c1e)

The training data is unusual and knows nothing about the actual model. We feed two 200 copies of two vectors with a collinear 1d array of vectors, i.e. [11111111...] and [000000...] with
the one-hot encoding defined as [0] -> [1 0] and [1] -> [0 1].

The first layer is a 400-node layer, and there are also two hidden layers, one with ReLu and the other with SoftMax activation.

The model gives predictions for the Wollf algorithm simulated for L number of sites. 




Prediction from the linear classifier based on number of 1 and 0 in the string:
![Figure_1](https://github.com/user-attachments/assets/5451e407-7a00-44ed-bab9-67702392510f)
Prediction of the MLP model:
![Figure_2](https://github.com/user-attachments/assets/b2171307-0a1c-4f6a-bcce-8cb159cf18ff)
