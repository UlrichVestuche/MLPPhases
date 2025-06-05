Studying Phases of the q-State Clock Model with an MLP
---

This repository demonstrates how a simple multilayer perceptron (MLP) can learn phase transitions in Potts- and clock-type spin models, using only minimally informed input data.

## Overview

- **Models considered**  
  - **2-state Potts model (Ising)**: Exhibits a well-known phase transition at the Kramers–Wannier dual point.  
  - **6-state Potts model**: A richer spin space with a more complex phase structure.

- **Key idea**  
  Instead of providing explicit spin configurations or Hamiltonian details, we feed the MLP only “flat” vectors consisting of all 1’s or all 0’s (of length 200) with a simple one-hot encoding:  
  ```
  [0] → [1 0]  
  [1] → [0 1]
  ```
  In other words, the network sees only uniform binary strings labeled by whether they came from a “high‐temperature” (disordered) or “low‐temperature” (ordered) ensemble, without direct knowledge of the underlying physics.

## Data Preparation

1. **Simulation**  
   - We generate spin configurations using the Wolff algorithm on an L × L lattice (with L chosen per experiment).  
   - Each configuration is reduced to two 1D vectors of length 200 by counting the number of up (1) and down (0) spins at each step; these counts are then thresholded into binary strings.

2. **One-Hot Encoding**  
   - Each bit 0 in the string is mapped to `[1 0]`, and each bit 1 to `[0 1]`.  
   - Concatenating two length-200 one-hot vectors produces a 400-component input for the MLP.

## Network Architecture

- **Input layer**: 400 nodes (flattened one-hot vectors).
- **Hidden layer 1**:  ReLU activation, 400 neurons.  
- **Hidden layer 2**: Softmax activation over two output classes (“ordered” vs. “disordered”).
- **Training objective**: Cross-entropy loss with stochastic gradient descent (SGD).

Despite its simplicity, this MLP learns to distinguish phases by picking up subtle statistical differences in the flattened strings, effectively learning to approximate the order parameter.

## Results

### Linear Classifier Baseline

As a baseline, we trained a linear classifier that uses only the counts of 1’s and 0’s (i.e., total magnetization) to predict phase labels. Below is an example of its accuracy across temperatures:

![Linear Classifier Predictions](https://github.com/user-attachments/assets/5451e407-7a00-44ed-bab9-67702392510f)

*Figure 1: Predictions from a linear classifier based solely on the total number of 1’s and 0’s in each input string.*

### MLP Performance

The MLP, in contrast, incorporates higher-order correlations implicit in the one-hot vectors. Its predictions align much more sharply with known critical temperatures:

![MLP Model Predictions](https://github.com/user-attachments/assets/2e18e745-55ad-43a6-95a6-c37fae753788)

*Figure 2: MLP predictions (probability of “ordered” phase) as a function of temperature for L sites. The curve becomes steeper near the true critical point.*

It is evident that the MLP outperforms the simple linear baseline, successfully capturing the phase transition despite never being given explicit information about the Hamiltonian or spin interactions.

## Comparison with Unsupervised Methods

For reference, prior work has applied PCA followed by k-means clustering to identify phases in similar spin models. While PCA + k-means can separate low- and high-temperature regimes to some extent, it typically requires handcrafted feature extraction (e.g., block‐averaged magnetization) and does not offer the same predictive accuracy or generalization as the supervised MLP approach.



## References

- Philipp Höller, Andreas Kräh, Johannes Imriška, **“Learning Phase Transitions in Spin Systems with Neural Networks”**, _arXiv:2112.06735_, 2021.  
  https://arxiv.org/abs/2112.06735

