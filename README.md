Studying Phases of the q-State Clock Model with an MLP
---

This repository demonstrates how a simple multilayer perceptron (MLP) can learn phase transitions in Potts- and clock-type spin models, using only minimally informed input data derived from raw spin configurations.

## Overview

- **Models considered**  
  - **2-state Potts model (Ising)**: Exhibits a well-known phase transition at the Kramers–Wannier dual point.  
  - **6-state clock model**: A richer spin space with a more complex phase structure.

- **Key idea**  
  We generate spin configurations at various temperatures using the Wolff cluster algorithm. Instead of providing the MLP with full lattice configurations or explicit Hamiltonian information, we sample 200 sites from each configuration, threshold their spin values to binary (0 or 1), and then one-hot encode those binaries to form a fixed-length input vector of length 400. The network thus sees only these one-hot–encoded binary strings labeled “ordered” or “disordered,” learning to distinguish phases from statistical patterns in those 200-site samples.

## Data Preparation

1. **Simulation**  
   - Use the Wolff algorithm to generate spin configurations on an L×L lattice for each temperature (T) and q-state (q).  
   - Each L×L configuration is flattened to a 1D array of length L².  
   - Randomly select 200 sites from the flattened array.  
   - For each selected site:  
     - In the Potts/clock case, treat the spin value (an integer from 0 to q−1) as an “angle” placeholder.  
     - Threshold: if spin < q/2 − 0.01, assign 0; otherwise assign 1.

2. **One-Hot Encoding**  
   - Map each binary bit (0 or 1) in the 200-site sample to a two-element one-hot vector:  
     ```
     0 → [1, 0]
     1 → [0, 1]
     ```  
   - Concatenate all 200 one-hot pairs into a single 400-component input vector.  
   - Repeat for each configuration to build the dataset.

3. **Labels**  
   - Label each 400-component input according to its temperature: “ordered” (low T) or “disordered” (high T), based on a chosen critical temperature.

## Network Architecture

- **Input layer**: 400 nodes (flattened one-hot vectors).  
- **First hidden layer**: 2 neurons, ReLU activation (i.e., `nn.Linear(400, 2)` followed by `ReLU`).  
- **Output layer**: 2 neurons (logits for “ordered” vs. “disordered”), corresponding to `nn.Linear(2, 2)`.  
  - Softmax is applied externally when computing probabilities (e.g., with `F.softmax` before interpretation).  
- **Loss**: `CrossEntropyLoss` (which combines `LogSoftmax` + NLL) plus an explicit L₂ penalty on all weights via a custom regularization term.  
- **Optimizer**: Adam (`lr=0.05`).

Despite its simplicity (only two hidden neurons), this MLP can learn subtle statistical differences in the flattened binary samples, effectively capturing an order-parameter–like signal.

## Training Details

- **Dataset**:  
  - Generate 200 total configurations (100 labeled “ordered,” 100 labeled “disordered”), then randomly shuffle.  
  - One-hot encode each 200-site sample to form a 400-dimensional input.  
  - Convert labels to integer class indices (0 or 1) for cross-entropy.

- **Hyperparameters**:  
  - Batch size = 40  
  - Epochs = 300  
  - L₂ regularization coefficient (lambda) = 0.005  
  - Learning rate = 0.05

- **Monitoring**:  
  - Track total loss (cross-entropy + L₂ penalty) each epoch.  
  - Optionally visualize first-layer weights every 10 epochs using a heatmap.

## Baseline: Linear Classifier

As a simple baseline, we also compute a “linear” decision metric from each 200-site sample:  
- For each binary sample (0/1 values at 200 sites), compute the fraction of 0’s vs. 1’s.  
- Define a linear classifier that assigns a phase based on which fraction is larger.  
- Compute the average magnitude:  
  \[
    R = \sqrt{\left(rac{\#\,	ext{zeros}}{200}
ight)^2 + \left(rac{\#\,	ext{ones}}{200}
ight)^2}
  \]  
- Plot \(R\) vs. temperature to see how a magnetization‐based thresholding performs relative to the MLP.

## Results

### Linear Classifier Predictions

Below is an example of how the linear classifier’s average magnitude \(R\) varies with temperature:

![Linear Classifier Predictions](https://github.com/user-attachments/assets/5451e407-7a00-44ed-bab9-67702392510f)

*Figure: Linear classifier’s \(R\) vs. temperature for the 6-state clock model.*

### MLP Predictions

The MLP’s output probabilities for the “ordered” phase (after applying softmax to the logits) produce a sharper transition near the known critical temperature. For each temperature, we record the average softmax probability norm and plot it with error bars:

![MLP Model Predictions](https://github.com/user-attachments/assets/2e18e745-55ad-43a6-95a6-c37fae753788)

*Figure: MLP’s average probability norm for “ordered” vs. temperature, with error bars over multiple samples.*

In practice, the MLP prediction curve is significantly steeper near \(T_c\), demonstrating that it can reliably detect the phase transition with only minimally processed input.


## References

- Philipp Höller, Andreas Kräh, Johannes Imriška, **“Learning Phase Transitions in Spin Systems with Neural Networks”**, _arXiv:2112.06735_, 2021.  
  https://arxiv.org/abs/2112.06735
