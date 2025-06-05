import numpy as np

import numpy as np
import random

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random


import numpy as np

def wolff_algorithm(L, q, J=1.0, T=1.0, steps=1000):
    """
    Runs the Wolff cluster algorithm for a q-state Potts model with temperature.

    Parameters:
    L: int
        Lattice size (assuming a square LxL lattice).
    q: int
        Number of spin states (q-state Potts model).
    J: float
        Interaction strength (default is 1.0).
    T: float
        Temperature (T replaces beta, T = 1/beta).
    steps: int
        Number of Monte Carlo steps to simulate.

    Returns:
    lattice: 2D numpy array
        The final lattice configuration after simulation.
    """
    # Ensure temperature is valid
    if T <= 0:
        raise ValueError("Temperature T must be greater than zero.")

    # Initialize the lattice randomly
    lattice = np.random.randint(0, q, size=(L, L))

    # Define the probability to add a spin to the cluster, using beta = 1/T
    p_add = 1 - np.exp(-J / T)

    def neighbors(i, j):
        """
        Find the 4 nearest neighbors on a 2D square lattice with periodic boundary conditions.
        """
        return [(i, (j - 1) % L), (i, (j + 1) % L), ((i - 1) % L, j), ((i + 1) % L, j)]

    def flip_cluster():
        """
        Perform one Wolff cluster update.
        """
        # Choose a random spin as the seed for the cluster
        i0, j0 = np.random.randint(0, L), np.random.randint(0, L)
        seed_spin = lattice[i0, j0]

        # Cluster formation using BFS/DFS
        cluster = set()
        cluster.add((i0, j0))
        stack = [(i0, j0)]  # Use a stack for DFS-like behavior

        while stack:
            i, j = stack.pop()
            for ni, nj in neighbors(i, j):
                if (ni, nj) not in cluster and lattice[ni, nj] == seed_spin:
                    # Decide whether to add the neighboring spin to the cluster
                    if np.random.random() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))

        # Flip all spins in the cluster to a random state different from seed_spin
        new_spin = (seed_spin + np.random.randint(1, q)) % q
        for i, j in cluster:
            lattice[i, j] = new_spin

    # Run the Wolff algorithm for the given number of steps
    for step in range(steps):
        flip_cluster()

    # Return the final configuration of the lattice
    return lattice


def wolff_algorithm_anim(L, q, J=1.0, T=1.0, steps=100):
    """
    Runs the Wolff cluster algorithm for a q-state Potts model with temperature
    and returns the lattice configuration over time for visualization.

    Parameters:
    L: int
        Lattice size (assuming a square LxL lattice).
    q: int
        Number of spin states (q-state Potts model).
    J: float
        Interaction strength (default is 1.0).
    T: float
        Temperature (T replaces beta, T = 1/beta).
    steps: int
        Number of Monte Carlo steps to simulate.

    Returns:
    frames: list of 2D numpy arrays
        A list of lattice configurations for each step, to be used for animation.
    """
    # Initialize the lattice randomly
    lattice = np.random.randint(0, q, size=(L, L))

    # Define the probability to add a spin to the cluster, using beta = 1/T
    if q == 2:
        # Ising
        p_add = 1 - np.exp(-2.0 * J / T)
    else:
        # General q-state Potts
        p_add = 1 - np.exp(-J / T)

    frames = [lattice.copy()]  # To store lattice configurations for animation

    def neighbors(i, j):
        """
        Find the 4 nearest neighbors on a 2D square lattice with periodic boundary conditions.
        """
        return [(i, (j - 1) % L), (i, (j + 1) % L), ((i - 1) % L, j), ((i + 1) % L, j)]

    def flip_cluster():
        """
        Perform one Wolff cluster update.
        """
        # Choose a random spin as the seed for the cluster
        i0, j0 = np.random.randint(0, L), np.random.randint(0, L)
        seed_spin = lattice[i0, j0]

        # Cluster formation using BFS/DFS
        cluster = set()
        cluster.add((i0, j0))
        stack = [(i0, j0)]  # Use a stack for DFS-like behavior

        while stack:
            i, j = stack.pop()
            for ni, nj in neighbors(i, j):
                if (ni, nj) not in cluster and lattice[ni, nj] == seed_spin:
                    # Decide whether to add the neighboring spin to the cluster
                    if random.random() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))

        # Flip all spins in the cluster to a random state different from seed_spin
        new_spin = (seed_spin + random.randint(1, q - 1)) % q
        for i, j in cluster:
            lattice[i, j] = new_spin

    # Run the Wolff algorithm for the given number of steps
    for step in range(steps):
        flip_cluster()
        frames.append(lattice.copy())  # Store the lattice configuration after each update

    return frames


def animate_wolff(frames, T):
    """
    Creates an animation from the list of frames generated by the Wolff algorithm.

    Parameters:
    frames: list of 2D numpy arrays
        A list of lattice configurations.
    """
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('viridis', np.max(frames[0]) - np.min(frames[0]) + 1)

    def update(frame):
        ax.clear()
        ax.imshow(frame, cmap=cmap, vmin=np.min(frames[0]), vmax=np.max(frames[0]))
        ax.set_title(f'Wolff Algorithm: q-state Potts Model for T={T}')

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, repeat=False)
    plt.show()


def generate_training_data(num_samples=200, num_sites=200):
    # Create a matrix of ones and zeros as before
    config_1 = np.ones((num_samples // 2, num_sites))  # Ones
    config_2 = np.zeros((num_samples // 2, num_sites))  # Zeros

    # Stack them into one matrix
    X_train = np.vstack((config_1, config_2))

    # One-hot encode the matrix: [1, 0] for ones and [0, 1] for zeros
    one_hot_encoded_X_train = np.array([[[1, 0] if val == 1 else [0, 1] for val in row] for row in X_train])

    # Flatten each sample by reshaping from (num_sites, 2) to (num_sites * 2)
    flattened_X_train = one_hot_encoded_X_train.reshape(num_samples, num_sites * 2)

    # Generate one-hot encoded y_train: [1, 0] for ones and [0, 1] for zeros
    y_train = np.array([[1, 0]] * (num_samples // 2) + [[0, 1]] * (num_samples // 2))
    # Shuffle the dataset randomly
    indices = np.random.permutation(num_samples)
    flattened_X_train = flattened_X_train[indices]
    y_train = y_train[indices]

    return flattened_X_train, y_train


def preprocess_data_clock(configurations, q=2):
    processed_data = []
    for config in configurations:
        # Randomly pick 200 sites and flatten them
        flattened = config.flatten()
        indices = np.random.choice(flattened.size, 200, replace=False)
        selected_angles = flattened[indices]

        # Apply condition based on angle
        binary_encoding = np.array([[1, 0] if angle < q/2-0.01 else [0, 1] for angle in selected_angles])
        # Flatten the binary encoded data
        processed_data.append(binary_encoding.flatten())

    return np.array(processed_data)


def process_data_tree(configurations, q=2):
    result = []
    for config in configurations:
        # Randomly pick 200 sites and flatten them
        flattened = config.flatten()
        indices = np.random.choice(flattened.size, 200, replace=False)
        selected_angles = flattened[indices]

        # Apply condition based on angle
        binary_encoding = np.array([1 if angle < q/2 else 0 for angle in selected_angles])
        sumbin = sum(binary_encoding) / 200
        result.append(np.sqrt(sumbin ** 2 + (1 - sumbin) ** 2))

    return result


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(400, 2)
        self.fc2 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.l2_reg = 0.01

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        # trying out without softmax before cross-entropy
        #return self.softmax(out)


# L2 Regularization
def l2_regularization(model, lambda_val=0.001):
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.sum(param.pow(2))
    return lambda_val * l2_loss


# Convert data to PyTorch tensors
X_train, y_train = generate_training_data()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


# Assuming SimpleNN is defined and generate_training_data() is available

# Function to plot weights of the first layer
def plot_weights(model):
    plt.clf()  # Clear the previous plot
    # Get the weights from the first layer (fc1)
    weights = model.fc1.weight.detach().cpu().numpy()  # Detach and convert to NumPy for plotting
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Weights of the first layer (fc1)")
    plt.xlabel("Neurons")
    plt.ylabel("Input Features")
    plt.draw()
    plt.pause(0.4)  # Pause to allow real-time updates


# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Initialize arrays to store loss
train_loss_history = []

# Training loop
epochs = 300
batch_size = 40

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train_tensor, torch.max(y_train_tensor, 1)[1])
loader = DataLoader(dataset, batch_size=40, shuffle=True)

for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        l2_loss = l2_regularization(model, lambda_val=0.005)
        total_loss = loss + l2_loss
        total_loss.backward()
        optimizer.step()

    # Store the loss for plotting
    train_loss_history.append(total_loss.item())

    # Update weight plot every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}')
        # plot_weights(model)

# To show the final plot
# plt.show()

PlotFunc = False
# # Plot learning curves
if PlotFunc:
    plt.plot(range(1, epochs + 1), train_loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve: Loss vs Epochs')
    plt.legend()
    plt.show()

# Generate configurations and test the model
L = 64
temperatures = [0.1+ 0.04 * i for i in range(1, 40)]
num_steps = 1000
q_sim = 6

# Generate multiple configurations for each temperature
num_samples_per_temp = 50 # Set how many configurations you want to generate per temperature
all_predictions = []
avg_linear = []
all_std_devs = []
for T in temperatures:
    sample_predictions = []
    linear_class_pre = []
    for _ in range(num_samples_per_temp):
        # Generate a new configuration for this temperature using the Wolff algorithm
        configuration = wolff_algorithm(L=L, T=T, steps=num_steps, q=q_sim)
        processed_configuration = torch.tensor(preprocess_data_clock([configuration], q = q_sim), dtype=torch.float32)

        # Predict using the neural network model
        import torch.nn.functional as F

        with torch.no_grad():
            logits = model(processed_configuration)  # shape (1, 2)
            probabilities = F.softmax(logits, dim=1)  # shape (1, 2)
            prob_norm = torch.norm(probabilities, dim=1)  # shape (1,)
            sample_predictions.append(prob_norm.item())
        linear_class_pre.append(process_data_tree([configuration]))

    # Compute the average magnitude of the predictions for this temperature
    avg_magnitude = np.mean(sample_predictions)
    std_dev = np.std(sample_predictions)

    avg_linear.append(np.mean(linear_class_pre))
    all_predictions.append(avg_magnitude)
    all_std_devs.append(std_dev)
# Plot the average magnitude of predictions for each temperature
plt.errorbar(temperatures, all_predictions, yerr=all_std_devs, fmt='o', capsize=5, label='Predictions with Error Bars')
plt.xlabel('Temperature')
plt.ylabel('Average Magnitude R of Neural Network Output')
plt.title(f'Average Magnitude R vs Temperature for {q_sim}-State Clock Model')
plt.legend()
plt.show()

# Display the predictions for each temperature
for i, T in enumerate(temperatures):
    print(f"Temperature {T}: Average NN Output Magnitude {all_predictions[i]}")
# Generate data
# configurations = [wolff_algorithm(L = L,T = T,steps=num_steps, q =q_sim) for T in temperatures]
# print(preprocess_data_clock(configurations))
# X_test = torch.tensor(preprocess_data_clock(configurations), dtype=torch.float32)
# Do a simple tree classifier

plt.plot(temperatures, avg_linear, marker='o', linestyle='none')
plt.xlabel('Temperature')
plt.ylabel('Magnitude R of Linear Classifier Output')
plt.title(f'Magnitude R vs Temperature for {q_sim}-State Clock Model')
plt.show()

# Display the configurations with their temperatures

# Select 5 random configurations with their temperatures
# random_indices = np.random.choice(len(configurations), 5, replace=False)
#random_indices = [i for i in range(1, len(temperatures), 10)]
random_indices = temperatures[::5]
# draw config of spins
PlotFuncState = False
if PlotFuncState:
    for idx in random_indices:
        config= wolff_algorithm(L=L, T=idx, steps=num_steps, q=q_sim)
        #temp = temperatures[idx]
        temp = idx
        plt.imshow(config, cmap='viridis')
        plt.title(f"Final Configuration of Wolff Algorithm (T={temp}, L={L}, steps={num_steps})")
        plt.colorbar(label='Spin State')
        plt.show()
        print(config)
# for idx in random_indices:
#     temp = temperatures[idx]
#     # Generate frames using the Wolff algorithm
#     frames = wolff_algorithm(L=L, q=2, T=temp, steps=num_steps)
#
#     # Animate the results
#     animate_wolff(frames,temp)


# Display the configurations with their temperatures
# for idx in random_indices:
#     config = configurations[idx]
#     temp = temperatures[idx]
#
#     plt.figure()
#     plt.imshow(config, cmap='hsv')  # Displaying the configuration as an image
#     plt.title(f"Temperature: {temp:.2f}")
#     plt.colorbar(label="Spin State")
#     plt.show()

# Predict
# with torch.no_grad():
#    predictions = model(X_test)
#    magnitudes = torch.norm(predictions, dim=1).numpy()
# Display predictions for each temperature
# for i, T in enumerate(temperatures):
#    print(f"Temperature {T}: NN Output {predictions[i]}")
# Plotting the magnitude R versus temperature
# plt.plot(temperatures, magnitudes, marker='o', linestyle='none')
# plt.xlabel('Temperature')
# plt.ylabel('Magnitude R of NN Output')
# plt.title(f'Magnitude R vs Temperature for {q_sim}-State Clock Model')
# plt.show()
