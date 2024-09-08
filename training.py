
import numpy as np


# Convert spin states to angles
angles = np.linspace(0, 2* np.pi, 6, endpoint=False)


def wolff_algorithm_clock(L, T, num_steps):
    # Initialize spins (LxL lattice) with random angles
    spins = np.random.choice(angles, size=(L, L))

    def delta_energy(i, j, new_spin):
        neighbors = spins[(i + 1) % L, j] + spins[i, (j + 1) % L] + spins[(i - 1) % L, j] + spins[i, (j - 1) % L]
        current_energy = -np.cos(spins[i, j] - neighbors)
        new_energy = -np.cos(new_spin - neighbors)
        return new_energy - current_energy

    for _ in range(num_steps):
        # Randomly select a seed spin
        i, j = np.random.randint(0, L, size=2)
        cluster = [(i, j)]
        spin_cluster = spins[i, j]
        new_spin = np.random.choice(angles)
        spins[i, j] = new_spin

        # Growing the cluster
        while cluster:
            x, y = cluster.pop()
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                nx %= L
                ny %= L
                if spins[nx, ny] == spin_cluster and np.random.rand() < 1 - np.exp(-delta_energy(nx, ny, new_spin) / T):
                    spins[nx, ny] = new_spin
                    cluster.append((nx, ny))

    return spins


import numpy as np

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

    return flattened_X_train, y_train


def preprocess_data_clock(configurations):
    processed_data = []
    for config in configurations:
        # Randomly pick 200 sites and flatten them
        flattened = config.flatten()
        indices = np.random.choice(flattened.size, 200, replace=False)
        selected_angles = flattened[indices]

        # Generate both cos and sin for each selected angle and flatten them
        cos_sin_data = np.column_stack((np.cos(selected_angles), np.sin(selected_angles)))
        processed_data.append(cos_sin_data.flatten())

    return np.array(processed_data)


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
        return self.softmax(out)


# L2 Regularization
def l2_regularization(model, lambda_val=0.01):
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.sum(param.pow(2))
    return lambda_val * l2_loss


# Convert data to PyTorch tensors
X_train, y_train = generate_training_data()
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
batch_size = 40

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)

    # Compute the loss
    loss = criterion(outputs, torch.max(y_train_tensor, 1)[1])
    l2_loss = l2_regularization(model, lambda_val=0.01)
    total_loss = loss + l2_loss

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}')

# Generate configurations and test the model
L = 32
temperatures = [0.1 + 0.005 * i for i in range(1, 200)]
num_steps = 1000


# Generate data
configurations = [wolff_algorithm_clock(L,T,num_steps=num_steps) for T in temperatures]
X_test = torch.tensor(preprocess_data_clock(configurations), dtype=torch.float32)

# Predict
with torch.no_grad():
    predictions = model(X_test)
    magnitudes = torch.norm(predictions, dim=1).numpy()

# Plotting the magnitude R versus temperature
plt.plot(temperatures, magnitudes, marker='o')
plt.xlabel('Temperature')
plt.ylabel('Magnitude R of NN Output')
plt.title('Magnitude R vs Temperature for 6-State Clock Model')
plt.show()
