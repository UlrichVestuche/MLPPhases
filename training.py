


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models

# Convert spin states to angles
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)

def wolff_algorithm_clock(L, T, num_steps):
    # Initialize spins (LxL lattice) with random angles
    spins = np.random.choice(angles, size=(L, L))

    def delta_energy(i, j, new_spin):
        neighbors = spins[(i+1) % L, j] + spins[i, (j+1) % L] + spins[(i-1) % L, j] + spins[i, (j-1) % L]
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
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                nx %= L
                ny %= L
                if spins[nx, ny] == spin_cluster and np.random.rand() < 1 - np.exp(-delta_energy(nx, ny, new_spin) / T):
                    spins[nx, ny] = new_spin
                    cluster.append((nx, ny))

    return spins
# Generate the artificial training data
def generate_training_data(num_samples=200, num_sites=200):
    config_1 = np.ones((num_samples//2, num_sites))
    config_2 = np.zeros((num_samples//2, num_sites))
    X_train = np.vstack((config_1, config_2))
    y_train = np.array([[1, 0]] * (num_samples//2) + [[0, 1]] * (num_samples//2))
    return X_train, y_train
def preprocess_data_clock(configurations):
    processed_data = []
    for config in configurations:
        # Randomly pick 200 sites and flatten them
        flattened = config.flatten()
        indices = np.random.choice(flattened.size, 200, replace=False)
        selected_angles = flattened[indices]
        # Normalize angles to a range suitable for NN input
        #processed_data.append(np.cos(selected_angles) + np.sin(selected_angles))
        # Compute theta mod pi for each selected angle
        processed_data.append(selected_angles/np.pi)
    return np.array(processed_data)

# Generate the training data
X_train, y_train = generate_training_data()


from tensorflow.keras import regularizers

# Define the model with L2 regularization
model = models.Sequential([
    layers.Dense(2, input_shape=(200,), activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization added here
    layers.Dense(2, activation='softmax',
                 kernel_regularizer=regularizers.l2(0.01))  # L2 regularization added here
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=40, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Training Loss: {loss:.4f}")
print(f"Training Accuracy: {accuracy:.4f}")



# Generate configurations at different temperatures
L = 32
temperatures = [0.1 + 0.5*i for i in range(1, 200)]
num_steps = 1000

configurations = [wolff_algorithm_clock(L, T, num_steps) for T in temperatures]
X_test = preprocess_data_clock(configurations)

# Predict the output for the generated configurations
predictions = model.predict(X_test)
magnitudes = np.linalg.norm(predictions, axis=1)

# Display predictions for each temperature
for i, T in enumerate(temperatures):
    print(f"Temperature {T}: NN Output {predictions[i]}")

# Plotting the magnitude R versus temperature
plt.plot(temperatures, magnitudes, marker='o')
plt.xlabel('Temperature')
plt.ylabel('Magnitude R of NN Output')
plt.title('Magnitude R vs Temperature for 6-State Clock Model')
plt.show()