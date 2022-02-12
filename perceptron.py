import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# Create random weights to assign to input neurons
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights')
print(synaptic_weights)

# Increase or decrease the number of iterations to see how the weights / outputs change
for iteration in range(20000):
    # Assign inputs from training examples to the input layer
    input_layer = training_inputs
    # Get output from neuron by multiplying by synaptic (neuron) weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # Determine error by subtracting produced outputs from expected outputs
    error = training_outputs - outputs
    # Back-propagation: Adjust weights based on error * input * gradient of sigmoid curve
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training")
print(synaptic_weights)
print('Outputs after training \n', outputs)