# basic neural network in python.
A neural network with one input layer and one output neuron.


## Training

![Alt text](perceptron.png?raw=true "Neural Network")

A neural network is initialized with a random set of weights for its neurons. In this case, these weights are called synaptic_weights. 

Inputs are taken from the training examples and multipled by their weights.

All the multiplied values are added and inputted into a sigmoid
activation function. This function outputs a value between [0,1]

## Backpropagation
The error is calculated by subtracting the produced outputs from the training set values.

The weights are adjusted according to the severeness of the error.

```sh
adjustments = error * sigmoid_derivative(outputs)
synaptic_weights += np.dot(input_layer.T, adjustments)
```
This process iterates 20000 times.
