import numpy as np
import random as rd

class NeuralNet(object):
    """a basic neural net"""

    def __init__(self, layer_neuron):
        """initializes the neural net
        Args:
           layer_neuron: a tuple with each entry representing the number of neurons
           per layer, does not include bias node"""

        self.num_layers = len(layer_neuron)
        self.layer_neuron = layer_neuron
        #a list of numpy ndarrays
        self.weights = []
        self.input_len = 0
        self.target_vals = []

        self.layer_inputs = [[]]*(len(layer_neuron))
        self.layer_outputs = [[]]*(len(layer_neuron))
        self.deltas = []

        #make the weight matrices, each matrix nXm matrix with m = # nodes in the ith layer (incl bias) and n = # nodes in the (i+1)th layer
        #each row represents the set of weights from all 3 neurons in the ith layer to a single neuron in the (i+1)th layer
        #conversely, each column is all the output weights from a node in the ith layer, to each node in the (i+1)th layer
        for i in range(len(self.layer_neuron)-1 ):
            self.weights.append(np.random.normal( scale = 0.1, size = (self.layer_neuron[i+1], self.layer_neuron[i] + 1)))
        print("weights:")
        print(self.weights)


    def activation_function(self, x, deriv=False):
        if not deriv:
            return 1/(1+np.exp(-x))
        else:
            out = activation_function(x)
            return out(1-out)

    def run(self, input_vals, target_vals):
        self.target_vals = target_vals.T
        """input_vals must be a numpy ndarray"""
        self.input_len = input_vals.shape[0]
        for i in input_vals:
            if not len(i)  == self.layer_neuron[0]:
                raise ValueError('input size does not match number of neurons in first layer')

        #transpose each input, (row vector) into a column vector and add a 1 to the bottom of each column for the bias node
        #first layer is just a buffer layer so no matrix multiplication necessary
        self.layer_inputs[0] = input_vals.T
        #as layer one is an output layer the inputs equal the outputs
        self.layer_outputs[0] = self.layer_inputs[0]
        self.input_len = input_vals.shape[0]
        #print(self.layer_inputs)
        #print(self.layer_outputs)
        self.update()
        
    def updateDeltas(self):
        self.deltas[0

    def update(self):
        "
        assert type(self.layer_inputs[0]) == np.ndarray
        for i in range(len(self.layer_inputs)-1):
            print(self.layer_outputs[i])
            next_layer_inputs = np.dot(self.weights[i], np.vstack((self.layer_outputs[i], np.ones((1, self.input_len)))))
            self.layer_inputs[i+1] = next_layer_inputs
            next_layer_outputs = self.activation_function(next_layer_inputs)
            self.layer_outputs[i+1] = next_layer_outputs
        updateDeltas
        print("inputs:")
        print(self.layer_inputs)
        print("outputs:")
        print(self.layer_outputs)
            



a = NeuralNet((2,2,1))
print(np.array([1.0, 2.0, 3.0]))
print(type(np.vstack([np.array([1.0, 2.0, 3.0]), np.ones((1, 3))]) ))

a.run(np.array([[ 1.0, 2.0]]))
print(len(np.array([[ 1.0, 1.0], [1.0,1.0]])[0]))
#print(a.layer_neuron)
#print(a.weights)


#print(type(a.layer_inputs[0])==np.ndarray)
print(np.array([1.0, 2.0, 3.0]).shape[0])

