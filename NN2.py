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
        self.current_guess = 0

        self.layer_inputs = [[]]*(len(layer_neuron))
        self.layer_outputs = [[]]*(len(layer_neuron))
        #deltas: don't include input layer but still put it in for spacing ie self.deltas[0] should always be empty
        self.deltas = [[]]*((len(layer_neuron)))

        #make the weight matrices, each matrix nXm matrix with m = # nodes in the ith layer (incl bias) and n = # nodes in the (i+1)th layer
        #each row represents the set of weights from all m neurons in the ith layer to a single neuron in the (i+1)th layer
        #conversely, each column is all the output weights from a node in the ith layer, to each n nodes in the (i+1)th layer
        #the right-most column represents output weights from the bias node
        for i in range(len(self.layer_neuron)-1 ):
            np.random.seed(0)
            self.weights.append(np.random.normal( scale = 0.2, size = (self.layer_neuron[i+1], self.layer_neuron[i] + 1)))
##        print("weights:")
##        print(self.weights)


    def activation_function(self, x, deriv=False):
        if not deriv:
            return 1/(1+np.exp(-x))
        else:
            out = activation_function(x)
            return out(1-out)

    def run(self, input_vals, target_vals):
        assert type(target_vals) == np.ndarray
        assert len(target_vals[0]) == self.layer_neuron[-1]
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
        
    def update_deltas(self):
        #update should have been run once, check outputs are non-null
        assert len(self.layer_outputs[-1]) == self.layer_neuron[-1]
        last_output = self.layer_outputs[-1]
        #calculate the derivative of the squared error wrt the output of each k
        err_deriv = last_output - self.target_vals
##        print(err_deriv)
##        print(last_output)
##        print(np.ones(last_output.shape))
       #set deltas for the output layer
        self.deltas[-1] = last_output*(np.ones(last_output.shape) - last_output)*err_deriv
        #set deltas for an arbitrary number of hidden layers each layer is denoted i
        for i in reversed(range(1, len(self.layer_neuron) -1 )):
            # take weights from layer i to layer i + 1
            weights = self.weights[i].T
##            print(weights)
##            print(self.deltas[i+1])
            #transpose the weight matrix so each row is the weights coming out of a single neuron, dot them with
            #the next layers delta
            #thus sum_next_deltas is a nX1 array wherein each "row" corresponds to the neuron at the same row in the weight matrix
            sum_next_deltas = np.dot(weights, self.deltas[i+1])
##            print("sum" , end='')
##            print(sum_next_deltas)
            #print(self.layer_outputs[i].shape)
            outputs = np.vstack([self.layer_outputs[i], np.ones((1, self.layer_outputs[i].shape[1]))])
            #NB: techincally there is no point calculating the delta for the bias node but I do here for completeness
            deltas = outputs*(np.ones( (outputs.shape[0], 1)) - outputs)*sum_next_deltas
            self.deltas[i] = deltas
            
                                     
    def update_weights(self):
        weight_deltas = []*(len(self.layer_neuron) - 1)
        for i in range(len(self.layer_neuron)-1):
            #add the bias outputs
            outputs = np.vstack((self.layer_outputs[i], np.ones((1, self.layer_outputs[i].shape[1]))))
            #print(outputs)

            next_layer_deltas = self.deltas[i+1]
            #print(next_layer_deltas[:-1])
            #we do not care about the deltas of the bias nodes (will always be zero) because nothing is going to them
            #therefore if the next layer is a hidden layer, remove the bias deltas
            if i+1 < len(self.layer_neuron)-1:
                next_layer_deltas = next_layer_deltas[:-1]
            #dot the deltas from the next layer with this layers outputs to obtain a nXm matrix in which each row is
            #the changes in the weights for all nodes in layer i to 1 node in layer i +1
            #conversely each column is the change in weights for a single node in i to all the nodes in i + 1
            #the last column is the bias node
            delta_weight = np.sum(next_layer_deltas[None,:,:].transpose(2, 1,0)*outputs[None,:,:].transpose(2,0,1), axis=0)
            #print(self.weights)
            #print(delta_weight)
            self.weights[i] += -0.1*( delta_weight)
##            print(self.weights)
            


        

    def update(self):
        ""
        assert type(self.layer_inputs[0]) == np.ndarray
        for i in range(len(self.layer_inputs)-1):
##            print(self.layer_outputs[i])
            #forward propogate by dotting the weight matrices with their correspoding output matrices
            next_layer_inputs = np.dot(self.weights[i], np.vstack((self.layer_outputs[i], np.ones((1, self.input_len)))))
            self.layer_inputs[i+1] = next_layer_inputs
            next_layer_outputs = self.activation_function(next_layer_inputs)
            self.layer_outputs[i+1] = next_layer_outputs

        self.current_guess = next_layer_outputs
##        print("old weights : ", end = "")
##        print(self.weights)
        self.update_deltas()
        self.update_weights()
##        print("new weights : ", end = "")
##        print(self.weights)
        #print("inputs:")
        #print(self.layer_inputs)
        #print("outputs:")
        #print(self.layer_outputs)
            



a = NeuralNet((3,8,1))
#print(np.array([1.0, 2.0, 3.0]))
#print(type(np.vstack([np.array([1.0, 2.0, 3.0]), np.ones((1, 3))]) ))
a.run(np.array([[ 1, 1, 1], [1,1, 0], [1, 0,1] ,[0,1,1], [0,0,1], [1,0,0] ,[0,1,0], [0,0,0]   ]), np.array([[0.95],[0.95], [0.95], [0.95], [0.95], [0.05],[0.05],[0.05]]))
#train for p^qvr
#print(len(np.array([[ 1.0, 1.0], [1.0,1.0]])[0]))
#print(a.layer_neuron)
#print(a.weights)
for i in range(10000):
    a.update()


#print(type(a.layer_inputs[0])==np.ndarray)
#print(np.array([1.0, 2.0, 3.0]).shape[0])

