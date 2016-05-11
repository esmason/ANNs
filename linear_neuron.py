import numpy as np
import random
'''simple linear neuron with example training data, highly sensitive to condition number of input array'''


def linear_neuron(input_vectors, target_vector, error_bound=1, epsilon=0.0001):
    """Args:
          input_vector (ndarray) vector of the inputs X [[x11, x12....x1n]....[xm1, xm2....xmn] ]
          target_vector (ndarray) vector of targets Y [y1, y2 ...... ym]
          error_bound (float) target error bound (per weight average error) of final weights
          epsilon (float) learning rate

       Returns:
          ndarray: the 'best' weight vector of inputs
    """
    inputs = input_vectors
    target = target_vector
    input_condition_number = np.linalg.cond(inputs)
    if input_condition_number > 100:
        raise ValueError("input matrix condition number too large")
    if not(type(input_vectors) == np.ndarray):
        raise TypeError
    if not(type(target_vector) == np.ndarray):
        raise TypeError
    if not len(inputs) == len(target):
        raise ValueError("different number of input and output")

    #initialize weight vector to the average target divided by the number of inputs
    initial_weight = np.mean(target)/len(inputs[0])
    weight_vec = np.array([initial_weight]*len(inputs[0]))
    weight_vec = np.transpose(weight_vec)
    error = determine_error(inputs, target, weight_vec)
    while abs(error) > error_bound*len(inputs):
        weight_vec = update_weights(input_vectors, target_vector, weight_vec, epsilon)
        error = determine_error(inputs, target, weight_vec)
        #print(weight_vec)
    return weight_vec

        


def determine_error(input_vectors, target_vector, weights):
    #calculate the current guess of the outputs
    current_guess = np.dot(input_vectors, weights)
    error_accumulator = np.subtract(target_vector, current_guess)
    #sum all the errors of the current guess
    error_accumulator = sum(error_accumulator)
    print(error_accumulator)
    #formal defintion of error below but very senstive to condition number of input matrix
    '''for i in range(len(input_vectors)):
        error_accumulator = float( target_vector[i] - (np.dot(input_vectors[i], weights) )**2 ) / 2.0
    #print(error_accumulator)'''
    return error_accumulator


def update_weights(input_vectors, target_vector, weights, epsilon):
    output = np.dot(input_vectors, weights)
    for i in range(len(weights)):
        for j in range(len(input_vectors)):
            weights[i] += epsilon*input_vectors[j][i]*(target_vector[j]-output[j])
    return weights
        

inputs = np.array([ [3, 6, 9], [4, 7, 2], [3, 26, 4], [2, 3, 3], [5, 0, 2], [3, 1, 4]
, [14, 8, 5], [4, 9, 2], [5, 7, 1], [1, 3, 7 ]])

print(len(inputs))

true_weights = np.array([10, 10, 10])

targets = np.dot(inputs, true_weights)
        
print(np.linalg.cond(inputs))
w=linear_neuron(inputs, targets)

