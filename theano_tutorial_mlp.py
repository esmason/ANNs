'''
This is the code from the theano deep learning tutorial multilayer
perceptron example (http://deeplearning.net/tutorial/mlp.html)
with addiitons by me:
  -implementations of update rules described in (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    -momentum
    -nesterov momentum
    -RMSprop
  -implementation of dropout https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
  
'''

"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np
import os
import sys
import timeit
import csv
import numpy
import theano
import theano.tensor as T
import pickle
from MnistReader import MnistReader
from theano.printing import pydotprint

from data import  load_data

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        #initialize accumulators for RMSprop
        accW = theano.shared(value = np.zeros_like(self.W.eval(), dtype = theano.config.floatX),
                                 name = "accW",
                                 borrow = True,
        )
        accB = theano.shared(value = np.zeros_like(self.b.eval(), dtype = theano.config.floatX),
                             name = "accB",
                             borrow = True,
       )

        self.accs = [accW, accB]


        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, dropout = False, dropout_rate = 0.5):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.rng = rng
        self.n_out = n_out
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output) )
        if dropout:
            self.set_dropout(dropout_rate)
  

        
  
        
        # parameters of the model
        self.params = [self.W, self.b]
        accW = theano.shared(value = np.zeros_like(self.W.eval(), dtype = theano.config.floatX),
                                name = "accW",
                                borrow = True)
        accB = theano.shared(value = np.zeros_like(self.b.eval(), dtype = theano.config.floatX),
                             name = "accB",
                             borrow = True)

        self.accs = [accW, accB]

    def set_dropout(self, p):
        self.dropout_rate = p
        self.dropout = theano.shared(value=np.ones((20,1)) *np.array( self.rng.binomial(size = (self.n_out, ), n = 1, p = self.dropout_rate),
                                                         dtype = theano.config.floatX),
                                         name = 'dropout',
                                         borrow = True
                                         )
        self.output *= self.dropout
        
    def set_predict(self):
        self.dropout = theano.shared(value=np.ones((28000,1)) *np.array( self.rng.binomial(size = (self.n_out, ), n = 1, p = 1),
                                                         dtype = theano.config.floatX),
                                         name = 'dropout',
                                         borrow = True
                                         )
        
        self.output *= theano.shared(value=np.ones((28000,1)) * ( np.transpose(np.array([self.dropout_rate]*self.n_out, dtype = theano.config.floatX)
                                                         )),
                                         name = 'dropout',
                                         borrow = True
                                         )


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout = False, dropout_rate = 0.5):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            dropout = dropout,
            dropout_rate = dropout_rate
        )
        if dropout:
            self.dropout = self.hiddenLayer.dropout
            self.dropout_rate = dropout_rate

        
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        #accumulators
        self.accs = self.hiddenLayer.accs + self.logRegressionLayer.accs

        # keep track of model input
        self.input = input

    ###TODO be able to set dropout rate for final run through
    def set_dropout(self, p):
        self.dropout_rate = p
        self.hiddenLayer.set_dropout(p)


def test_mlp(learning_rate=0.002, L1_reg=0.00, L2_reg=0.0001, n_epochs=600,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500,
             update_rule = 'standard',
             dropout = False):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    : type update_rule: string
    : param update_rule: the method of updating the weights, either RMS, momentum, or nesterov
   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        dropout = dropout
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    if update_rule != 'nesterov':
        gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    #init epoch here so it can be used to smoothly scale up momentum
    epoch = 0

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    if update_rule == 'standard':
        updates = [  (param, param - learning_rate * gparam)
       for param, gparam in zip(classifier.params, gparams)]
    elif update_rule =='RMS':
        updates = RMSprop(classifier.params, gparams, classifier.accs, lr = learning_rate)
    elif update_rule == 'momentum':
        updates = classical_momentum(classifier.params, gparams, classifier.accs, epoch, n_epochs, lr = learning_rate)
    elif update_rule == 'nesterov':
        updates = nesterov_momentum(classifier.params, classifier.accs, epoch, n_epochs,  cost, lr = learning_rate)

    #get a new random dropout matrix
    if dropout:
        new_dropout = np.ones((20,1)) * rng.binomial(size = (500, ), n = 1, p = 0.5)
        updates.append( (classifier.dropout, new_dropout))
        
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates, 
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    validation_errors = []
    while (epoch < n_epochs): # and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                validation_errors.append(this_validation_loss * 100)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # save the best model
                    with open('best_model_mlp.pkl', 'wb') as f:
                        pickle.dump(classifier, f) 

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                #break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    with open(('optimization_%s%f.csv')%(update_rule, learning_rate) , 'w') as csvfile:
        fieldnames = ['error_validation_set',
                      'val_freq',
                      'minibatch/epoch',
                     'batch_size',
                      'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict([('error_validation_set', validation_errors),
                              ('val_freq', validation_frequency),
                              ('minibatch/epoch', n_train_batches),
                              ('batch_size', batch_size),
                              ('learning_rate', learning_rate)]))
 
def RMSprop(params, grads, acc, lr=0.00001, rho=0.9, epsilon=1e-6):
    updates = []
    for p, g, acc in zip(params, grads, acc):
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def classical_momentum(params, grads, acc, epoch, total_epochs,
                       lr=0.0001, decay = 0.9):
    updates = []
    decay = 0.5 + (decay-0.5)*epoch/total_epochs

    for p, g, acc in zip(params, grads, acc):
        acc_new = decay*acc - lr*g
        
        updates.append((acc, acc_new))
        updates.append((p, p +acc))
    return updates

def nesterov_momentum(params, acc, epoch, total_epochs, cost,
                      lr=0.0001, decay = 0.9):
    updates = []
    decay = 0.5 + (decay-0.5)*epoch/total_epochs
    for p, acc in zip(params, acc):
        step = p + decay*acc
        gparams = [T.grad(cost, param) for param in step]
        acc_new = decay*acc - lr*gparams
        updates.append((acc, acc_new))
        updates.append((p, p+ acc))
    return updates

def predict(first_ten = True):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open(r'best_model_mlp.pkl', "rb"), encoding = 'latin1')
    new_drop = T.matrix()
    set_predict_mode = theano.function(inputs = [new_drop], updates = [(classifier.dropout, new_drop)])
    set_predict_mode( np.ones((28000,500)) * classifier.dropout_rate )
    

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)
    # We can test it on some examples from test test
    A = MnistReader("test.csv")
    print("loading test data....")
    A.read_test_file()
    
    test_set_x = A.inputs
#    datasets = load_data(dataset)
#    test_set_x, test_set_y = datasets[2]
#    test_set_x = test_set_x.get_value()

    if first_ten:
        predicted_values = predict_model(test_set_x[:10])
    else:
        predicted_values = predict_model(test_set_x[:])

    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    
    return predicted_values


if __name__ == '__main__':
    rule= 'momentum'
    for rate in [0.001]:
        test_mlp(n_epochs = 60, update_rule = rule, learning_rate = rate, dropout = False)

    x = predict(first_ten = False)

    with open('ans_NN.csv', 'w') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i,j in enumerate(x):
            writer.writerow(dict(zip(fieldnames, (i+1, j) )))
##        

    

    
