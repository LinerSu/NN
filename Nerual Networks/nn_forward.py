import numpy as np
from nn_softmax import softmax, softmax_grad
from nn_sigmoid import sigmoid, sigmoid_grad
from nn_relu import relu, relu_grad
from nn_tanh import tanh, tanh_grad

def linear_forward(A, W, b):
    """
    Computer linear forward model

    Linear function:
        Z[l] = W[l].A[l-1] + b[l]

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    # Linear forward functions
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (Z, W, b)

    return Z, cache

def linear_activate_forward(A_prev, W, b, activation):
    """
    the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "softmax":
        A = softmax(Z)
    elif activation == "tanh":
        A = tanh(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, Z)

    return A, cache

def nn_forward(X, parameters, activations):
    """
    Implement forward propagation for the NN computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    activations -- activation function for each layer

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the network

    for l in range(1, L + 1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activations[l])
        caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y, activationL):
    """
    Implement the cost function.

    Sigmoid:
        -1/m * (y . log(AL) + (1 - y) . log(1 - AL))

    Softmax:
        -y . log(AL)

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    activationL --- activation function for output layer

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1
    if activationL == 'sigmoid':
        cost = -1/m * (np.dot(y, np.log(AL).T) + np.dot(1 - y, np.log(1 - AL).T))
    elif activationL == 'softmax':
        cost = -np.dot(y, np.log(AL).T)

    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect

    assert(cost.shape == ())

    return cost


def linear_forward_check():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running linear_forward tests..."
    W1 = np.array([
        [ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
        [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
    b1 = np.array([[ 0.], [ 0.], [ 0.], [ 0.]])
    A0 = np.array([
        [1.5, 2, 4, 5],
        [7, 3, 2, 4],
        [8, 2, 1, 1],
        [7, 8, 2, 3],
        [2, 2, 2, 2] ])
    Z1, linear_cache = linear_forward(A0, W1, b1)
    print("Z1 = " + str(Z1))
    ansZ1 = np.array([
        [-0.07088739, -0.1038294,   0.03842267,  0.04640422],
        [-0.07388496, -0.03516723, -0.03253591, -0.0381765 ],
        [ 0.23339179,  0.15565424,  0.00914316,  0.03079268],
        [-0.12121919,  0.00118412, -0.04493307, -0.05006337]])
    assert np.allclose(Z1, ansZ1, rtol=1e-05, atol=1e-06)
    print "Pass all tests..."

if __name__ == "__main__":
    linear_forward_check()
