import numpy as np
from nn_softmax import softmax, softmax_grad
from nn_sigmoid import sigmoid, sigmoid_grad
from nn_relu import sigmoid, sigmoid_grad

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

    if activation == "sigmoid":
    #TODO: implementation



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
