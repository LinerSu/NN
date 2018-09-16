import numpy as np
from nn_softmax import softmax, softmax_grad
from nn_sigmoid import sigmoid, sigmoid_grad
from nn_relu import relu, relu_grad
from nn_tanh import tanh, tanh_grad

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer l

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation of the previous layer l-1, same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # Compute the derivatives
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dz)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for current layer l.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, Z = cache

    if activation == "sigmoid":
        dZ = np.multiply(dA, sigmoid_grad(Z))
    elif activation == "relu":
        dZ = np.multiply(dA, relu_grad(Z))
    """
        TODO: Understand this!
    elif activation == "softmax":
        A = softmax(Z)
    """
    elif activation == "tanh":
        dZ = np.multiply(dA, tanh_grad(Z))

    assert (dZ.shape == Z.shape)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def nn_backward(AL, Y, caches, activations):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    activations -- activation function for each layer

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shapes(1)
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    # Consider the loss function
    grads["dA" + str(L)] = 0 # TODO: Change this!!

    # backpropagation --- Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l)], grads["db" + str(l)] = \
            linear_activation_backward(grads["dA" + str(l+1)], current_cache, activations[l])

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters
