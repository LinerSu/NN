import numpy as np

def initialize_parameters(layer_dims):
    """
    Initialize parameters for L-layer Nerual Network
    For layer l,
        the shape of W is n_l, n_l-1
        the shape of b is n_l, 1

    Use random initialization for the weight matrices (np.random.randn(shape) * 0.01).
    Use zeros initialization for the biases (np.zeros(shape)).


    Arguments:
    layer_dims -- array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)   # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_parameters_check():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running initialize_parameters tests..."
    test1 = initialize_parameters([5,4,3])
    print("W1 = " + str(test1["W1"]))
    print("b1 = " + str(test1["b1"]))
    print("W2 = " + str(test1["W2"]))
    print("b2 = " + str(test1["b2"]))
    ansW1 = np.array([
        [ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
        [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
    ansb1 = np.array([[ 0.], [ 0.], [ 0.], [ 0.]])
    ansW2 = np.array([
        [-0.01185047, -0.0020565, 0.01486148, 0.00236716],
        [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
        [-0.00768836, -0.00230031, 0.00745056, 0.01976111]])
    ansb2 = np.array([[ 0.], [ 0.], [ 0.]])
    assert np.allclose(test1["W1"], ansW1, rtol=1e-05, atol=1e-06)
    assert np.allclose(test1["b1"], ansb1, rtol=1e-05, atol=1e-06)
    assert np.allclose(test1["W2"], ansW2, rtol=1e-05, atol=1e-06)
    assert np.allclose(test1["b2"], ansb2, rtol=1e-05, atol=1e-06)
    print "Pass all tests..."

if __name__ == "__main__":
    initialize_parameters_check()
