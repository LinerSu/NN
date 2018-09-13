import numpy as np

def relu(x):
    """
    Compute the relu function for the input.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    r -- ReLU(x)
    """
    r = np.maximum(x, 0)
    return r

def relu_grad(x):
    """
    Compute the gradient for the relu function.
    relu'(x) = 0 for x <= 0;
               1 for x > 0

    Arguments:
    x -- A scalar or numpy array.

    Return:
    dr -- Your computed gradient.
    """
    dr = x
    dr[x<=0] = 0
    dr[x>0] = 1
    return dr

def test_relu():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running relu & relu_grad tests..."
    x = np.array([
        [ 0.41287266, -0.73082379,  0.78215209],
        [ 0.76983443,  0.46052273,  0.4283139 ],
        [-0.18905708,  0.57197116,  0.53226954]])
    f = relu(x)
    g = relu_grad(x)
    print f
    f_ans = np.array([
        [0.41287266, 0, 0.78215209],
        [0.76983443,  0.46052273,  0.4283139],
        [0,  0.57197116,  0.53226954]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [ 1.,  0.,  1.],
        [ 1.,  1.,  1.],
        [ 0.,  1.,  1.]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "Pass all tests..."


if __name__ == "__main__":
    test_relu()
