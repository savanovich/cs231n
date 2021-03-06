import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':

    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y

    net = init_toy_model()
    X, y = init_toy_data()

    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()
    # The difference should be very small. We get < 1e-7
    print('Difference between your scores and correct scores:')
    print(np.sum(np.abs(scores - correct_scores)))

    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133
    # should be very small, we get < 1e-12
    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))