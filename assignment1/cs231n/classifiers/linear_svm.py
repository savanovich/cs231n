import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. # (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data.  # (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means  # 500
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # (3073, 10)

    # compute the loss and the gradient
    num_classes = W.shape[1]  # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    for i in range(num_train):  # L_i
        scores = X[i].dot(W)  # s = Wx = (1, 3073) @ (3073, 10) = (10, )
        for j in range(num_classes):  # sum j != y_i  # iterate over all classes
            if j == y[i]:  # except right
                continue
            margin = scores[j] - scores[y[i]] + 1  # s_j - s_y_i + 1
            if margin > 0:  # max(0, margin)
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

            # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_train = X.shape[0]  # 500
    scores = X.dot(W)  # (500, 3073) @ (3073, 10) = (500 examples, 10 classes)

    correct_class_score = scores[np.arange(num_train), y]  # (500, )
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)

    # margins = np.maximum(0, scores - scores[np.arange(num_train), y]+ 1)  # scores - real scores + 1

    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins / num_train)
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    nonzero = margins.copy()  #(500, 10)
    nonzero[nonzero != 0] = 1
    nonzero[range(num_train), y] = -(np.sum(nonzero, axis=1))
    dW = X.T.dot(nonzero)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
