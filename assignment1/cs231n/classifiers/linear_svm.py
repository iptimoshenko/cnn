import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i, :]
        dW[:,y[i]] -= X[i, :]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Average gradients as well
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]

  Scores = np.dot(X, W) # (N,C) matrix of scores
  ############ DEBUG HERE, comes out as N*N instead of N*1
  correct_class_score = np.mat(Scores[np.arange(0, Scores.shape[0]), y]).T
  margin = Scores - correct_class_score + 1 # (N, C) matrix of margins
  # don't count loss from correct class
  margin[np.arange(0, Scores.shape[0]), y] = np.zeros(num_train)
  mask = margin > 0
  loss = np.sum(margin[mask])
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  margin[mask] = 1
  margin[margin < 0] = 0
  # TODO: debug, doesn't work with the current version of python
  # margin[np.arange(0, margin.shape[0]), y] = -1 * np.asarray(np.sum(margin, axis=1))
  for i in xrange(margin.shape[0]):
    margin[i, y[i]] = -1 * np.sum(margin[i])
  dW = np.dot(X.T, margin)

  # Average gradients as well
  dW /= num_train

  # Add regularization to the gradient
  dW += reg * W

  return loss, dW
