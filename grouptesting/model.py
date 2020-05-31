import numpy as np
from scipy.stats import bernoulli
import scipy

# Distributions of random matrices as input for the group testing problem
def Diag(N=100, p=0.1):
  """ Returns a diagonal matrix: test anyone with probability p """
  r = bernoulli.rvs(p, size=int(N))
  return np.diag(r)

def Ber(N=100, T=30, p=0.2):
  """ Bernoulli random matrix, N is population size, T is number of tests, p is
  infection probability. Every person has a probability p of being included in a test"""
  r = bernoulli.rvs(p, size=int(N*T))
  r = np.reshape(r, (N, T))
  return r

def CCW(N=100, T=30, L=None, p=0.2):
  """ Constant column weight random matrix. N: population size, T: number of tests,
  p: probability of element being included in test. For each element, assign it
  to any test with a probability p with replacement. Independently within each
  column of X, L random entries are selected uniformly at random with replacement
  and set to 1. So, each person is put into L tests. So roughly, p=L/T. TESTED."""
  X = np.zeros(shape=(N, T))
  if L is None:
    L = int(p * T)
  # Sample indices between 0 and T-1 with replacement, of size L. Set those
  # indices in each row of X to 1.
  for i in range(N):
    X[i][np.random.choice(T, size=L, replace=True)] = 1
  return X

# Create the defective set
def D(N=100, I=10):
  """ Sample a defectives set, where there are I number of infected people. TESTED."""
  idx = np.array(list(range(N)))
  d = np.zeros(shape=(N,))
  d[np.random.choice(idx, size=I, replace=False)] = 1
  return d

def Y(_X, _D):
  """ _X is the input measurement matrix, _D is the defective set vector. Outputs
  the vector with the test results. TESTED. """
  test_idx = np.where(_D > 0)[0]
  return (np.sum(_X[test_idx, :], axis=0) > 0) * 1

def symmetric_noise(y, noise):
  """ Performs a symmetric noise operation on the raw test output vector Y"""
  return np.mod(y + bernoulli.rvs(noise, size=y.shape[0]), 2)

def dilution_noise(X, noise):
  """ Performs a dilution noise operation on the input matrix columns X """
  return np.maximum(X - bernoulli.rvs(noise, size=X.shape), 0)

def z_channel_noise(y, noise):
  """ Performs a z-channel noise operation on the raw test output vector Y"""
  return np.maximum(y - bernoulli.rvs(noise, size=y.shape[0]), 0)

def rz_channel_noise(y, noise):
  """ Performs a reverse z-channel noise operation on the raw test output vector Y"""
  return np.minimum(y + bernoulli.rvs(noise, size=y.shape[0]), 1)

def accuracy(sigma, sigma_hat):
  return np.sum(sigma*sigma_hat) / np.count_nonzero(sigma)

def symmetric_accuracy(sigma, sigma_hat):
  diff = sigma - sigma_hat
  return (diff.shape[0] - np.sum(np.abs(diff))) / diff.shape[0]

def error(sigma, sigma_hat):
  """ Calculates error = 1 if sigma does not exactly equal sigma_hat """
  return 1 - (np.all(sigma == sigma_hat) * 1)
