import numpy as np
from scipy.stats import bernoulli

# Outline of algorithms: https://www.wikiwand.com/en/Group_testing

def NDD(X, y, pi=0.5, alpha=0.5, T = 100, d = 10):
  """ NDD for z-channel noise as in http://export.arxiv.org/pdf/1808.09143.
  Pi has to be in (rho, 1). """

  sigma_hat = np.zeros(shape=(X.shape[0],)) # Generate the sigma hat matrix

  # Calculate ND
  neg_tests = np.where(y < 1)[0] # Find the location of negative tests
  N_neg_j = np.sum(X[:, neg_tests], axis=1) # Sum over columns, return size (Nx1)

  possibly_defectives = np.where(N_neg_j < (pi * T * alpha / d))[0]

  # Step 2: Next the algorithm looks at all the positive tests.
  # If an item appears as the only "possible defective" in a test,
  # then it must be defective, so the algorithm declares it to be defective.
  pos_tests = np.where(y > 0)[0]
  pos_tests_with_one_defective = np.where(np.sum(X[possibly_defectives][:, pos_tests], axis=0) == 1)[0]
  onedefective = np.where(X[possibly_defectives][:, pos_tests[pos_tests_with_one_defective]] == 1)[0]
  onedefective = list(set(onedefective)) # Removes duplicates
  sole_item_in_pos_test = possibly_defectives[onedefective]

  # Apply all logic sigma hat:
  sigma_hat[sole_item_in_pos_test] = 1 # These are definitely defective
  return sigma_hat

def NCOMP(X, y, q=0.5, delta=0.5):
  """ The Noisy COMP algorithm.
  Input: X measurement matrix (NxT), y output vector of test results (Tx1).
  Output: sigma_hat, an estimate of the defective set. size (Nx1)

  q is the assumed noise in the system
  delta is a design parameter
  If |Si| ≥ |Ti|(1−q(1+∆)), then the decoder declares the ith item to be defective
  , else it declares it to be non-defective.

  Convention: 1 is defective, 0 is non defective.
  """

  # |Ti|: size of set of indices in row (person) j where m_j,i = 1
  mag_Ti = np.sum(X, axis=1)  # Sum over columns, return size (Nx1)

  # |Si|: size of set of indices where both yi and mij = 1.
  y_reshaped = np.reshape(y, (y.shape[0], 1)) # In order for the numpy multiplication to work
  and_op = (X.T * y_reshaped).T # Multiplies every row of X by y element wise, performing an and operation. Should return a matrix of size NxT
  mag_Si = np.sum(and_op, axis=1) # Sum over columns, should return a matrix of size (Nx1)

  # Return defectives based on condition.
  return (mag_Si >= mag_Ti * (1 - q * (1 + delta))) * 1

def COMP(X, y):
  """ The COMP algorithm, which is the first step of the DD algorithm.
  Input: X measurement matrix (NxT), y output vector of test results (Tx1).
  Output: sigma_hat, an estimate of the defective set. size (Nx1)

  The algorithm may produce false positives, but does not produce false
  negatives. Convention: 1 is defective, 0 is non defective. Preliminarily tested.
  """
  # Step 1: Mark all as defective.
  sigma_hat = np.ones(shape=(X.shape[0],))

  # Step 2: X is an (NxT) matrix. Decoding step proceeds rowwise. If every test in which a
  # an item appears is positive, then the item is declared defective; otherwise,
  # the item is assumed to be non-defective.

  # Set as non defective the items that appear in negative tests
  neg_tests = np.where(y < 1)[0]
  presence_in_neg_tests = np.sum(X[:, neg_tests], axis=1)
  true_non_defectives_neg = np.where(presence_in_neg_tests > 0)[0]

  # Update the sigma_hat vector according to the findings:
  sigma_hat[true_non_defectives_neg] = 0
  return sigma_hat

def DD(X, y):
  """ Definitely Defectives algorithm. Preliminarily tested. """
  # Step 1: Run COMP and any non-defectives that it detects are removed.
  # All remaining items are now "possibly defective".
  sigma_hat = COMP(X, y)
  possibly_defectives = np.where(sigma_hat > 0)[0] # Since COMP identifies all definite non defectives

  # Step 2: Next the algorithm looks at all the positive tests.
  # If an item appears as the only "possible defective" in a test,
  # then it must be defective, so the algorithm declares it to be defective.
  pos_tests = np.where(y > 0)[0]
  pos_tests_with_one_defective = np.where(np.sum(X[possibly_defectives][:, pos_tests], axis=0) == 1)[0]
  # Find the one defective in that positive test
  onedefective = np.where(X[possibly_defectives][:, pos_tests[pos_tests_with_one_defective]] == 1)[0]
  # xidx now contains the index of the x among the possible defectives that is the sole one occurring in a positive test
  sole_item_in_pos_test = possibly_defectives[onedefective]

  # Step 3: All other items are assumed to be non-defective
  list_of_possibly_defectives = np.arange(possibly_defectives.shape[0])
  assume_non_defective = possibly_defectives[np.setdiff1d(list_of_possibly_defectives, onedefective)]

  # Apply all logic sigma hat:
  sigma_hat[sole_item_in_pos_test] = 1 # These are definitely defective
  sigma_hat[assume_non_defective] = 0
  return sigma_hat

def NoisyDD(X, y, gamma1, gamma2, nu, k): # ONLY VALID FOR BERNOULLI TESTING ACCORDING TO [1]
  """ Noisy Definitely Defectives algorithm. Preliminarily tested. """
  # Step 1: Run COMP and any non-defectives that it detects are removed.
  # All remaining items are now "possibly defective".
  sigma_hat = COMP(X, y)
  possibly_defectives = np.where(sigma_hat > 0)[0] # Since COMP identifies all definite non defectives

  # Step 2: Next the algorithm looks at all the positive tests.
  # If an item appears as the only "possible defective" in a test,
  # then it must be defective, so the algorithm declares it to be defective.
  pos_tests = np.where(y > 0)[0]
  pos_tests_with_one_defective = np.where(np.sum(X[possibly_defectives][:, pos_tests], axis=0) == 1)[0]
  # Find the one defective in that positive test
  onedefective = np.where(X[possibly_defectives][:, pos_tests[pos_tests_with_one_defective]] == 1)[0]
  # xidx now contains the index of the x among the possible defectives that is the sole one occurring in a positive test
  sole_item_in_pos_test = possibly_defectives[onedefective]

  # Step 3: All other items are assumed to be non-defective
  list_of_possibly_defectives = np.arange(possibly_defectives.shape[0])
  assume_non_defective = possibly_defectives[np.setdiff1d(list_of_possibly_defectives, onedefective)]

  # Apply all logic sigma hat:
  sigma_hat[sole_item_in_pos_test] = 1 # These are definitely defective
  sigma_hat[assume_non_defective] = 0
  return sigma_hat
