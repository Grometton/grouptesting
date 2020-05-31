# Group Testing

A python package for running group testing simulations.
Made by Gabriel Arpino and Nicolò Grometto at ETH Zürich in 2020.

### Setup

1. Clone the package to your local machine
2. Run `python setup.py install` or `pip install .` if you are using anaconda.

## Example

``` python
import grouptesting
from grouptesting.model import *
from grouptesting.algorithms import *

n = 1000 # population size
theta = 0.15
C = 2
n_theta = C * (n**theta)
k = round(n_theta) # number of infected
alpha = 0.5
p = alpha/k # Bernoulli test design probability parameter
q = 0.1
T = 300

sigma = D(n, k) # Generate the vector of defectives
X_ber = Ber(n, int(round(T)), p)
y_ber = Y(dilution_noise(X_ber, q), sigma)

# NCOMP - Bernoulli
sigma_hat_ber = NCOMP(X_ber, y_ber, q=q, delta=0)
err = error(sigma, sigma_hat_ber)
print("NCOMP error: ", err)

# NDD - Bernoulli
sigma_hat_ber = NDD(X_ber, y_ber, pi=q+0.1, alpha=alpha, T=T, d=k)
err = error(sigma, sigma_hat_ber)
print("NDD error: ", err)

# COMP - Bernoulli
sigma_hat_ber = COMP(X_ber, y_ber)
err = (error(sigma, sigma_hat_ber))
print("COMP error: ", err)
```
