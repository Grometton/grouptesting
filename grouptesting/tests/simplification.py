import sympy
from sympy import *
from sympy.plotting import plot

eta, alpha, theta, q, delta = symbols('eta, alpha, theta, q, delta')
n = symbols('n', integer=True)
# f, g, h = symbols('f g h', cls=Function)
init_printing(use_unicode=True)

d = n**theta

# z = ((1- alpha/(n**theta) * (1-q))**(k-1) + (1/q)*(1 - alpha/(n**theta) * (1-q))**k) / 2
# f = ((1.00001) * (1/(q**2)) * (n**theta) * (np.log(n) / np.log(2))) / (alpha * (1 - np.exp(-2)) * (z - (1/q) * (1- alpha/(n**theta) * (1-q))**k)**2)

Tminus = ((1 + eta) * theta * (1/(q**2)) * (n**theta) * (log(n) / log(2))) / (alpha * (1 - exp(-2)) * ((1+delta) - (1- alpha/(n**theta) * (1-q))**(d-1))**2)
Tplus = ((1 + eta) * (1/(q**2)) * (n**theta) * (log(n) / log(2))) / (alpha * (1 - exp(-2)) * ((1+delta) - (1/q) * (1- alpha/(n**theta) * (1-q))**d)**2)

l = (1 - alpha / d * (1-q))**(d-1)
u = (1 - alpha / d * (1-q))**(d) * (1/q)

delta_star = solve(Tplus - Tminus, delta)

# Now solve is a list. Get the elements of the list that satisfy the inequality
# solve(l-1 < delta_star[0] < u-1, delta)
# solve_univariate_inequality(delta_star[0] < u-1, delta)
# p1 = plot(Tplus - Tminus)
# p1.show()


from sympy import latex

print("\\begin{array}{rcl}")
print("delta_star0 & = & %s \\\\" % latex((delta_star[0]).factor()))
print("delta_star1 & = & %s \\\\" % latex((delta_star[1]).factor()))
print("\\end{array}")
