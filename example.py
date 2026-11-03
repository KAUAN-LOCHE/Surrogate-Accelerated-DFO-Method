#!/usr/bin/python3
# ---------------------------------------------------- #
#  Example of use of the surrogate-accelerated method  #
# ---------------------------------------------------- #


import numpy as np
from methodWithSurrogate import finiteDiffGradientWithSurrogate
# import pycutest
import time
# Find unconstrained, variable-dimension problems
# probs = pycutest.find_problems(constraints='unconstrained', userN=True)
# print(sorted(probs)[:5])



# p = pycutest.import_problem('ARGLALE', sifParams={'N':10, 'M':20})
# x0 = p.x0
# print("Problem name: " + p.name)
# print(x0)
# f = lambda x: p.obj(x)


# # Rosenbrock function with dimension n = 2*m
m = 12
f = lambda x: np.sum(100*(x[::2]**2-x[1::2])**2 + (x[::2]-1)**2)
x0 = np.zeros(2*m)

# f = lambda x: 10 * (2*m) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Ponto de partida inicializado longe do mínimo global (que é zero)
# para forçar os algoritmos a navegarem pelas oscilações
# x0 = np.ones(2*m) * 10

tol = 1e-4
evalMax = 200*m
sobolev = True
start_time = time.time()
print("Starting the NN-accelerated DFO method")
np.random.seed(0)  # Set the seed value for reproducibility
xnn, fnn = finiteDiffGradientWithSurrogate(f,np.copy(x0),evalMax,tol,"NeuralNet",sobolev)
print("Last objective value for the NN-accelerated DFO method is " + str(fnn))

# print("Starting the XNETLinear-accelerated DFO method")
# np.random.seed(0)  # Set the seed value for reproducibility
# xnetLinear, fxnetLinear = finiteDiffGradientWithSurrogate(f,np.copy(x0),evalMax,tol,"XNetLinear",sobolev)
# print("Last objective value for the XNETLinear-accelerated DFO method is " + str(fxnetLinear))

# print("Starting the XnetMLP-accelerated DFO method")
# np.random.seed(0)  # Set the seed value for reproducibility
# xnn2c, fnn2c = finiteDiffGradientWithSurrogate(f,np.copy(x0),evalMax,tol,"XnetMLP",sobolev)
# print("Last objective value for the XnetMLP-accelerated DFO method is " + str(fnn2c))

print(time.time() - start_time, "seconds")
# print("Starting the NeuralNet4C-accelerated DFO method")
# np.random.seed(0)  # Set the seed value for reproducibility
# xnn4c, fnn4c = finiteDiffGradientWithSurrogate(f,np.copy(x0),evalMax,tol,"NeuralNet4C",sobolev)
# print("Last objective value for the NeuralNet4C-accelerated DFO method is " + str(fnn4c))

# print("Starting the XNET-accelerated DFO method")
np.random.seed(0)  # Set the seed value for reproducibility
xnet, fxnet = finiteDiffGradientWithSurrogate(f,np.copy(x0),evalMax,tol,"XNet",sobolev)
print("Last objective value for the XNET-accelerated DFO method is " + str(fxnet))




