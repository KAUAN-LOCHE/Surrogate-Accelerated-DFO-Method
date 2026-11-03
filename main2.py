import numpy as np
from methodWithSurrogate import finiteDiffGradientWithSurrogate

# Rosenbrock function with dimension n = 2*m[cite: 8]
m = 5
f = lambda x: np.sum(100*(x[::2]**2-x[1::2])**2 + (x[::2]-1)**2)
x0 = np.zeros(2*m)

tol = 1e-4
evalMax = 200*m
sobolev = True

# Rodando os modelos do artigo
x1, f1 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"NeuralNet",sobolev)
x2, f2 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"RadialBasis",sobolev)

# Rodando as suas novas arquiteturas PyTorch!
x3, f3 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"XNet", sobolev)

# x4, f4 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"ResidualMLP",sobolev)

print("Last objective value for the NN-accelerated DFO method is " + str(f1))
print("Last objective value for the RBF-accelerated DFO method is " + str(f2))
print("Last objective value for the XNet-accelerated DFO method is " + str(f3))
# print("Last objective value for the ResidualMLP-accelerated DFO method is " + str(f4))