import numpy as np

def gradient_descent(X, Y, lr, iterations=1000, tol=1e-6):
    m,n = np.shape(X)
    # seed
    np.random.seed(42)
    weights = np.random.rand(n)
    bias = 0
    i = 0
    erros = []
    parameters_past = []
    for i in range(iterations):
        y_pred = np.dot(X, weights) + bias
        total_erro = 1/m*np.sum(Y-y_pred)
        if np.abs(total_erro) < tol:
            print(f'Converged after {i} iterations')
            break
        print(f'Erro: {total_erro}')

        
               
        erros.append(total_erro)

        #gradients
        grad_erro = Y - y_pred
        grad_weights = -2/m * np.dot(X.T, grad_erro)
        grad_bias = -2/m*np.sum(grad_erro)

        weights -= lr*grad_weights
        bias -= lr*grad_bias
    return weights, bias

X = np.array([[1,1],[2,2], [3,3]] )
Y = np.array([2,4,6])

weights, bias = gradient_descent(X,Y,0.01,10000)

print(f'Weights: {weights}, Bias: {bias}')
print(f'Predicted values: {X.dot(weights) + bias}')

print("\n\n\n")




def momentum_gradient_descent(X, Y, lr, iterations=1000, tol=1e-6, momentum=0.9):
    m,n = np.shape(X)
    # seed
    np.random.seed(42)
    weights = np.random.rand(n)
    bias = 0
    i = 0
    erros = []
    parameters_past = []
    v_weights = 0
    v_bias = 0
    for i in range(iterations):
        y_pred = np.dot(X, weights) + bias
        total_erro = 1/m*np.sum(Y-y_pred)
        if np.abs(total_erro) < tol:
            print(f'Converged after {i} iterations')
            break
        print(f'Erro: {total_erro}')
               
        erros.append(total_erro)

        #gradients
        grad_erro = Y - y_pred
        grad_weights = -2/m * np.dot(X.T, grad_erro)
        grad_bias = -2/m*np.sum(grad_erro)

        v_weights = momentum*v_weights + lr*grad_weights
        v_bias = momentum*v_bias + lr*grad_bias

        weights -= v_weights
        bias -= v_bias
    return weights, bias

weights, bias = momentum_gradient_descent(X,Y,0.01,10000)

print(f'Weights: {weights}, Bias: {bias}')
print(f'Predicted values: {X.dot(weights) + bias}')