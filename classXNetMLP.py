#!/usr/bin/python3
# ---------------------------------------------------- #
#  XNet (Cauchy MLP Topology) Surrogate Model          #
# ---------------------------------------------------- #

import numpy as np
from classModel import MyModel

class MyXNet(MyModel):
    
    def __init__(self, n, num_basis, f0, x0, simplexMemory, regParam, maxIterTraining):
        super().__init__(n, num_basis, f0, x0, simplexMemory)

        self.regParam = regParam
        self.maxIterTraining = maxIterTraining
        
        m = self.netDim[1]
        
        # --- PARÂMETROS MLP ---
        # W1 projeta a entrada (n) para os neurônios ocultos (m)
        self.W1 = np.random.uniform(low=-np.sqrt(3/n), high=np.sqrt(3/n), size=(m, n))
        self.b1 = np.zeros(m)
        
        # Parâmetros da Ativação de Cauchy
        self.d = np.ones(m) * 1.0
        self.delta1 = np.random.uniform(low=-1.0, high=1.0, size=(m,))
        self.delta2 = np.random.uniform(low=-1.0, high=1.0, size=(m,))
        
        # Conexão de Saída
        self.W2 = np.random.uniform(low=-np.sqrt(3/m), high=np.sqrt(3/m), size=(m,))
        self.b2 = np.zeros(1)

    # 1. FORWARD PASS
    def __call__(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]
            
        Z = np.matmul(x, self.W1.T) + self.b1
        
        # Ativação de Cauchy Element-wise
        H = 1.0 / (Z**2 + self.d**2 + 1.0)           
        P = self.delta1 * Z + self.delta2
        A = P * H
        
        val = np.matmul(A, self.W2) + self.b2
        
        return np.squeeze(val, axis=0) if is_single else val

    # 2. GRADIENTE DO MODELO (w.r.t. input x)
    def grad(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]
            
        Z = np.matmul(x, self.W1.T) + self.b1
        H = 1.0 / (Z**2 + self.d**2 + 1.0)
        P = self.delta1 * Z + self.delta2
        
        # Derivada da Ativação
        A_prime = self.delta1 * H - 2.0 * P * Z * (H**2)
        
        # Chain rule: sum_j W2_j * A'_j * W1_j
        val = np.matmul(A_prime * self.W2, self.W1)
        
        return np.squeeze(val, axis=0) if is_single else val

    def _unpack(self, fullWeight):
        n, m = self.netDim[:2]
        idx1 = m*n
        idx2 = idx1 + m
        idx3 = idx2 + m
        idx4 = idx3 + m
        idx5 = idx4 + m
        idx6 = idx5 + m
        
        W1 = np.reshape(fullWeight[:idx1], (m, n))
        b1 = fullWeight[idx1:idx2]
        d_val = fullWeight[idx2:idx3]
        delta1 = fullWeight[idx3:idx4]
        delta2 = fullWeight[idx4:idx5]
        W2 = fullWeight[idx5:idx6]
        b2 = fullWeight[idx6:]
        return W1, b1, d_val, delta1, delta2, W2, b2, idx1, idx2, idx3, idx4, idx5, idx6

    # 3. AVALIAÇÃO DA PERDA (MSE + Sobolev)
    def evalLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        W1, b1, d_val, delta1, delta2, W2, b2, *_ = self._unpack(fullWeight)

        X = self.trainInput[:tl[0], :]
        Y = self.trainOutput[:tl[0]]
        
        Z = np.matmul(X, W1.T) + b1
        H = 1.0 / (Z**2 + d_val**2 + 1.0)
        P = delta1 * Z + delta2
        A = P * H
        pred = np.matmul(A, W2) + b2
        
        loss = np.mean((pred - Y)**2)

        if tl[1] > 0:
            X_s = self.trainInputDerivative[:tl[1], :]
            Y_s = self.trainOutputDerivative[:tl[1], :]
            
            Z_s = np.matmul(X_s, W1.T) + b1
            H_s = 1.0 / (Z_s**2 + d_val**2 + 1.0)
            P_s = delta1 * Z_s + delta2
            
            A_prime_s = delta1 * H_s - 2.0 * P_s * Z_s * (H_s**2)
            pred_grad = np.matmul(A_prime_s * W2, W1)
            
            loss += np.mean(np.sum((pred_grad - Y_s)**2, axis=-1))

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight)**2

        return loss

    # 4. BACKPROPAGATION ANALÍTICO PARA O LM-BFGS
    def evalGradLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        gradLoss = np.zeros_like(fullWeight)
        
        W1, b1, d_val, delta1, delta2, W2, b2, idx1, idx2, idx3, idx4, idx5, idx6 = self._unpack(fullWeight)

        # --- Gradientes Base ---
        X = self.trainInput[:tl[0], :]
        Y = self.trainOutput[:tl[0]]
        
        Z = np.matmul(X, W1.T) + b1
        H = 1.0 / (Z**2 + d_val**2 + 1.0)
        P = delta1 * Z + delta2
        A = P * H
        A_prime = delta1 * H - 2.0 * P * Z * (H**2)
        
        pred = np.matmul(A, W2) + b2
        E = 2.0 / tl[0] * (pred - Y) 
        
        db2 = np.sum(E)
        dW2 = np.matmul(E.T, A)
        
        dA = E[:, np.newaxis] * W2
        dZ = dA * A_prime
        
        db1 = np.sum(dZ, axis=0)
        dW1 = np.matmul(dZ.T, X)
        
        ddelta1 = np.sum(dA * Z * H, axis=0)
        ddelta2 = np.sum(dA * H, axis=0)
        dd = np.sum(dA * P * (-2.0 * d_val * (H**2)), axis=0)

        # --- Gradientes Sobolev ---
        if tl[1] > 0:
            X_s = self.trainInputDerivative[:tl[1], :]
            Y_s = self.trainOutputDerivative[:tl[1], :]
            
            Z_s = np.matmul(X_s, W1.T) + b1
            H_s = 1.0 / (Z_s**2 + d_val**2 + 1.0)
            P_s = delta1 * Z_s + delta2
            A_prime_s = delta1 * H_s - 2.0 * P_s * Z_s * (H_s**2)
            
            pred_grad = np.matmul(A_prime_s * W2, W1)
            E_s = 2.0 / tl[1] * (pred_grad - Y_s)
            
            V = np.matmul(E_s, W1.T)
            K = V * W2
            
            # Segunda derivada da Ativação de Cauchy (Hessiana Local)
            A_double_prime_s = -2.0 * P_s * (H_s**2) - 4.0 * Z_s * H_s * A_prime_s
            dZ_s = K * A_double_prime_s
            
            db1 += np.sum(dZ_s, axis=0)
            dW1 += np.matmul((A_prime_s * W2).T, E_s) + np.matmul(dZ_s.T, X_s)
            dW2 += np.sum(V * A_prime_s, axis=0)
            
            ddelta1 += np.sum(K * (H_s - 2.0 * (Z_s**2) * (H_s**2)), axis=0)
            ddelta2 += np.sum(K * (-2.0 * Z_s * (H_s**2)), axis=0)
            dd += np.sum(K * (2.0 * delta1 * d_val * (H_s**2) - 4.0 * d_val * H_s * A_prime_s), axis=0)

        # --- Regularização ---
        if self.regParam > 0:
            dW1 += 2.0 * self.regParam * W1
            db1 += 2.0 * self.regParam * b1
            dd += 2.0 * self.regParam * d_val
            ddelta1 += 2.0 * self.regParam * delta1
            ddelta2 += 2.0 * self.regParam * delta2
            dW2 += 2.0 * self.regParam * W2
            db2 += 2.0 * self.regParam * b2

        gradLoss[:idx1] = dW1.flatten()
        gradLoss[idx1:idx2] = db1
        gradLoss[idx2:idx3] = dd
        gradLoss[idx3:idx4] = ddelta1
        gradLoss[idx4:idx5] = ddelta2
        gradLoss[idx5:idx6] = dW2
        gradLoss[idx6] = db2[0] if isinstance(db2, np.ndarray) else db2

        return gradLoss

    def train(self, rtolTraining):
        n, m = self.netDim[:2]
        
        idx1 = m*n
        idx2 = idx1 + m
        idx3 = idx2 + m
        idx4 = idx3 + m
        idx5 = idx4 + m
        idx6 = idx5 + m
        
        fullWeight = np.zeros(idx6 + 1)
        fullWeight[:idx1] = np.reshape(self.W1, (m*n,))
        fullWeight[idx1:idx2] = self.b1
        fullWeight[idx2:idx3] = self.d
        fullWeight[idx3:idx4] = self.delta1
        fullWeight[idx4:idx5] = self.delta2
        fullWeight[idx5:idx6] = self.W2
        fullWeight[idx6] = self.b2[0] if isinstance(self.b2, np.ndarray) else self.b2

        gradLossNorm, fullWeight = super().train(fullWeight, rtolTraining)

        self.W1, self.b1, self.d, self.delta1, self.delta2, self.W2, self.b2, *_ = self._unpack(fullWeight)

        return gradLossNorm