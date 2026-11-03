#!/usr/bin/python3
# ----------------------------------- #
#  XNet (Cauchy) Surrogate Model      #
# ----------------------------------- #

import numpy as np
from classModel import MyModel #

class MyXNet(MyModel):
    
    def __init__(self, n, num_basis, f0, x0, simplexMemory, regParam, maxIterTraining):
        super().__init__(n, num_basis, f0, x0, simplexMemory)

        self.regParam = regParam
        self.maxIterTraining = maxIterTraining
        
        m = self.netDim[1]
        
        # --- INICIALIZAÇÃO DOS PARÂMETROS ---
        # Centros c (m, n)
        self.C = np.random.uniform(low=-1.0, high=1.0, size=(m, n))
        
        # Largura d (m) -> Evita divisão por zero e controla a "gordura" do pico
        self.d = np.ones(m) * 0.1 
        
        # delta_1 (m, n) -> Peso da parte Dispersiva (Ímpar). Como (x-c) é vetor, delta1 também é.
        self.delta1 = np.random.uniform(low=-1.0, high=1.0, size=(m, n))
        
        # delta_2 (m) -> Peso da parte Absortiva (Par).
        self.delta2 = np.random.uniform(low=-1.0, high=1.0, size=(m,))
        
        # Viés b (1)
        self.b = np.zeros(1)

    # 1. FORWARD PASS: f(x)
    def __call__(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]
            
        diff = x[:, np.newaxis, :] - self.C       # (N, m, n) -> (x - c)
        sq_dist = np.sum(diff**2, axis=-1)        # (N, m) -> ||x - c||^2
        
        # H = 1 / ((x-c)^2 + d^2)
        H = 1.0 / (sq_dist + self.d**2)           # (N, m)
        
        # Termo Dispersivo (Produto escalar delta1 * (x-c))
        P1 = np.sum(self.delta1 * diff, axis=-1)  # (N, m)
        
        # Saída = Sum( (P1 + delta2) * H ) + b
        val = np.sum((P1 + self.delta2) * H, axis=-1) + self.b
        
        return np.squeeze(val, axis=0) if is_single else val

    # 2. GRADIENTE DO MODELO 
    def grad(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]
            
        diff = x[:, np.newaxis, :] - self.C
        sq_dist = np.sum(diff**2, axis=-1)
        H = 1.0 / (sq_dist + self.d**2)
        
        P1 = np.sum(self.delta1 * diff, axis=-1)
        Q = P1 + self.delta2  # Numerador consolidado
        
        # Gradiente em relação a x: delta1 * H - 2 * Q * H^2 * (x-c)
        dfdx = self.delta1[np.newaxis, :, :] * H[:, :, np.newaxis] \
               - 2.0 * Q[:, :, np.newaxis] * (H**2)[:, :, np.newaxis] * diff
               
        val = np.sum(dfdx, axis=1) # Soma sobre todas as bases
        
        return np.squeeze(val, axis=0) if is_single else val

    # 3. PERDA (MSE + Sobolev Learning + Regularização)
    def evalLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        
        # Desempacota o vetor 1D exigido pelo LM-BFGS
        C = np.reshape(fullWeight[:m*n], (m, n))
        d = fullWeight[m*n : m*n+m]
        delta1 = np.reshape(fullWeight[m*n+m : 2*m*n+m], (m, n))
        delta2 = fullWeight[2*m*n+m : 2*m*n+2*m]
        b = fullWeight[-1]

        # --- Perda Padrão (Valores F) ---
        X = self.trainInput[:tl[0], :]
        Y = self.trainOutput[:tl[0]]
        
        diff = X[:, np.newaxis, :] - C
        sq_dist = np.sum(diff**2, axis=-1)
        H = 1.0 / (sq_dist + d**2)
        
        P1 = np.sum(delta1 * diff, axis=-1)
        pred = np.sum((P1 + delta2) * H, axis=-1) + b
        loss = np.mean((pred - Y)**2)

        # --- Perda Sobolev (Derivadas G) ---
        if tl[1] > 0:
            X_sob = self.trainInputDerivative[:tl[1], :]
            Y_sob = self.trainOutputDerivative[:tl[1], :]
            
            diff_sob = X_sob[:, np.newaxis, :] - C
            sq_dist_sob = np.sum(diff_sob**2, axis=-1)
            H_sob = 1.0 / (sq_dist_sob + d**2)
            
            P1_sob = np.sum(delta1 * diff_sob, axis=-1)
            Q_sob = P1_sob + delta2
            
            pred_grad = np.sum(delta1[np.newaxis, :, :] * H_sob[:, :, np.newaxis] \
                               - 2.0 * Q_sob[:, :, np.newaxis] * (H_sob**2)[:, :, np.newaxis] * diff_sob, axis=1)
            
            loss += np.mean(np.sum((pred_grad - Y_sob)**2, axis=-1))

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight)**2

        return loss

    # 4. BACKPROPAGATION ANALÍTICO PARA O LM-BFGS
    def evalGradLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        gradLoss = np.zeros_like(fullWeight)
        
        C = np.reshape(fullWeight[:m*n], (m, n))
        d_val = fullWeight[m*n : m*n+m]
        delta1 = np.reshape(fullWeight[m*n+m : 2*m*n+m], (m, n))
        delta2 = fullWeight[2*m*n+m : 2*m*n+2*m]
        b = fullWeight[-1]

        # Forward
        X = self.trainInput[:tl[0], :]
        Y = self.trainOutput[:tl[0]]
        diff = X[:, np.newaxis, :] - C
        sq_dist = np.sum(diff**2, axis=-1)
        H = 1.0 / (sq_dist + d_val**2)
        P1 = np.sum(delta1 * diff, axis=-1)
        Q = P1 + delta2
        pred = np.sum(Q * H, axis=-1) + b
        
        err = 2.0 / tl[0] * (pred - Y) # dL/dPred
        
        # --- Gradientes Base ---
        db = np.sum(err)
        ddelta2 = np.sum(err[:, np.newaxis] * H, axis=0)
        ddelta1 = np.sum((err[:, np.newaxis] * H)[:, :, np.newaxis] * diff, axis=0)
        
        # Derivada em relação a d: Q * (-H^2) * (2d)
        dHdd = -2.0 * d_val * (H**2)
        dd = np.sum(err[:, np.newaxis] * Q * dHdd, axis=0)
        
        # Derivada em relação a C (Centros)
        # Termo 1: derivada do numerador -> delta1 * (-1)
        # Termo 2: derivada do denominador -> -2 * Q * H^2 * (-(x-c))
        dC = np.sum(err[:, np.newaxis, np.newaxis] * \
             (-delta1[np.newaxis, :, :] * H[:, :, np.newaxis] \
              + 2.0 * Q[:, :, np.newaxis] * (H**2)[:, :, np.newaxis] * diff), axis=0)

        # --- Gradientes Sobolev ---
        # (Para estabilidade, calculamos as derivadas secundárias apenas se requisitado)
        if tl[1] > 0:
            X_sob = self.trainInputDerivative[:tl[1], :]
            Y_sob = self.trainOutputDerivative[:tl[1], :]
            
            diff_s = X_sob[:, np.newaxis, :] - C
            sq_dist_s = np.sum(diff_s**2, axis=-1)
            H_s = 1.0 / (sq_dist_s + d_val**2)
            P1_s = np.sum(delta1 * diff_s, axis=-1)
            Q_s = P1_s + delta2
            
            dfdx_s = delta1[np.newaxis, :, :] * H_s[:, :, np.newaxis] \
                     - 2.0 * Q_s[:, :, np.newaxis] * (H_s**2)[:, :, np.newaxis] * diff_s
            pred_grad = np.sum(dfdx_s, axis=1)
            
            err_sob = 2.0 / tl[1] * (pred_grad - Y_sob) # (N, n)
            
            # Produtos vetoriais para Sobolev
            sum_err_diff = np.sum(err_sob[:, np.newaxis, :] * diff_s, axis=-1) # (N, m)
            sum_err_delta = np.sum(err_sob[:, np.newaxis, :] * delta1[np.newaxis, :, :], axis=-1) # (N, m)
            
            ddelta2 += np.sum(-2.0 * (H_s**2) * sum_err_diff, axis=0)
            
            
            d_val_br = d_val[np.newaxis, :, np.newaxis]
            dd += np.sum(err_sob[:, np.newaxis, :] * (
                -2.0 * d_val_br * (H_s**2)[:, :, np.newaxis] * delta1[np.newaxis, :, :] 
                + 8.0 * d_val_br * Q_s[:, :, np.newaxis] * (H_s**3)[:, :, np.newaxis] * diff_s
            ), axis=(0,2))
            
            ddelta1 += np.sum(err_sob[:, np.newaxis, :] * H_s[:, :, np.newaxis], axis=0) \
                       - np.sum(2.0 * (H_s**2)[:, :, np.newaxis] * sum_err_diff[:, :, np.newaxis] * diff_s, axis=0)
            
            dC += np.sum(
                2.0 * (H_s**2)[:, :, np.newaxis] * (delta1[np.newaxis, :, :] * sum_err_diff[:, :, np.newaxis] + diff_s * sum_err_delta[:, :, np.newaxis])
                - 8.0 * Q_s[:, :, np.newaxis] * (H_s**3)[:, :, np.newaxis] * diff_s * sum_err_diff[:, :, np.newaxis]
                + 2.0 * Q_s[:, :, np.newaxis] * (H_s**2)[:, :, np.newaxis] * err_sob[:, np.newaxis, :],
                axis=0
            )

        # --- Regularização ---
        if self.regParam > 0:
            dC += 2.0 * self.regParam * C
            dd += 2.0 * self.regParam * d_val
            ddelta1 += 2.0 * self.regParam * delta1
            ddelta2 += 2.0 * self.regParam * delta2
            db += 2.0 * self.regParam * b

        # Empacota gradientes
        gradLoss[:m*n] = dC.flatten()
        gradLoss[m*n : m*n+m] = dd
        gradLoss[m*n+m : 2*m*n+m] = ddelta1.flatten()
        gradLoss[2*m*n+m : 2*m*n+2*m] = ddelta2
        gradLoss[-1] = db

        return gradLoss

    def train(self, rtolTraining):
        n, m = self.netDim[:2]
        
        fullWeight = np.zeros(2*m*n + 2*m + 1)
        fullWeight[:m*n] = np.reshape(self.C, (m*n,))
        fullWeight[m*n : m*n+m] = self.d
        fullWeight[m*n+m : 2*m*n+m] = np.reshape(self.delta1, (m*n,))
        fullWeight[2*m*n+m : 2*m*n+2*m] = self.delta2
        fullWeight[-1] = self.b[0] if isinstance(self.b, np.ndarray) else self.b

        gradLossNorm, fullWeight = super().train(fullWeight, rtolTraining)

        self.C = np.reshape(fullWeight[:m*n], (m, n))
        self.d = fullWeight[m*n : m*n+m]
        self.delta1 = np.reshape(fullWeight[m*n+m : 2*m*n+m], (m, n))
        self.delta2 = fullWeight[2*m*n+m : 2*m*n+2*m]
        self.b = fullWeight[-1]

        return gradLossNorm