#!/usr/bin/python3
# ------------------------------------------------- #
#  Two-hidden-layer neural network surrogate (2C)   #
# ------------------------------------------------- #

import numpy as np
from classModel import MyModel

np.random.seed(0)


class MyNeuralNet2C(MyModel):

    def __init__(self, n, f0, x0, activationType, simplexMemory, regParam, ratioHiddenInputSize, maxIterTraining):
        super().__init__(n, ratioHiddenInputSize * n, f0, x0, simplexMemory)

        self.activationType = activationType
        self.maxIterTraining = maxIterTraining
        self.regParam = regParam

        if activationType == "sigmoid":
            self.activation = lambda x: np.exp(-np.logaddexp(0, -x))
            self.activationD1 = lambda x: np.exp(-np.logaddexp(0, -x)) * (1.0 - np.exp(-np.logaddexp(0, -x)))
            self.activationD2 = lambda x: (
                np.exp(-np.logaddexp(0, -x))
                * (1.0 - np.exp(-np.logaddexp(0, -x)))
                * (1.0 - 2.0 * np.exp(-np.logaddexp(0, -x)))
            )
        elif activationType == "arctan":
            self.activation = lambda x: np.arctan(x)
            self.activationD1 = lambda x: 1.0 / (x * x + 1.0)
            self.activationD2 = lambda x: -2.0 * x / (x * x + 1.0) ** 2
        elif activationType == "softplus":
            self.activation = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
            self.activationD1 = lambda x: np.exp(-np.logaddexp(0, -x))
            self.activationD2 = lambda x: np.exp(-np.logaddexp(0, -x)) * (1.0 - np.exp(-np.logaddexp(0, -x)))
        elif activationType == "silu":
            self.activation = lambda x: x * np.exp(-np.logaddexp(0, -x))
            self.activationD1 = lambda x: (
                np.exp(-np.logaddexp(0, -x))
                + x * np.exp(-np.logaddexp(0, -x)) * (1.0 - np.exp(-np.logaddexp(0, -x)))
            )
            self.activationD2 = lambda x: (
                2.0 * np.exp(-np.logaddexp(0, -x)) * (1.0 - np.exp(-np.logaddexp(0, -x)))
                + x
                * np.exp(-np.logaddexp(0, -x))
                * (1.0 - np.exp(-np.logaddexp(0, -x)))
                * (1.0 - 2.0 * np.exp(-np.logaddexp(0, -x)))
            )
        else:
            raise ValueError(f"Unknown activation function: {activationType}")

        n, m = self.netDim[:2]
        guessMaxW1 = np.sqrt(3.0 / n)
        guessMaxW2 = np.sqrt(3.0 / m)
        guessMaxW3 = np.sqrt(3.0 / m)

        if activationType in ("softplus", "silu"):
            guessMaxW1 *= np.sqrt(2.0)
            guessMaxW2 *= np.sqrt(2.0)
            guessMaxW3 *= np.sqrt(2.0)

        self.W1 = np.random.uniform(low=-guessMaxW1, high=guessMaxW1, size=(m, n))
        self.b1 = np.zeros(m)
        self.W2 = np.random.uniform(low=-guessMaxW2, high=guessMaxW2, size=(m, m))
        self.b2 = np.zeros(m)
        self.W3 = np.random.uniform(low=-guessMaxW3, high=guessMaxW3, size=(m,))
        self.b3 = np.zeros(1)

    def _unpack_weights(self, fullWeight):
        n, m = self.netDim[:2]
        idx1 = m * n
        idx2 = idx1 + m
        idx3 = idx2 + m * m
        idx4 = idx3 + m
        idx5 = idx4 + m

        W1 = np.reshape(fullWeight[:idx1], (m, n))
        b1 = fullWeight[idx1:idx2]
        W2 = np.reshape(fullWeight[idx2:idx3], (m, m))
        b2 = fullWeight[idx3:idx4]
        W3 = fullWeight[idx4:idx5]
        b3 = fullWeight[idx5]

        return W1, b1, W2, b2, W3, b3, idx1, idx2, idx3, idx4, idx5

    def __call__(self, x):
        z1 = np.einsum("ij,...j->...i", self.W1, x) + self.b1
        a1 = self.activation(z1)
        z2 = np.einsum("ij,...j->...i", self.W2, a1) + self.b2
        a2 = self.activation(z2)
        return np.einsum("k,...k->...", self.W3, a2) + self.b3

    def grad(self, x):
        z1 = np.einsum("ij,...j->...i", self.W1, x) + self.b1
        a1 = self.activation(z1)
        z2 = np.einsum("ij,...j->...i", self.W2, a1) + self.b2

        d2 = self.activationD1(z2) * self.W3
        d1 = self.activationD1(z1) * np.einsum("ji,...j->...i", self.W2, d2)
        return np.einsum("ji,...j->...i", self.W1, d1)

    def evalLoss(self, fullWeight):
        tl = self.trainLength
        W1, b1, W2, b2, W3, b3, _, _, _, _, _ = self._unpack_weights(fullWeight)

        # Function-value MSE loss
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]

        Z1 = np.matmul(X, W1.T) + b1
        A1 = self.activation(Z1)
        Z2 = np.matmul(A1, W2.T) + b2
        A2 = self.activation(Z2)
        pred = np.matmul(A2, W3) + b3

        loss = (1.0 / tl[0]) * np.linalg.norm(pred - Y) ** 2

        # Sobolev derivative-matching loss
        if tl[1] > 0:
            X_sob = self.trainInputDerivative[: tl[1], :]
            Y_sob = self.trainOutputDerivative[: tl[1], :]

            Z1_s = np.matmul(X_sob, W1.T) + b1
            A1_s = self.activation(Z1_s)
            Z2_s = np.matmul(A1_s, W2.T) + b2

            D2 = self.activationD1(Z2_s) * W3
            D1 = self.activationD1(Z1_s) * np.matmul(D2, W2)
            pred_grad = np.matmul(D1, W1)

            loss += (1.0 / tl[1]) * np.linalg.norm(pred_grad - Y_sob, ord="fro") ** 2

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight) ** 2

        return loss

    def evalGradLoss(self, fullWeight):
        tl = self.trainLength
        W1, b1, W2, b2, W3, b3, idx1, idx2, idx3, idx4, idx5 = self._unpack_weights(fullWeight)
        gradLoss = np.zeros_like(fullWeight)

        # Base gradients
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]

        Z1 = np.matmul(X, W1.T) + b1
        A1 = self.activation(Z1)
        Z2 = np.matmul(A1, W2.T) + b2
        A2 = self.activation(Z2)
        pred = np.matmul(A2, W3) + b3

        E_std = (2.0 / tl[0]) * (pred - Y)
        db3 = np.sum(E_std)
        dW3 = np.matmul(E_std.T, A2)

        up_A2 = E_std[:, np.newaxis] * W3
        dZ2 = up_A2 * self.activationD1(Z2)
        db2 = np.sum(dZ2, axis=0)
        dW2 = np.matmul(dZ2.T, A1)

        up_A1 = np.matmul(dZ2, W2)
        dZ1 = up_A1 * self.activationD1(Z1)
        db1 = np.sum(dZ1, axis=0)
        dW1 = np.matmul(dZ1.T, X)

        gradLoss[:idx1] = dW1.flatten()
        gradLoss[idx1:idx2] = db1
        gradLoss[idx2:idx3] = dW2.flatten()
        gradLoss[idx3:idx4] = db2
        gradLoss[idx4:idx5] = dW3
        gradLoss[idx5] = db3

        # Sobolev gradients (second-order chain rule)
        if tl[1] > 0:
            X_s = self.trainInputDerivative[: tl[1], :]
            Y_s = self.trainOutputDerivative[: tl[1], :]

            Z1_s = np.matmul(X_s, W1.T) + b1
            A1_s = self.activation(Z1_s)
            Z2_s = np.matmul(A1_s, W2.T) + b2

            D2 = self.activationD1(Z2_s) * W3
            D1 = self.activationD1(Z1_s) * np.matmul(D2, W2)
            pred_grad = np.matmul(D1, W1)

            E_sob = (2.0 / tl[1]) * (pred_grad - Y_s)

            dW1_sob = np.matmul(D1.T, E_sob)
            up_D1 = np.matmul(E_sob, W1.T)

            V1 = up_D1 * self.activationD2(Z1_s) * np.matmul(D2, W2)
            K1 = up_D1 * self.activationD1(Z1_s)

            dW2_part1 = np.matmul(D2.T, K1)
            up_D2 = np.matmul(K1, W2.T)

            dW3_sob = np.sum(up_D2 * self.activationD1(Z2_s), axis=0)

            V2 = up_D2 * self.activationD2(Z2_s) * W3
            dW2_part2 = np.matmul(V2.T, A1_s)
            db2_sob = np.sum(V2, axis=0)

            up_A1_s = np.matmul(V2, W2)
            V1_extra = up_A1_s * self.activationD1(Z1_s)

            dW1_sob += np.matmul((V1 + V1_extra).T, X_s)
            db1_sob = np.sum(V1 + V1_extra, axis=0)

            gradLoss[:idx1] += dW1_sob.flatten()
            gradLoss[idx1:idx2] += db1_sob
            gradLoss[idx2:idx3] += (dW2_part1 + dW2_part2).flatten()
            gradLoss[idx3:idx4] += db2_sob
            gradLoss[idx4:idx5] += dW3_sob

        if self.regParam > 0:
            gradLoss += 2.0 * self.regParam * fullWeight

        return gradLoss

    def train(self, rtolTraining):
        n, m = self.netDim[:2]
        idx1 = m * n
        idx2 = idx1 + m
        idx3 = idx2 + m * m
        idx4 = idx3 + m
        idx5 = idx4 + m

        fullWeight = np.zeros(idx5 + 1)
        fullWeight[:idx1] = np.reshape(self.W1, (m * n,))
        fullWeight[idx1:idx2] = self.b1
        fullWeight[idx2:idx3] = np.reshape(self.W2, (m * m,))
        fullWeight[idx3:idx4] = self.b2
        fullWeight[idx4:idx5] = self.W3
        fullWeight[idx5] = self.b3[0] if isinstance(self.b3, np.ndarray) else self.b3

        gradLossNorm, fullWeight = super().train(fullWeight, rtolTraining)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, _, _, _, _, _ = self._unpack_weights(fullWeight)
        return gradLossNorm