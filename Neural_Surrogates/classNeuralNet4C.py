#!/usr/bin/python3
# -------------------------------------------------- #
#  Four-hidden-layer neural network surrogate (4C)   #
# -------------------------------------------------- #

import numpy as np
from classModel import MyModel

np.random.seed(0)


class MyNeuralNet4C(MyModel):

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
        self.W3 = np.random.uniform(low=-guessMaxW3, high=guessMaxW3, size=(m, m))
        self.b3 = np.zeros(m)
        self.W4 = np.random.uniform(low=-guessMaxW3, high=guessMaxW3, size=(m, m))
        self.b4 = np.zeros(m)
        self.W5 = np.random.uniform(low=-guessMaxW3, high=guessMaxW3, size=(m,))
        self.b5 = np.zeros(1)

    def _unpack_weights(self, fullWeight):
        n, m = self.netDim[:2]

        idx1 = m * n
        idx2 = idx1 + m
        idx3 = idx2 + m * m
        idx4 = idx3 + m
        idx5 = idx4 + m * m
        idx6 = idx5 + m
        idx7 = idx6 + m * m
        idx8 = idx7 + m
        idx9 = idx8 + m

        W1 = np.reshape(fullWeight[:idx1], (m, n))
        b1 = fullWeight[idx1:idx2]
        W2 = np.reshape(fullWeight[idx2:idx3], (m, m))
        b2 = fullWeight[idx3:idx4]
        W3 = np.reshape(fullWeight[idx4:idx5], (m, m))
        b3 = fullWeight[idx5:idx6]
        W4 = np.reshape(fullWeight[idx6:idx7], (m, m))
        b4 = fullWeight[idx7:idx8]
        W5 = fullWeight[idx8:idx9]
        b5 = fullWeight[idx9]

        return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9

    def __call__(self, x):
        Z1 = np.einsum("ij,...j->...i", self.W1, x) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.einsum("ij,...j->...i", self.W2, A1) + self.b2
        A2 = self.activation(Z2)
        Z3 = np.einsum("ij,...j->...i", self.W3, A2) + self.b3
        A3 = self.activation(Z3)
        Z4 = np.einsum("ij,...j->...i", self.W4, A3) + self.b4
        A4 = self.activation(Z4)
        return np.einsum("k,...k->...", self.W5, A4) + self.b5

    def grad(self, x):
        Z1 = np.einsum("ij,...j->...i", self.W1, x) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.einsum("ij,...j->...i", self.W2, A1) + self.b2
        A2 = self.activation(Z2)
        Z3 = np.einsum("ij,...j->...i", self.W3, A2) + self.b3
        A3 = self.activation(Z3)
        Z4 = np.einsum("ij,...j->...i", self.W4, A3) + self.b4

        d4 = self.activationD1(Z4) * self.W5
        d3 = self.activationD1(Z3) * np.einsum("ji,...j->...i", self.W4, d4)
        d2 = self.activationD1(Z2) * np.einsum("ji,...j->...i", self.W3, d3)
        d1 = self.activationD1(Z1) * np.einsum("ji,...j->...i", self.W2, d2)
        return np.einsum("ji,...j->...i", self.W1, d1)

    def evalLoss(self, fullWeight):
        tl = self.trainLength
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, _, _, _, _, _, _, _, _, _ = self._unpack_weights(fullWeight)

        # Function-value MSE loss
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]

        Z1 = np.matmul(X, W1.T) + b1
        A1 = self.activation(Z1)
        Z2 = np.matmul(A1, W2.T) + b2
        A2 = self.activation(Z2)
        Z3 = np.matmul(A2, W3.T) + b3
        A3 = self.activation(Z3)
        Z4 = np.matmul(A3, W4.T) + b4
        A4 = self.activation(Z4)
        pred = np.matmul(A4, W5.T) + b5

        loss = (1.0 / tl[0]) * np.linalg.norm(pred - Y) ** 2

        # Sobolev derivative-matching loss
        if tl[1] > 0:
            X_s = self.trainInputDerivative[: tl[1], :]
            Y_s = self.trainOutputDerivative[: tl[1], :]

            Z1_s = np.matmul(X_s, W1.T) + b1
            A1_s = self.activation(Z1_s)
            Z2_s = np.matmul(A1_s, W2.T) + b2
            A2_s = self.activation(Z2_s)
            Z3_s = np.matmul(A2_s, W3.T) + b3
            A3_s = self.activation(Z3_s)
            Z4_s = np.matmul(A3_s, W4.T) + b4

            D4 = self.activationD1(Z4_s) * W5
            D3 = self.activationD1(Z3_s) * np.matmul(D4, W4)
            D2 = self.activationD1(Z2_s) * np.matmul(D3, W3)
            D1 = self.activationD1(Z1_s) * np.matmul(D2, W2)
            pred_grad = np.matmul(D1, W1)

            loss += (1.0 / tl[1]) * np.linalg.norm(pred_grad - Y_s, ord="fro") ** 2

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight) ** 2

        return loss

    def evalGradLoss(self, fullWeight):
        tl = self.trainLength
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9 = (
            self._unpack_weights(fullWeight)
        )
        gradLoss = np.zeros_like(fullWeight)

        # Base gradients
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]

        Z1 = np.matmul(X, W1.T) + b1
        A1 = self.activation(Z1)
        Z2 = np.matmul(A1, W2.T) + b2
        A2 = self.activation(Z2)
        Z3 = np.matmul(A2, W3.T) + b3
        A3 = self.activation(Z3)
        Z4 = np.matmul(A3, W4.T) + b4
        A4 = self.activation(Z4)
        pred = np.matmul(A4, W5.T) + b5

        lGrad = (2.0 / tl[0]) * (pred - Y)

        db5 = np.sum(lGrad)
        dw5 = np.matmul(lGrad.T, A4)

        s1 = lGrad[:, np.newaxis] * W5 * self.activationD1(Z4)
        db4 = np.sum(s1, axis=0)
        dw4 = np.matmul(s1.T, A3)

        s2 = np.matmul(s1, W4) * self.activationD1(Z3)
        db3 = np.sum(s2, axis=0)
        dw3 = np.matmul(s2.T, A2)

        s3 = np.matmul(s2, W3) * self.activationD1(Z2)
        db2 = np.sum(s3, axis=0)
        dw2 = np.matmul(s3.T, A1)

        s4 = np.matmul(s3, W2) * self.activationD1(Z1)
        db1 = np.sum(s4, axis=0)
        dw1 = np.matmul(s4.T, X)

        gradLoss[:idx1] = dw1.flatten()
        gradLoss[idx1:idx2] = db1
        gradLoss[idx2:idx3] = dw2.flatten()
        gradLoss[idx3:idx4] = db2
        gradLoss[idx4:idx5] = dw3.flatten()
        gradLoss[idx5:idx6] = db3
        gradLoss[idx6:idx7] = dw4.flatten()
        gradLoss[idx7:idx8] = db4
        gradLoss[idx8:idx9] = dw5.flatten()
        gradLoss[idx9:] = db5

        # Sobolev gradients (second-order chain rule)
        if tl[1] > 0:
            X_s = self.trainInputDerivative[: tl[1], :]
            Y_s = self.trainOutputDerivative[: tl[1], :]

            Z1_s = np.matmul(X_s, W1.T) + b1
            A1_s = self.activation(Z1_s)
            Z2_s = np.matmul(A1_s, W2.T) + b2
            A2_s = self.activation(Z2_s)
            Z3_s = np.matmul(A2_s, W3.T) + b3
            A3_s = self.activation(Z3_s)
            Z4_s = np.matmul(A3_s, W4.T) + b4

            D4 = self.activationD1(Z4_s) * W5
            D3 = self.activationD1(Z3_s) * np.matmul(D4, W4)
            D2 = self.activationD1(Z2_s) * np.matmul(D3, W3)
            D1 = self.activationD1(Z1_s) * np.matmul(D2, W2)
            pred_grad = np.matmul(D1, W1)

            E_sob = (2.0 / tl[1]) * (pred_grad - Y_s)

            # Outer derivative path
            dW1_out = np.matmul(D1.T, E_sob)
            U1 = np.matmul(E_sob, W1.T)
            V1 = U1 * np.matmul(D2, W2) * self.activationD2(Z1_s)
            K1 = U1 * self.activationD1(Z1_s)

            dW2_out = np.matmul(D2.T, K1)
            U2 = np.matmul(K1, W2.T)
            V2 = U2 * np.matmul(D3, W3) * self.activationD2(Z2_s)
            K2 = U2 * self.activationD1(Z2_s)

            dW3_out = np.matmul(D3.T, K2)
            U3 = np.matmul(K2, W3.T)
            V3 = U3 * np.matmul(D4, W4) * self.activationD2(Z3_s)
            K3 = U3 * self.activationD1(Z3_s)

            dW4_out = np.matmul(D4.T, K3)
            U4 = np.matmul(K3, W4.T)
            V4 = U4 * W5 * self.activationD2(Z4_s)
            K4 = U4 * self.activationD1(Z4_s)

            dW5_out = np.sum(K4, axis=0)

            # Forward activation path
            Z4_bar = V4
            dW4_fwd = np.matmul(Z4_bar.T, A3_s)
            db4_sob = np.sum(Z4_bar, axis=0)

            A3_bar = np.matmul(Z4_bar, W4)
            Z3_bar = V3 + A3_bar * self.activationD1(Z3_s)
            dW3_fwd = np.matmul(Z3_bar.T, A2_s)
            db3_sob = np.sum(Z3_bar, axis=0)

            A2_bar = np.matmul(Z3_bar, W3)
            Z2_bar = V2 + A2_bar * self.activationD1(Z2_s)
            dW2_fwd = np.matmul(Z2_bar.T, A1_s)
            db2_sob = np.sum(Z2_bar, axis=0)

            A1_bar = np.matmul(Z2_bar, W2)
            Z1_bar = V1 + A1_bar * self.activationD1(Z1_s)
            dW1_fwd = np.matmul(Z1_bar.T, X_s)
            db1_sob = np.sum(Z1_bar, axis=0)

            gradLoss[:idx1] += (dW1_out + dW1_fwd).flatten()
            gradLoss[idx1:idx2] += db1_sob
            gradLoss[idx2:idx3] += (dW2_out + dW2_fwd).flatten()
            gradLoss[idx3:idx4] += db2_sob
            gradLoss[idx4:idx5] += (dW3_out + dW3_fwd).flatten()
            gradLoss[idx5:idx6] += db3_sob
            gradLoss[idx6:idx7] += (dW4_out + dW4_fwd).flatten()
            gradLoss[idx7:idx8] += db4_sob
            gradLoss[idx8:idx9] += dW5_out

        if self.regParam > 0:
            gradLoss += 2.0 * self.regParam * fullWeight

        return gradLoss

    def train(self, rtolTraining):
        n, m = self.netDim[:2]

        idx1 = m * n
        idx2 = idx1 + m
        idx3 = idx2 + m * m
        idx4 = idx3 + m
        idx5 = idx4 + m * m
        idx6 = idx5 + m
        idx7 = idx6 + m * m
        idx8 = idx7 + m
        idx9 = idx8 + m

        fullWeight = np.zeros(idx9 + 1)
        fullWeight[:idx1] = np.reshape(self.W1, (m * n,))
        fullWeight[idx1:idx2] = self.b1
        fullWeight[idx2:idx3] = np.reshape(self.W2, (m * m,))
        fullWeight[idx3:idx4] = self.b2
        fullWeight[idx4:idx5] = np.reshape(self.W3, (m * m,))
        fullWeight[idx5:idx6] = self.b3
        fullWeight[idx6:idx7] = np.reshape(self.W4, (m * m,))
        fullWeight[idx7:idx8] = self.b4
        fullWeight[idx8:idx9] = self.W5
        fullWeight[idx9] = self.b5[0] if isinstance(self.b5, np.ndarray) else self.b5

        gradLossNorm, fullWeight = super().train(fullWeight, rtolTraining)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5, _, _, _, _, _, _, _, _, _ = (
            self._unpack_weights(fullWeight)
        )
        return gradLossNorm