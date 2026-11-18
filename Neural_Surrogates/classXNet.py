#!/usr/bin/python3
# ----------------------------------- #
#  XNet (Cauchy) Surrogate Model      #
# ----------------------------------- #

import numpy as np
from classModel import MyModel

np.random.seed(0)


class classXNet(MyModel):

    def __init__(self, n, num_basis, f0, x0, simplexMemory, regParam, maxIterTraining):
        super().__init__(n, num_basis, f0, x0, simplexMemory)

        self.regParam = regParam
        self.maxIterTraining = maxIterTraining

        m = self.netDim[1]
        lim_delta = np.sqrt(3.0 / n)
        lim_W = np.sqrt(3.0 / m)

        # Parameter initialization
        center_noise = np.random.uniform(low=-0.1, high=0.1, size=(m, n))
        self.C = np.tile(x0, (m, 1)) + center_noise
        self.d = np.ones(m)

        self.delta1 = np.random.uniform(low=-lim_delta, high=lim_delta, size=(m, n))
        self.delta2 = np.random.uniform(low=-lim_delta, high=lim_delta, size=(m,))
        self.W = np.random.uniform(low=-lim_W, high=lim_W, size=(m,))
        self.b = np.zeros(1)

    def _unpack_weights(self, fullWeight):
        n, m = self.netDim[:2]
        C = np.reshape(fullWeight[: m * n], (m, n))
        d = fullWeight[m * n : m * n + m]
        delta1 = np.reshape(fullWeight[m * n + m : 2 * m * n + m], (m, n))
        delta2 = fullWeight[2 * m * n + m : 2 * m * n + 2 * m]
        W = fullWeight[2 * m * n + 2 * m : 2 * m * n + 3 * m]
        b = fullWeight[-1]
        return C, d, delta1, delta2, W, b

    def __call__(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]

        diff = x[:, np.newaxis, :] - self.C
        sq_dist = np.sum(diff ** 2, axis=-1)

        H = 1.0 / (sq_dist + self.d ** 2 + 1e-8)
        P1 = np.sum(self.delta1 * diff, axis=-1)

        val = np.sum(self.W * (P1 + self.delta2) * H, axis=-1) + self.b
        return np.squeeze(val, axis=0) if is_single else val

    def grad(self, x):
        is_single = len(np.shape(x)) == 1
        if is_single:
            x = x[np.newaxis, :]

        diff = x[:, np.newaxis, :] - self.C
        sq_dist = np.sum(diff ** 2, axis=-1)
        H = 1.0 / (sq_dist + self.d ** 2 + 1e-8)

        P1 = np.sum(self.delta1 * diff, axis=-1)
        Q = P1 + self.delta2

        dfdx = self.W[np.newaxis, :, np.newaxis] * (
            self.delta1[np.newaxis, :, :] * H[:, :, np.newaxis]
            - 2.0 * Q[:, :, np.newaxis] * (H ** 2)[:, :, np.newaxis] * diff
        )

        val = np.sum(dfdx, axis=1)
        return np.squeeze(val, axis=0) if is_single else val

    def evalLoss(self, fullWeight):
        tl = self.trainLength
        C, d, delta1, delta2, W, b = self._unpack_weights(fullWeight)

        # Function-value MSE loss
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]

        diff = X[:, np.newaxis, :] - C
        sq_dist = np.sum(diff ** 2, axis=-1)
        H = 1.0 / (sq_dist + d ** 2 + 1e-8)

        P1 = np.sum(delta1 * diff, axis=-1)
        pred = np.sum(W * (P1 + delta2) * H, axis=-1) + b
        loss = np.mean((pred - Y) ** 2)

        # Sobolev derivative-matching loss
        if tl[1] > 0:
            X_sob = self.trainInputDerivative[: tl[1], :]
            Y_sob = self.trainOutputDerivative[: tl[1], :]

            diff_sob = X_sob[:, np.newaxis, :] - C
            sq_dist_sob = np.sum(diff_sob ** 2, axis=-1)
            H_sob = 1.0 / (sq_dist_sob + d ** 2 + 1e-8)

            P1_sob = np.sum(delta1 * diff_sob, axis=-1)
            Q_sob = P1_sob + delta2

            pred_grad = np.sum(
                W[np.newaxis, :, np.newaxis]
                * (
                    delta1[np.newaxis, :, :] * H_sob[:, :, np.newaxis]
                    - 2.0 * Q_sob[:, :, np.newaxis] * (H_sob ** 2)[:, :, np.newaxis] * diff_sob
                ),
                axis=1,
            )
            loss += np.mean(np.sum((pred_grad - Y_sob) ** 2, axis=-1))

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight) ** 2

        return loss

    def evalGradLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        gradLoss = np.zeros_like(fullWeight)

        C, d_val, delta1, delta2, W, b = self._unpack_weights(fullWeight)

        # Forward pass on function values
        X = self.trainInput[: tl[0], :]
        Y = self.trainOutput[: tl[0]]
        diff = X[:, np.newaxis, :] - C
        sq_dist = np.sum(diff ** 2, axis=-1)
        H = 1.0 / (sq_dist + d_val ** 2 + 1e-8)
        P1 = np.sum(delta1 * diff, axis=-1)
        Q = P1 + delta2
        Phi = Q * H
        pred = np.sum(W * Phi, axis=-1) + b

        err = (2.0 / tl[0]) * (pred - Y)

        # Base gradients
        db = np.sum(err)
        dW = np.sum(err[:, np.newaxis] * Phi, axis=0)

        err_W = err[:, np.newaxis] * W
        ddelta2 = np.sum(err_W * H, axis=0)
        ddelta1 = np.sum((err_W * H)[:, :, np.newaxis] * diff, axis=0)

        dHdd = -2.0 * d_val * (H ** 2)
        dd = np.sum(err_W * Q * dHdd, axis=0)

        dC = np.sum(
            err_W[:, :, np.newaxis]
            * (
                -delta1[np.newaxis, :, :] * H[:, :, np.newaxis]
                + 2.0 * Q[:, :, np.newaxis] * (H ** 2)[:, :, np.newaxis] * diff
            ),
            axis=0,
        )

        # Sobolev gradients
        if tl[1] > 0:
            X_sob = self.trainInputDerivative[: tl[1], :]
            Y_sob = self.trainOutputDerivative[: tl[1], :]

            diff_s = X_sob[:, np.newaxis, :] - C
            sq_dist_s = np.sum(diff_s ** 2, axis=-1)
            H_s = 1.0 / (sq_dist_s + d_val ** 2 + 1e-8)
            P1_s = np.sum(delta1 * diff_s, axis=-1)
            Q_s = P1_s + delta2

            dfdx_s = (
                delta1[np.newaxis, :, :] * H_s[:, :, np.newaxis]
                - 2.0 * Q_s[:, :, np.newaxis] * (H_s ** 2)[:, :, np.newaxis] * diff_s
            )

            pred_grad = np.sum(W[np.newaxis, :, np.newaxis] * dfdx_s, axis=1)
            err_sob = (2.0 / tl[1]) * (pred_grad - Y_sob)

            dW += np.sum(err_sob[:, np.newaxis, :] * dfdx_s, axis=(0, 2))

            err_sob_W = err_sob[:, np.newaxis, :] * W[np.newaxis, :, np.newaxis]
            sum_err_diff_W = np.sum(err_sob_W * diff_s, axis=-1)
            sum_err_delta_W = np.sum(err_sob_W * delta1[np.newaxis, :, :], axis=-1)

            ddelta2 += np.sum(-2.0 * (H_s ** 2) * sum_err_diff_W, axis=0)

            d_val_br = d_val[np.newaxis, :, np.newaxis]
            dd += np.sum(
                err_sob_W
                * (
                    -2.0 * d_val_br * (H_s ** 2)[:, :, np.newaxis] * delta1[np.newaxis, :, :]
                    + 8.0 * d_val_br * Q_s[:, :, np.newaxis] * (H_s ** 3)[:, :, np.newaxis] * diff_s
                ),
                axis=(0, 2),
            )

            ddelta1 += (
                np.sum(err_sob_W * H_s[:, :, np.newaxis], axis=0)
                - np.sum(2.0 * (H_s ** 2)[:, :, np.newaxis] * sum_err_diff_W[:, :, np.newaxis] * diff_s, axis=0)
            )

            dC += np.sum(
                2.0 * (H_s ** 2)[:, :, np.newaxis]
                * (
                    delta1[np.newaxis, :, :] * sum_err_diff_W[:, :, np.newaxis]
                    + diff_s * sum_err_delta_W[:, :, np.newaxis]
                )
                - 8.0 * Q_s[:, :, np.newaxis] * (H_s ** 3)[:, :, np.newaxis] * diff_s * sum_err_diff_W[:, :, np.newaxis]
                + 2.0 * Q_s[:, :, np.newaxis] * (H_s ** 2)[:, :, np.newaxis] * err_sob_W,
                axis=0,
            )

        # L2 regularization gradients
        if self.regParam > 0:
            dC += 2.0 * self.regParam * C
            dd += 2.0 * self.regParam * d_val
            ddelta1 += 2.0 * self.regParam * delta1
            ddelta2 += 2.0 * self.regParam * delta2
            dW += 2.0 * self.regParam * W
            db += 2.0 * self.regParam * b

        gradLoss[: m * n] = dC.flatten()
        gradLoss[m * n : m * n + m] = dd
        gradLoss[m * n + m : 2 * m * n + m] = ddelta1.flatten()
        gradLoss[2 * m * n + m : 2 * m * n + 2 * m] = ddelta2
        gradLoss[2 * m * n + 2 * m : 2 * m * n + 3 * m] = dW
        gradLoss[-1] = db

        return gradLoss

    def train(self, rtolTraining):
        n, m = self.netDim[:2]

        fullWeight = np.zeros(2 * m * n + 3 * m + 1)
        fullWeight[: m * n] = np.reshape(self.C, (m * n,))
        fullWeight[m * n : m * n + m] = self.d
        fullWeight[m * n + m : 2 * m * n + m] = np.reshape(self.delta1, (m * n,))
        fullWeight[2 * m * n + m : 2 * m * n + 2 * m] = self.delta2
        fullWeight[2 * m * n + 2 * m : 2 * m * n + 3 * m] = self.W
        fullWeight[-1] = self.b[0] if isinstance(self.b, np.ndarray) else self.b

        gradLossNorm, fullWeight = super().train(fullWeight, rtolTraining)

        self.C, self.d, self.delta1, self.delta2, self.W, self.b = self._unpack_weights(fullWeight)
        return gradLossNorm