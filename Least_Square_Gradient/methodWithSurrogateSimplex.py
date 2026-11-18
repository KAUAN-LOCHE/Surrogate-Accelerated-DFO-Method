#!/usr/bin/python3
# --------------------------------------------------------- #
#  Surrogate-accelerated finite-difference gradient method  #
# --------------------------------------------------------- #
""" Imports """
import numpy as np

""" Parameters """
sigmaMin = 1e-2
iterLineSearchMax = 100
extrapolMax = 500 # Maximum number of extrapolation by iteration
rho = 1e-4
M = 4

""" Main code """
def finiteDiffGradientWithSurrogate(f, x0, evalMax, tol, model):
    n = len(x0)
    I = np.eye(n)
    sigma = 1
    memory = 4*(n+1)
    f0 = f(x0)
    nEval = 1

    # =========================================================================
    # Surrogate steps tracking
    # =========================================================================
    if model == "Simplex":
        hist_x = []
        hist_f = []

        k = 0
        fSample = np.zeros(n)
        x_perturbed = np.zeros((n,n))

        g0 = np.zeros(n)
        reincident_flaw = 0
        while True:
            h = 2*tol/(5*sigma*np.sqrt(n))
            stepFound = False
            sigmaBefore = sigma
            for l in range(iterLineSearchMax):
                for j in range(n):
                    x_perturbed[j] = x0 + h*I[:,j]
                    fSample[j] = f(x_perturbed[j])
                    hist_x.append(x_perturbed[j].copy())
                    hist_f.append(fSample[j])
                    nEval += 1
                    if nEval >= evalMax:
                      
                        return x0, f0

                while len(hist_x) > memory:
                    hist_x.pop(0)
                    hist_f.pop(0)

                g0 = (fSample- f0)/h
               
                if np.linalg.norm(g0) >= 4/5 * tol:
                    xNew = x0 - 1/sigma * g0
                    fNew = f(xNew)
                    nEval +=1

                    if nEval >= evalMax:
                        return x0, f0

                    if f0 - fNew >= 1/(8*sigma) * np.linalg.norm(g0)**2:
                        hist_x.append(x0.copy())
                        hist_f.append(f0)

                        while len(hist_x) > memory:
                            hist_x.pop(0)
                            hist_f.pop(0)

                        x0 = xNew
                        f0 = fNew
                        stepFound = True
                        break
                    hist_x.append(xNew.copy()) 
                    hist_f.append(fNew)
                    while len(hist_x) > memory:
                        hist_x.pop(0)
                        hist_f.pop(0)
                h/=2
                sigma *= 2

            if not stepFound:
                reincident_flaw += 1
                
                if reincident_flaw >= 2: # Limited number of attempts to find a step
                    print("No step found after multiple attempts. Exiting optimization.")
                    return x0, f0
                sigma = max(sigmaBefore/2, sigmaMin)
                continue
            else:
                reincident_flaw = 0

            L = sigma 
            limit_extrapolation = M
            v = 0
            for j in range(extrapolMax):
                while v < limit_extrapolation:
                    # Obtain the differences in x and f from the history
                    dx_mat = np.array([hist_x[idx] - x0 for idx in range(len(hist_x))])
                    df_vec = np.array([hist_f[idx] - f0 for idx in range(len(hist_f))])
                    norms = np.linalg.norm(dx_mat, axis=1)

                    # Filter out invalid points: those with very small norms or non-finite values
                    valid = (norms > 1e-15) & np.isfinite(df_vec) & np.all(np.isfinite(dx_mat), axis=1)
                    if np.sum(valid) < n:
                        v = limit_extrapolation 
                        break

                    dx_mat = dx_mat[valid]
                    df_vec = df_vec[valid]
                    norms = norms[valid]

                    # Compute weights based on the relative distances
                    d_rel = norms / np.min(norms)
                    weights = 1.0 / (d_rel**2)
                    
                    V_T = dx_mat * weights[:, np.newaxis]
                    delta_f = df_vec * weights

                    g0_simplex = np.linalg.lstsq(V_T, delta_f, rcond=1e-4)[0]
                    norm_g_simplex = np.linalg.norm(g0_simplex)

                    norm2g0_simplex = norm_g_simplex**2
                    xSurrogate = x0 - (1.0 / L) * g0_simplex
                    fSurrogate = f(xSurrogate)
                    nEval += 1

                    # Condição de aceitação baseada no sigma dinâmico
                    if (f0 - fSurrogate >= rho / L * norm2g0_simplex) and (f0 - fSurrogate > 2 / (25 * sigma) * tol**2):                        
                        hist_x.append(x0.copy())
                        hist_f.append(f0)
                        while len(hist_x) > memory:
                            hist_x.pop(0)
                            hist_f.pop(0)
                        
                        x0 = xSurrogate
                        f0 = fSurrogate
                        L /= 2
                        if nEval >= evalMax:
                            return x0, f0
                        break
                    else:
                        if nEval >= evalMax:
                            return x0, f0
                        if np.isfinite(fSurrogate) and (fSurrogate - f0 <= max(1e4 * abs(f0), 1e8)):
                            hist_x.append(xSurrogate.copy())
                            hist_f.append(fSurrogate)
                            while len(hist_x) > memory:
                                hist_x.pop(0)
                                hist_f.pop(0)
                        L *= 2
                        v += 1

                if v >= limit_extrapolation:
                    print(f"Extrapolation failed after {limit_extrapolation} attempts. Exiting extrapolation loop.")
                    break

            # Update sigma for the next iteration
            sigma = max(sigma/2, sigmaMin)
            k += 1


