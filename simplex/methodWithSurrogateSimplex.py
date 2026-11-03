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

def compute_mismatch(surrogate_steps, n):
    if len(surrogate_steps) == 0:
        return 1.0, 0.0

    S_T = float(np.mean(surrogate_steps))
    eta = (1.0 + M*S_T/(2.0*(n+1))) / (1.0 + S_T)
    return float(eta), S_T

""" Main code """
def finiteDiffGradientWithSurrogate(f, x0, evalMax, tol, model):
    n = len(x0)
    I = np.eye(n)
    sigma = 1
    memory = 4*(n+1)
    f0 = f(x0)
    f0_copy = np.copy(f0)
    nEval = 1
    # =========================================================================
    # LÓGICA 1: HORIZONTE L2O (HÍBRIDO pré-treinado + treinamento online)
    # =========================================================================
    if model == "Simplex":
        

        hist_x = []
        hist_f = []

        sigmas = set()
        sigmas.add(sigma)
        k = 0
        fSample = np.zeros(n)
        x_perturbed = np.zeros((n,n))
        surrogate_steps = []
        step_extrapol = None

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
                        print(f"norm(g0) = {np.linalg.norm(g0)}")
                        print(f"Maximum decrease: {f0_copy - f0}")
                        eta, S_T = compute_mismatch(surrogate_steps, n)
                        return x0,f0, k, eta, S_T

                while len(hist_x) > memory:
                    hist_x.pop(0)
                    hist_f.pop(0)

                g0 = (fSample- f0)/h
               
                

                if np.linalg.norm(g0) >= 4/5 * tol:
                    xNew = x0 - 1/sigma * g0
                    fNew = f(xNew)
                    nEval +=1

                    if nEval >= evalMax:
                        print(f"norm(g0) = {np.linalg.norm(g0)}")
                        print(f"Maximum decrease: {f0_copy - f0}")
                        eta, S_T = compute_mismatch(surrogate_steps, n)
                        return x0, f0, k, eta, S_T

                    if f0 - fNew >= 1/(8*sigma) * np.linalg.norm(g0)**2:
                        print("Norm of g0:", np.linalg.norm(g0))
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
                sigmas.add(sigma)

            if not stepFound:
                reincident_flaw += 1
                print(f"Step not found in line search. Tentativa {reincident_flaw}.")
                
                if reincident_flaw >= 2: # Limite de falhas toleradas
                    print("Failsafe ativado: Múltiplas falhas na busca de linha. Encerrando.")
                    eta, S_T = compute_mismatch(surrogate_steps, n)
                    return x0, f0, k, eta, S_T
                sigma = max(sigmaBefore/2, sigmaMin)
                sigmas.add(sigma)
                continue
            else:
                reincident_flaw = 0

            print(f"Iniciating extrapolation with simplex model at iteration {k}.")
            t_k = 0
            f0_before = f0
            L = sigma 
            # if step_extrapol is None else step_extrapol
            # step_extrapol = None
            limit_extrapolation = M
            v = 0
            for j in range(extrapolMax):
                while v < limit_extrapolation:
                    # 1. Calcula dx e df para todos os pontos do histórico
                    dx_mat = np.array([hist_x[idx] - x0 for idx in range(len(hist_x))])
                    df_vec = np.array([hist_f[idx] - f0 for idx in range(len(hist_f))])
                    norms = np.linalg.norm(dx_mat, axis=1)

                    # 2. Filtra pontos idênticos a x0 (norma ~ 0) ou com f infinito/NaN
                    valid = (norms > 1e-15) & np.isfinite(df_vec) & np.all(np.isfinite(dx_mat), axis=1)
                    if np.sum(valid) < n:
                        v = limit_extrapolation # Garante que vai sair do for j externo!
                        break

                    dx_mat = dx_mat[valid]
                    df_vec = df_vec[valid]
                    norms = norms[valid]

                    # 3. Normaliza a distância pela menor distância atual
                    d_rel = norms / np.min(norms)
                    weights = 1.0 / (d_rel**2)
                    
                    V_T = dx_mat * weights[:, np.newaxis]
                    delta_f = df_vec * weights

                    # 4. rcond=1e-3 evita inverter direções quase colineares vistas de x0_novo
                    g0_simplex = np.linalg.lstsq(V_T, delta_f, rcond=1e-4)[0]
                    norm_g_simplex = np.linalg.norm(g0_simplex)

                    norm2g0_simplex = norm_g_simplex**2
                    xSurrogate = x0 - (1.0 / L) * g0_simplex
                    fSurrogate = f(xSurrogate)
                    nEval += 1

                    print("Norm of g0_simplex:", norm_g_simplex)


                    # Condição de aceitação baseada no sigma dinâmico
                    if (f0 - fSurrogate >= rho / L * norm2g0_simplex) and (f0 - fSurrogate > 2 / (25 * sigma) * tol**2):                        
                        t_k += 1
                        hist_x.append(x0.copy())
                        hist_f.append(f0)
                        while len(hist_x) > memory:
                            hist_x.pop(0)
                            hist_f.pop(0)
                        
                        x0 = xSurrogate
                        f0 = fSurrogate
                        L /= 2
                        # step_extrapol = L
                        if nEval >= evalMax:
                            print(f"norm(g0) = {np.linalg.norm(g0)}")
                            print(f"Maximum decrease: {f0_copy - f0}")
                            surrogate_steps.append(t_k)
                            eta, S_T = compute_mismatch(surrogate_steps, n)
                            return x0, f0, k, eta, S_T
                        break
                    else:
                        if nEval >= evalMax:
                            print(f"norm(g0) = {np.linalg.norm(g0)}")
                            print(f"Maximum decrease: {f0_copy - f0}")
                            surrogate_steps.append(t_k)
                            eta, S_T = compute_mismatch(surrogate_steps, n)
                            return x0, f0, k, eta, S_T
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

            total_fall = f0_before - f0
            print(f"t_k: {t_k}, Total_fall: {total_fall}")

            # Atualiza o sigma para a próxima iteração macro
            surrogate_steps.append(t_k)
            sigma = max(sigma/2, sigmaMin)
            sigmas.add(sigma)
            k += 1


