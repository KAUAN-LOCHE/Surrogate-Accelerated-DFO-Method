#!/usr/bin/python3
# --------------------------------------------------------- #
#  Surrogate-accelerated finite-difference gradient method  #
# --------------------------------------------------------- #

""" Imports """
import numpy as np
import torch # <- NOVO IMPORT
from l2o_optimizer import L2O_BlackBox_LSTM # <- NOVO IMPORT (Certifique-se de ter o l2o_optimizer.py na pasta)



""" Parameters """
sigmaMin = 1e-2
iterLineSearchMax = 100

simplexMemory = 10  # Maximum size of the training set in simplex gradients
extrapolMax = 500  # Maximum number of extrapolation by iteration
rho = 1e-4  # Armijo linesearch parameter in extrapolation steps

reg = 1e-4  # Regularization parameter for neural network training
ratioHiddenInputSize = 5  # Ratio between input and hidden width of the neural network
rtolTraining = 1e-6  # Tolerance for training the neural network
maxIterTraining = 1000  # Maximum number of iterations for training the neural network

def compute_mismatch(surrogate_steps, n):
    if len(surrogate_steps) == 0:
        return 1.0, 0.0

    S_T = float(np.mean(surrogate_steps))
    eta = (1.0 + S_T/(2.0*(n+1))) / (1.0 + S_T)
    return float(eta), S_T


""" Main code """
def finiteDiffGradientWithSurrogate(f, x0, evalMax, tol, model):
    n = len(x0)
    I = np.eye(n)
    sigma = 1

    f0 = f(x0)
    f0_copy = np.copy(f0)
    nEval = 1
    # =========================================================================
    # LÓGICA 1: HORIZONTE L2O (HÍBRIDO pré-treinado + treinamento online)
    # =========================================================================
    if model == "L2O_BlackBox":
        l2o_net = L2O_BlackBox_LSTM(hidden_sz=20, num_layers=2)

        try:
            l2o_net.load_state_dict(torch.load("l2o_quadratic_pretrained_gradient_100_original_DF.pt"))
            # l2o_net.load_state_dict(torch.load("modelos_gradiente/l2o_quadratic_pretrained_gradient_100_original.pt"))
            print("Modelo L2O pré-treinado carregado com sucesso.")
        except FileNotFoundError:
            print("Modelo L2O pré-treinado não encontrado. Treinando do zero.")
            # Aqui você pode adicionar código para treinar o modelo do zero, se desejar.

        l2o_opt = torch.optim.Adam(l2o_net.parameters(), lr=1e-4, weight_decay=1e-5)
        M_history = 7
        iterations = 1 # Número de iterações de treinamento do L2O por iteração do método
        hist_dx = []
        hist_df = []
        counter = 0
        reincident_flaw = 0 
        surrogate_steps = [] 
        k = 0
        fSample = np.zeros(n)
        while True:
            h = 2*tol/(5*sigma*np.sqrt(n))
            stepFound = False
            for l in range(iterLineSearchMax):
                for j in range(n):
                    fSample[j] = f(x0 + h*I[:,j])
                nEval += n
                g0 = (fSample- f0)/h

                if nEval >= evalMax:
                    print(f"norm(g0) = {np.linalg.norm(g0)}")
                    print(f"Maximum decrease: {f0_copy - f0}")
                    eta, S_T = compute_mismatch(hist_dx, n)
                    return x0,f0, k, eta, S_T
                


                if np.linalg.norm(g0) >= 4/5 * tol:
                    xNew = x0 - 1/sigma * g0
                    fNew = f(xNew)
                    nEval +=1

                    if nEval >= evalMax:
                        print(f"norm(g0) = {np.linalg.norm(g0)}")
                        print(f"Maximum decrease: {f0_copy - f0}")
                        eta, S_T = compute_mismatch(hist_dx, n)
                        return x0, f0, k, eta, S_T

                    if f0 - fNew >= 1/(8*sigma) * np.linalg.norm(g0)**2:
                        dx = xNew - x0
                        df = fNew - f0

                        x0 = xNew
                        f0 = fNew
                        stepFound = True
                        break
                h/=2
                sigma *= 2

            if not stepFound:
                reincident_flaw += 1
                print(f"Step not found in line search. Tentativa {reincident_flaw}.")
                
                if reincident_flaw >= 3: # Limite de falhas toleradas
                    print("Failsafe ativado: Múltiplas falhas na busca de linha. Encerrando.")
                    eta, S_T = compute_mismatch(surrogate_steps, n)
                    return x0, f0, k, eta, S_T
                
                continue
            else:
                reincident_flaw = 0

            if len(hist_dx) < M_history:
                hist_dx.append(dx)
                hist_df.append(df)
                k += 1
                sigma = max(sigma/2, sigmaMin)
                continue

            
            seq_dx = torch.tensor(np.array(hist_dx), dtype=torch.float32)      
            seq_df = torch.tensor(np.array(hist_df), dtype=torch.float32)
            target = torch.tensor(dx, dtype=torch.float32).unsqueeze(1)
                
                    
            print(f"Training L2O model at iteration {k} with {len(hist_dx)} samples.")
            scale_target = torch.mean(torch.abs(seq_dx[-2:]), dim=0, keepdim=True).T + 1e-8
            for _ in range(iterations):  # Treina por 7 iterações
                l2o_net.train()
                l2o_opt.zero_grad()
                step_pred = l2o_net(seq_dx, seq_df)

                # Como deve ficar:
                norm_pred = step_pred / scale_target
                norm_targ = torch.clamp(target / scale_target, min=-10.0, max=10.0)
                loss_huber = torch.nn.functional.smooth_l1_loss(norm_pred, norm_targ)

                cos_sim = torch.nn.functional.cosine_similarity(step_pred, target, dim=0).mean()
                loss = loss_huber + 0.1 * (1.0 - cos_sim)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), max_norm=1.0)
                l2o_opt.step()  

            hist_dx.append(dx)
            hist_df.append(df)

            if len(hist_dx) > M_history:
                hist_dx.pop(0)
                hist_df.pop(0)

            ext_dx = list(hist_dx)
            ext_df = list(hist_df)
            last_valid_dx = None
            last_valid_df = None
            f0_before = f0
            counter = 0
            # damping = 1.0
            # tried_damping = False
            for j in range(extrapolMax):
                l2o_net.eval()
                with torch.no_grad():
                    seq_dx = torch.tensor(np.array(ext_dx), dtype=torch.float32)
                    seq_df = torch.tensor(np.array(ext_df), dtype=torch.float32)
                    step_pred = l2o_net(seq_dx, seq_df).reshape(-1).numpy()
                xSurrogate = x0 + step_pred
                fSurrogate = f(xSurrogate)
                nEval += 1


                if f0 - fSurrogate > 2/(25*sigma) * tol**2:
                    dx_lstm = xSurrogate - x0
                    df_lstm = fSurrogate - f0
                    counter += 1
                    # damping = 1.0
                    # tried_damping = False

                    ext_dx.append(dx_lstm)
                    ext_df.append(df_lstm)
                    if len(ext_dx) > M_history:
                        ext_dx.pop(0)
                        ext_df.pop(0)

                    x0 = xSurrogate
                    f0 = fSurrogate
                    if nEval >= evalMax:
                        print(f"norm(g0) = {np.linalg.norm(g0)}")
                        print(f"Maximum decrease: {f0_copy - f0}")
                        surrogate_steps.append(counter)
                        eta, S_T = compute_mismatch(surrogate_steps, n)
                        return x0, f0, k+1, eta, S_T
                    last_valid_dx = dx_lstm
                    last_valid_df = df_lstm
                    
                else:
                    if nEval >= evalMax:
                        print(f"norm(g0) = {np.linalg.norm(g0)}")
                        print(f"Maximum decrease: {f0_copy - f0}")
                        surrogate_steps.append(counter)
                        eta, S_T = compute_mismatch(surrogate_steps, n)
                        return x0, f0, k+1, eta, S_T
                    break
                   
            if counter > 0:
                hist_dx.append(last_valid_dx)
                hist_df.append(last_valid_df)
                if len(hist_dx) > M_history:
                    hist_dx.pop(0)
                    hist_df.pop(0)
                # hist_dx = ext_dx.copy()
                # hist_df = ext_df.copy()


            total_fall = f0_before - f0
            print(f"Counter: {counter}, Total_fall: {total_fall}")

            surrogate_steps.append(counter)
            sigma = max(sigma/2, sigmaMin)
            k += 1


