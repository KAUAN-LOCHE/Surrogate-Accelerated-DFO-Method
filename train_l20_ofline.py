import torch
import torch.nn as nn
import numpy as np
from l2o_optimizer import L2O_BlackBox_LSTM

def train_l2o_offline(n_epochs=1000, dim=10, lr=0.001, M_history=7):
    print("Iniciando o Meta-Treinamento Offline com Escalas Variáveis...")
    l2o_model = L2O_BlackBox_LSTM(hidden_sz=20)
    meta_opt = torch.optim.Adam(l2o_model.parameters(), lr=lr)
    
    for epoch in range(1, n_epochs + 1):
        # Aleatoriza completamente a escala (de 0.01 até 100)
        scale = 10 ** (torch.rand(1).item() * 4 - 2)
        W = torch.randn(dim, dim) * scale
        y = torch.randn(dim) * scale
        
        def f(x):
            return torch.sum((torch.matmul(W, x) - y) ** 2)
        
        theta = torch.randn(dim) * scale
        L_est = torch.linalg.norm(W, ord=2).item() ** 2 + 1e-6
        lr_gd = (torch.rand(1).item() * 0.4 + 0.1) / L_est
        
        hist_dx, hist_df = [], []
        f_val = f(theta)
        divergiu = False
        
        for _ in range(60):
            theta.requires_grad_(True)
            f_val = f(theta)
            f_val.backward()
            
            with torch.no_grad():
                grad = theta.grad
                if torch.isnan(grad).any() or torch.norm(grad) > 1e6:
                    divergiu = True
                    break
                    
                dx = -lr_gd * grad 
                theta_new = theta + dx
                f_new = f(theta_new)
                
                hist_dx.append(dx.clone())
                hist_df.append((f_new - f_val).clone())
                
                theta = theta_new.detach()
                f_val = f_new.detach()
                
        if divergiu or len(hist_dx) <= M_history:
            continue
            
        l2o_model.train()
        epoch_loss = 0.0
        
        for t in range(M_history, len(hist_dx) - 1):
            seq_dx = torch.stack(hist_dx[t - M_history : t]) 
            seq_df = torch.stack(hist_df[t - M_history : t]) 
            target_dx = hist_dx[t].unsqueeze(1) 
            
            meta_opt.zero_grad()
            pred_dx = l2o_model(seq_dx, seq_df)
            scale_target = torch.mean(torch.abs(seq_dx[-2:]), dim=0, keepdim=True).T + 1e-8
            loss_mse = nn.functional.mse_loss(pred_dx / scale_target, target_dx / scale_target)
            cos_sim = nn.functional.cosine_similarity(pred_dx, target_dx, dim=0)
            loss = loss_mse + 0.1 * (1.0 - cos_sim)
            loss.backward()
            
            # Clipping do Gradiente para estabilizar o treino
            torch.nn.utils.clip_grad_norm_(l2o_model.parameters(), max_norm=1.0)
            meta_opt.step()
            epoch_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"Época {epoch}/{n_epochs} | Erro Médio (MSE): {epoch_loss / (len(hist_dx) - 1 - M_history):.8f}")

    torch.save(l2o_model.state_dict(), "l2o_quadratic_pretrained.pt")
    print("\nConcluído!")

if __name__ == "__main__":
    train_l2o_offline()