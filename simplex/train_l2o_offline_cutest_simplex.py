import os
import gc
import json
import torch
import torch.nn as nn
import numpy as np
import pycutest
import multiprocessing
from l2o_optimizer_simplex import L2O_BlackBox_LSTM
import copy


info_path = "../historicos_dimensao_base/problem_info_LSTM_L2O_BlackBox_7dim.json"
buffer_path = "cutest_trajectories_buffer_gradient_simplex.pt"
buffer_path_val = "cutest_trajectories_buffer_gradient_simplex_val.pt"
model_save_path = "l2o_quadratic_pretrained_gradient_50_simplex_15.pt"
EPOCHS_TRAIN = 50
M_history = 15


# Problemas com bug de decodificação SIF ou matrizes densas gigantescas
BLACKLIST = {'AKIVA', 'TRIDIA', 'YATP1CLS', 'YATP1LS', 'YATP2CLS', 'YATP2LS', 'JIMACK'}


def worker_collect_problem(prob_name, max_dim, n_starts, steps_per_traj, tol, return_list):
    """Executa a coleta de 1 problema em processo isolado para liberar 100% da RAM."""
    try:
        props = pycutest.problem_properties(prob_name)
        n_meta = props.get('n')
        if isinstance(n_meta, int) and (n_meta < 1 or n_meta > max_dim):
            return

        # Tenta importar com dimensão padrão; se for 'variable' e passar de max_dim, tenta N=50
        p = pycutest.import_problem(prob_name)
        if p.n > max_dim and n_meta == 'variable':
            try:
                p = pycutest.import_problem(prob_name, sifParams={'N': 50})
            except Exception:
                return

        n = p.n
        if n < 1 or n > max_dim:
            return

        print(f"  [Coletando] {prob_name} (n={n})...")
        np.random.seed(42)

        # custom_n_starts = n_starts if n > 20 else int(n_starts * 2)

        for s in range(n_starts):
            if s == 0:
                x0 = np.copy(p.x0)
            else:
                scale_x0 = np.maximum(np.abs(p.x0), 1.0)
                x0 = np.copy(p.x0) + np.random.randn(n) * 0.1 * scale_x0

            sigma = 1.0
            sigmaMin = 1e-2
            f0 = float(p.obj(x0))
            if not np.isfinite(f0):
                continue

            traj_x, traj_f, gs = [], [], []

            for _ in range(steps_per_traj):
                stepFound = False
                g0 = p.grad(x0)
                norm_g0 = np.linalg.norm(g0)

                if not np.isfinite(norm_g0) or norm_g0 > 1e12:
                    break

                for _ in range(100):

                    # h = 2 * tol / (5 * sigma * np.sqrt(n))
                    # fSample = np.zeros(n)
                    # for j in range(n):
                    #     fSample[j] = float(p.obj(x0 + h * np.eye(n)[:, j]))
                    # g0 = (fSample - f0) / h
                    # norm_g0 = np.linalg.norm(g0)

                    # if not np.isfinite(norm_g0) or norm_g0 > 1e12:
                    #     break
                    if norm_g0 >= 4 / 5 * tol:
                        xNew = x0 - (1.0 / sigma) * g0
                        fNew = float(p.obj(xNew))
                        if np.isfinite(fNew) and (f0 - fNew >= 1.0 / (8 * sigma) * (norm_g0 ** 2)):
                            dx = (xNew - x0).astype(np.float32)
                            df = np.float32(fNew - f0)
                            traj_x.append(x0)
                            traj_f.append(f0)
                            gs.append(g0)
                            x0, f0 = xNew, fNew
                            stepFound = True
                            break
                    # h /= 2.0
                    sigma *= 2.0

                if not stepFound:
                    break
                sigma = max(sigma / 2.0, sigmaMin)

            if len(traj_x) > 10:
                return_list.append((traj_x, traj_f, gs))

    except Exception:
        pass


def get_or_collect_trajectories(train_candidates, max_dim=110, buffer_path=buffer_path):
    # Se já coletou antes e salvou em disco, carrega direto sem rodar o PyCUTEst!
    if os.path.exists(buffer_path):
        print(f"\nBuffer encontrado em '{buffer_path}'! Carregando trajetórias do disco...")
        raw_buffer = torch.load(buffer_path, weights_only=False)
        trajectories = [
            ([torch.tensor(x, dtype=torch.float32) for x in t_x],
             [torch.tensor(f, dtype=torch.float32) for f in t_f],
             [torch.tensor(g, dtype=torch.float32) for g in t_g])
            for t_x, t_f, t_g in raw_buffer
        ]
        print(f"Total de trajetórias carregadas instantaneamente: {len(trajectories)}")
        return trajectories

    print("\nFase 1: Coletando trajetórias reais em processos isolados...")
    manager = multiprocessing.Manager()
    raw_buffer = []

    for prob_name in train_candidates:
        return_list = manager.list()
        p_worker = multiprocessing.Process(
            target=worker_collect_problem,
            args=(prob_name, max_dim, 3, 50, 1e-4, return_list)
        )
        p_worker.start()
        p_worker.join(timeout=45)  # Aborta se travar ou demorar mais de 45s

        if p_worker.is_alive():
            print(f"  [Timeout] {prob_name} demorou demais e foi pulado.")
            p_worker.terminate()
            p_worker.join()

        if len(return_list) > 0:
            raw_buffer.extend(list(return_list))
            print(f"  [OK] {prob_name} -> Total acumulado no buffer: {len(raw_buffer)}")
            # Salva incrementalmente em disco para nunca perder progresso
            torch.save(raw_buffer, buffer_path)

        gc.collect()

    trajectories = [
        ([torch.tensor(x, dtype=torch.float32) for x in t_x],
         [torch.tensor(f, dtype=torch.float32) for f in t_f],
         [torch.tensor(g, dtype=torch.float32) for g in t_g])
        for t_x, t_f, t_g in raw_buffer
    ]
    return trajectories


def train_l2o_on_cutest(n_epochs=EPOCHS_TRAIN, batch_windows=16, lr=1e-3, M_history=M_history):
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Arquivo '{info_path}' não encontrado.")

    with open(info_path, "r") as f:
        test_set_134 = set(json.load(f).keys())
    print(f"Total de problemas reservados para TESTE (excluídos do treino): {len(test_set_134)}")

    all_unconstrained = sorted(pycutest.find_problems(constraints='unconstrained', regular=True))
    train_candidates = [p for p in all_unconstrained if p not in test_set_134 and p not in BLACKLIST]
    np.random.seed(42)
    np.random.shuffle(train_candidates)


    #Separando 85% para treino e 15% para validação
    split_index = int(len(train_candidates) * 0.85)
    train_candidates, val_candidates = train_candidates[:split_index], train_candidates[split_index:]
    print(f"Total de problemas para TREINO: {len(train_candidates)}")
    print(f"Total de problemas para VALIDAÇÃO: {len(val_candidates)}")

    # Carrega instantaneamente as 177 trajetórias já salvas em cutest_trajectories_buffer.pt
    trajectories = get_or_collect_trajectories(train_candidates , max_dim=110)
    val_trajectories = get_or_collect_trajectories(val_candidates , max_dim=110, buffer_path=buffer_path_val)

    if len(trajectories) == 0:
        print("Nenhuma trajetória coletada.")
        return

    # Pré-extrai todas as janelas deslizantes de todas as 177 trajetórias
    all_windows = []
    val_windows = []
    for hist_x, hist_f, hist_g in trajectories:
        for t in range(M_history, len(hist_x)):
            x_ref = hist_x[t]
            f_ref = hist_f[t]
            seq_dx = torch.stack(hist_x[t - M_history : t])-x_ref
            seq_df = torch.stack(hist_f[t - M_history : t])-f_ref
            
            target_g = hist_g[t].unsqueeze(1)
            all_windows.append((seq_dx, seq_df, target_g))

    for hist_x, hist_f, hist_g in val_trajectories:
        for t in range(M_history, len(hist_x) ):
            x_ref = hist_x[t]
            f_ref = hist_f[t]

            seq_dx = torch.stack(hist_x[t - M_history : t]) - x_ref
            seq_df = torch.stack(hist_f[t - M_history : t]) - f_ref
            target_g = hist_g[t].unsqueeze(1)
            val_windows.append((seq_dx, seq_df, target_g))

    

    print(f"\nFase 2: Treinando em {len(all_windows)} janelas extraídas das {len(trajectories)} trajetórias...")
    l2o_model = L2O_BlackBox_LSTM(hidden_sz=20)
    meta_opt = torch.optim.Adam(l2o_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=n_epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_net = None

    for epoch in range(1, n_epochs + 1):
        l2o_model.train()
        # Embaralha todas as janelas para misturar problemas diferentes no mesmo lote
        indices = np.random.permutation(len(all_windows))
        
        epoch_loss = 0.0
        valid_windows = 0
        meta_opt.zero_grad()

        for count, idx in enumerate(indices, 1):
            seq_dx, seq_df, target_g = all_windows[idx]
            pred_g = l2o_model(seq_dx, seq_df)

            # Nova Escala Proximal do Gradiente (delta F / delta X)
            scale_df = torch.max(torch.abs(seq_df)) + 1e-8
            scale_dx = torch.max(torch.abs(seq_dx), dim=0, keepdim=True)[0].T + 1e-8
            scale_target_g = scale_df / scale_dx
            
            norm_pred = pred_g / scale_target_g
            norm_targ = torch.clamp(target_g / scale_target_g, min=-10.0, max=10.0)
            
            loss_huber = nn.functional.smooth_l1_loss(norm_pred, norm_targ)
            cos_sim = nn.functional.cosine_similarity(pred_g, target_g, dim=0).mean()
            loss = (loss_huber + 0.1 * (1.0 - cos_sim)) / batch_windows
            

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            epoch_loss += loss.item() * batch_windows
            valid_windows += 1

            # Atualiza os pesos a cada 'batch_windows' janelas misturadas
            if count % batch_windows == 0 or count == len(indices):
                torch.nn.utils.clip_grad_norm_(l2o_model.parameters(), max_norm=1.0)
                meta_opt.step()
                meta_opt.zero_grad()

        scheduler.step()
        l2o_model.eval()
        
        avg_global_loss = epoch_loss / max(valid_windows, 1)

        # ==========================================
        # INÍCIO DO LOOP DE VALIDAÇÃO
        # ==========================================
        val_loss_total = 0.0
        val_valid_windows = 0
        
        with torch.no_grad():
            for seq_dx, seq_df, target_g in val_windows:
                pred_g = l2o_model(seq_dx, seq_df)
                
                scale_df = torch.max(torch.abs(seq_df)) + 1e-8
                scale_dx = torch.max(torch.abs(seq_dx), dim=0, keepdim=True)[0].T + 1e-8
                scale_target_g = scale_df / scale_dx
                
                norm_pred = pred_g / scale_target_g
                norm_targ = torch.clamp(target_g / scale_target_g, min=-10.0, max=10.0)
                
                loss_huber = nn.functional.smooth_l1_loss(norm_pred, norm_targ)
                cos_sim = nn.functional.cosine_similarity(pred_g, target_g, dim=0).mean()
                loss = (loss_huber + 0.1 * (1.0 - cos_sim)) 
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss_total += loss.item()
                    val_valid_windows += 1
                    
        avg_val_loss = val_loss_total / max(val_valid_windows, 1)
        
        # ==========================================
        # SALVAMENTO BASEADO NA VALIDAÇÃO
        # ==========================================
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_net = copy.deepcopy(l2o_model.state_dict())
            torch.save(best_net, model_save_path)
            marcador = " * [Novo Melhor]"
        else:
            marcador = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Época {epoch:02d}/{n_epochs} | Treino: {avg_global_loss:.6f} | Validação: {avg_val_loss:.6f}{marcador}")




    torch.save(best_net, model_save_path)
    print(f"\nConcluído! Melhor perda global: {best_loss:.6f}. Pesos salvos em '{model_save_path}'.")


if __name__ == "__main__":
    train_l2o_on_cutest()