#!/usr/bin/python3
# --------------------------------------------------------- #
#  Offline Training Script for L2O LSTM Optimizer           #
# --------------------------------------------------------- #

import os
import gc
import json
import torch
import torch.nn as nn
import numpy as np
import pycutest
import multiprocessing
from l2o_optimizer import L2O_BlackBox_LSTM
import copy

info_path = "historicos_dimensao_base/problem_info_LSTM_L2O_BlackBox_7dim.json"
buffer_path = "PATH_TRAJECTORIES_BUFFER_TRAIN.pt"
buffer_path_val = "PATH_TRAJECTORIES_BUFFER_VALIDATION.pt"
model_save_path = "PATH_TO_SAVED_MODEL.pt"
EPOCHS_TRAIN = 100
M_history = 7
STEPS_PER_TRAJ = 50
TRAJECTORIES_PER_PROBLEM = 3
NUM_LAYERS = 2

# List of problems that are known to cause issues or are not suitable for training
BLACKLIST = {'AKIVA', 'TRIDIA', 'YATP1CLS', 'YATP1LS', 'YATP2CLS', 'YATP2LS', 'JIMACK'}

def worker_collect_problem(prob_name, max_dim, n_starts, steps_per_traj, tol, return_list):
    """Execution of a single problem in a separate process to collect trajectories."""
    try:
        props = pycutest.problem_properties(prob_name)
        n_meta = props.get('n')
        if isinstance(n_meta, int) and (n_meta < 1 or n_meta > max_dim):
            return

        p = pycutest.import_problem(prob_name)
        if p.n > max_dim and n_meta == 'variable':
            try:
                p = pycutest.import_problem(prob_name, sifParams={'N': 50})
            except Exception:
                return

        n = p.n
        if n < 1 or n > max_dim:
            return

        print(f"  [Collecting] {prob_name} (n={n})...")
        np.random.seed(42)

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

            traj_dx, traj_df = [], []

            for _ in range(steps_per_traj):
                stepFound = False   

                for _ in range(100):

                    h = 2 * tol / (5 * sigma * np.sqrt(n))
                    fSample = np.zeros(n)
                    for j in range(n):
                        fSample[j] = float(p.obj(x0 + h * np.eye(n)[:, j]))
                    g0 = (fSample - f0) / h
                    norm_g0 = np.linalg.norm(g0)

                    if not np.isfinite(norm_g0) or norm_g0 > 1e12:
                        break
                    if norm_g0 >= 4 / 5 * tol:
                        xNew = x0 - (1.0 / sigma) * g0
                        fNew = float(p.obj(xNew))
                        if np.isfinite(fNew) and (f0 - fNew >= 1.0 / (2*sigma) * (norm_g0 ** 2)):
                            dx = (xNew - x0).astype(np.float32)
                            df = np.float32(fNew - f0)
                            traj_dx.append(dx)
                            traj_df.append(df)
                            x0, f0 = xNew, fNew
                            stepFound = True
                            break
                    h /= 2.0
                    sigma *= 2.0

                if not stepFound:
                    break
                sigma = max(sigma / 2.0, sigmaMin)

            if len(traj_dx) > 10:
                return_list.append((traj_dx, traj_df))

    except Exception:
        pass

def get_or_collect_trajectories(train_candidates, max_dim=110, buffer_path=buffer_path):
    # If the buffer file exists, load trajectories from disk
    if os.path.exists(buffer_path):
        print(f"\nBuffer found '{buffer_path}'! Loading...")
        raw_buffer = torch.load(buffer_path, weights_only=False)
        trajectories = [
            ([torch.tensor(dx, dtype=torch.float32) for dx in t_dx],
             [torch.tensor(df, dtype=torch.float32) for df in t_df])
            for t_dx, t_df in raw_buffer
        ]
        print(f"Total trajectories: {len(trajectories)}")
        return trajectories

    manager = multiprocessing.Manager()
    raw_buffer = []

    for prob_name in train_candidates:
        return_list = manager.list()
        p_worker = multiprocessing.Process(
            target=worker_collect_problem,
            args=(prob_name, max_dim, TRAJECTORIES_PER_PROBLEM, STEPS_PER_TRAJ, 1e-4, return_list)
        )
        p_worker.start()
        p_worker.join(timeout=45)  

        if p_worker.is_alive():
            print(f"  [Timeout] {prob_name} took too long. Terminating process.")
            p_worker.terminate()
            p_worker.join()

        if len(return_list) > 0:
            raw_buffer.extend(list(return_list))
            print(f"  [OK] {prob_name} -> total acumulated: {len(raw_buffer)}")
            #Saves the buffer after each problem to avoid losing data in case of interruption
            torch.save(raw_buffer, buffer_path)

        gc.collect()

    trajectories = [
        ([torch.tensor(dx, dtype=torch.float32) for dx in t_dx],
         [torch.tensor(df, dtype=torch.float32) for df in t_df])
        for t_dx, t_df in raw_buffer
    ]
    return trajectories


def train_l2o_on_cutest(n_epochs=EPOCHS_TRAIN, batch_windows=16, lr=1e-3, M_history=M_history):
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"File '{info_path}' not found.")

    with open(info_path, "r") as f:
        test_set_134 = set(json.load(f).keys())
    print(f"Total problems reserved for TEST (excluded from training): {len(test_set_134)}")

    all_unconstrained = sorted(pycutest.find_problems(constraints='unconstrained', regular=True))
    train_candidates = [p for p in all_unconstrained if p not in test_set_134 and p not in BLACKLIST]
    np.random.seed(42)
    np.random.shuffle(train_candidates)

    # Split the candidates into training and validation sets (85% train, 15% validation)
    split_index = int(len(train_candidates) * 0.85)
    train_candidates, val_candidates = train_candidates[:split_index], train_candidates[split_index:]
    print(f"Total problems for TRAINING: {len(train_candidates)}")
    print(f"Total problems for VALIDATION: {len(val_candidates)}")

    trajectories = get_or_collect_trajectories(train_candidates , max_dim=110)
    val_trajectories = get_or_collect_trajectories(val_candidates , max_dim=110, buffer_path=buffer_path_val)

    if len(trajectories) == 0:
        print("No trajectories collected.")
        return

    all_windows = []
    val_windows = []
    for hist_dx, hist_df in trajectories:
        for t in range(M_history, len(hist_dx) - 1):
            seq_dx = torch.stack(hist_dx[t - M_history : t])
            seq_df = torch.stack(hist_df[t - M_history : t])
            target_dx = hist_dx[t].unsqueeze(1)
            all_windows.append((seq_dx, seq_df, target_dx))

    for hist_dx, hist_df in val_trajectories:
        for t in range(M_history, len(hist_dx) - 1):
            seq_dx = torch.stack(hist_dx[t - M_history : t])
            seq_df = torch.stack(hist_df[t - M_history : t])
            target_dx = hist_dx[t].unsqueeze(1)
            val_windows.append((seq_dx, seq_df, target_dx))

    print(f"\nFase 2: training L2O model with {len(all_windows)} windows from {len(trajectories)} trajectories.")
    l2o_model = L2O_BlackBox_LSTM(hidden_sz=20, num_layers=NUM_LAYERS)
    meta_opt = torch.optim.Adam(l2o_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=n_epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_net = None

    for epoch in range(1, n_epochs + 1):
        l2o_model.train()
        # Shuffle the windows for each epoch to ensure diverse training batches
        indices = np.random.permutation(len(all_windows))
        
        epoch_loss = 0.0
        valid_windows = 0
        meta_opt.zero_grad()

        for count, idx in enumerate(indices, 1):
            seq_dx, seq_df, target_dx = all_windows[idx]
            pred_dx = l2o_model(seq_dx, seq_df)

            scale_target = torch.mean(torch.abs(seq_dx[-2:]), dim=0, keepdim=True).T + 1e-8
            norm_pred = pred_dx / scale_target
            norm_targ = torch.clamp(target_dx / scale_target, min=-10.0, max=10.0)
            loss_huber = nn.functional.smooth_l1_loss(norm_pred, norm_targ)
            
            cos_sim = nn.functional.cosine_similarity(pred_dx, target_dx, dim=0).mean()
            loss = (loss_huber + 0.1 * (1.0 - cos_sim)) / batch_windows

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            epoch_loss += loss.item() * batch_windows
            valid_windows += 1

            if count % batch_windows == 0 or count == len(indices):
                torch.nn.utils.clip_grad_norm_(l2o_model.parameters(), max_norm=1.0)
                meta_opt.step()
                meta_opt.zero_grad()

        scheduler.step()
        l2o_model.eval()
        
        avg_global_loss = epoch_loss / max(valid_windows, 1)

        # ==========================================
        # Start of validation phase
        # ==========================================
        val_loss_total = 0.0
        val_valid_windows = 0
        
        with torch.no_grad():
            for seq_dx, seq_df, target_dx in val_windows:
                pred_dx = l2o_model(seq_dx, seq_df)
                
                scale_target = torch.mean(torch.abs(seq_dx[-2:]), dim=0, keepdim=True).T + 1e-8
                
                norm_pred = pred_dx / scale_target
                norm_targ = torch.clamp(target_dx / scale_target, min=-10.0, max=10.0)
                
                loss_huber = nn.functional.smooth_l1_loss(norm_pred, norm_targ)
                cos_sim = nn.functional.cosine_similarity(pred_dx, target_dx, dim=0).mean()
                
                loss = loss_huber + 0.1 * (1.0 - cos_sim)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss_total += loss.item()
                    val_valid_windows += 1
                    
        avg_val_loss = val_loss_total / max(val_valid_windows, 1)
        
        # ==========================================
        # Saving the best model based on validation loss
        # ==========================================
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_net = copy.deepcopy(l2o_model.state_dict())
            torch.save(best_net, model_save_path)
            marker = " * [New Best Model Saved]"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epochs {epoch:02d}/{n_epochs} | Train: {avg_global_loss:.6f} | Validation: {avg_val_loss:.6f}{marker}")

    torch.save(best_net, model_save_path)
    print(f"\nConcluded training. Best model saved to '{model_save_path}' with validation loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_l2o_on_cutest()