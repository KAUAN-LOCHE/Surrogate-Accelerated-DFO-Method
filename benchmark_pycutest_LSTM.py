import numpy as np
import pycutest
import time
import warnings
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import gc
# from methodWithSurrogate import finiteDiffGradientWithSurrogate
from methodWithSurrogateLSTM import finiteDiffGradientWithSurrogate

# ==========================================
# 1. Configurações de Diretório e Ferramentas
# ==========================================
# Cria a pasta para salvar os dados se ela não existir
history_dir = "historicos_dimensao_base"
results_dir = "results_dimensao_base"
os.makedirs(history_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

class HistoryWrapper:
    def __init__(self, func):
        self.func = func
        self.history = []
        self.best_f = np.inf

    def __call__(self, x):
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
        self.history.append(self.best_f)
        return val

# Função auxiliar para limpar nomes de arquivos
def sanitize_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

# ==========================================
# 2. Configuração dos Otimizadores
# ==========================================
solvers = [

    # LSTM BlackBox dimensão máxima de treino igual a 7
    # {"name": "L2O-LSTM-BlackBox", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12
    # {"name": "L2O-LSTM-BlackBox 12", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12 sem pre-treino
    # {"name": "L2O-LSTM-BlackBox 12 sem pre-treino", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # Modelo LSTM BlackBox dimensão máxima de treino igual a 12 pre-treinado 25dim
    # {"name": "L2O-LSTM-BlackBox 7 pre-treinado 10dim cutest", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12 pre-treinado 10dim 3 iterações
    {"name": "L2O-LSTM v6", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}

]


tol = 1e-4
max_simplex_gradients = 100
tau = 1e-4
dimension = 110
problems_limit = 134

# ==========================================
# 3. FASE DE SIMULAÇÃO (Execução e Salvamento)
# ==========================================
print("Buscando problemas no CUTEst...")
all_unconstrained = pycutest.find_problems(constraints='unconstrained', regular=True)
test_problems = sorted(all_unconstrained)

problem_info = {}
problem_count = 0

# info_path = f"{history_dir}/problem_info_LSTM_L2O_BlackBox_7_pre_treinado_10dim_cutest.json"
# info_path = f"{history_dir}/problem_info_LSTM_L2O_BlackBox_7_pre_treinado_10dim_cutest_3_iteracoes.json"
info_path = f"{history_dir}/problem_info_LSTM_L2O_v6.json"

def evaluate_single_problem(prob_name, problem_counter, dimension, max_simplex_gradients, tol, solvers, history_dir, return_dict):
    """Executa um único problema em um processo isolado para evitar vazamento de memória."""
   
    
    try:
        props = pycutest.problem_properties(prob_name)
        n_meta = props.get('n')
        if isinstance(n_meta, int) and (n_meta < 1 or n_meta > dimension):
            return
            
        p = pycutest.import_problem(prob_name)
        n = p.n
        if n < 1 or n > dimension:
            return
            
        x0 = p.x0
        f = lambda x: p.obj(x)
        f0_initial = f(x0)
        evalMax = max_simplex_gradients * (n + 1)
        eta = 1.0
        S = 0.0
        print(f"[{problem_counter}] Resolvendo: {prob_name} (n={n})")
        for solver in solvers:
            np.random.seed(0)
            wrapped_func = HistoryWrapper(f)
            print(f"  -> {solver['name']}")
            k = 0
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                try:
                    _, _, k, eta, S =finiteDiffGradientWithSurrogate(
                        wrapped_func, np.copy(x0), evalMax, tol, solver["model"]
                    )
                except RuntimeWarning:
                    print(f"    -> Divergiu (NaN).")
                except Exception:
                    pass 
            

            # Salva no disco
            hist_array = np.array(wrapped_func.history)
            safe_s_name = sanitize_name(solver["name"])
            np.save(f"{history_dir}/{safe_s_name}_{prob_name}.npy", hist_array)

            np.save(f"{history_dir}/{safe_s_name}_{prob_name}_mismatch.npy", np.array([eta, S]))
            
        # Salva o resultado no dicionário compartilhado para atualizar o JSON na thread principal
        return_dict[prob_name] = {'n': n, 'f0': f0_initial, 'k': k, 'eta': eta, 'S': S}
        
    except Exception as e:
        print(f"Falha no problema {prob_name}: {e}")

# ==========================================
# 3. FASE DE SIMULAÇÃO (Execução e Salvamento com Multiprocessing)
# ==========================================
if __name__ == '__main__':
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict() 
    
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            problem_info = json.load(f)
            problem_count = len(problem_info)
    else:
        problem_info = {}
        problem_count = 0

    print(f"Iniciando/Continuando Benchmark para até {problems_limit} problemas...\n")
    means_dim = {}
    for prob_name in test_problems:
        if problem_count >= problems_limit: 
            break
        if prob_name == 'AKIVA' or prob_name in problem_info:
            continue
        
        p_worker = multiprocessing.Process(
            target=evaluate_single_problem,
            args=(prob_name, problem_count, dimension, max_simplex_gradients, tol, solvers, history_dir, return_dict)
        )
        
        p_worker.start()
        p_worker.join() 


        if prob_name in return_dict:
            problem_info[prob_name] = return_dict[prob_name]
            means_dim.setdefault(return_dict[prob_name]['n'], []).append(return_dict[prob_name]['k'])
            
            with open(info_path, "w") as f:
                json.dump(problem_info, f)
            problem_count += 1
            
        gc.collect()


    print("Resultados:")
# oordenar items
    means_dim = dict(sorted(means_dim.items()))

    for n, iterations in means_dim.items():
        if iterations:
            quantity = len(iterations)
            mean_iterations = np.mean(iterations)
            print(f"Dimensão {n}, Quantidade = {quantity}, Média de iterações = {mean_iterations:.2f}")
        else:
            print(f"Dimensão {n}: Nenhuma execução válida.")

    # # ==========================================
    # # 4. FASE DE PLOTAGEM (Leitura do Disco)
    # # ==========================================
  
    # print("Lendo históricos do disco para gerar o Data Profile...")

    # if not os.path.exists(info_path):
    #     print("Nenhum dado encontrado para plotar.")
    #     exit()

    # with open(info_path, "r") as f:
    #     loaded_info = json.load(f)

    # valid_problems = len(loaded_info)
    # if valid_problems == 0:
    #     exit()

    # budgets = np.linspace(0, max_simplex_gradients, 200)
    # data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers}

    # for prob_name, info in loaded_info.items():
    #     n = info['n']
    #     f0 = info['f0']
        
    #     f_best = np.inf
    #     for solver in solvers:
    #         safe_s_name = sanitize_name(solver["name"])
    #         file_path = f"{history_dir}/{safe_s_name}_{prob_name}.npy"
    #         if os.path.exists(file_path):
    #             hist = np.load(file_path)
    #             if len(hist) > 0:
    #                 f_best = min(f_best, np.min(hist))
                    
    #     threshold = f0 - (1 - tau) * (f0 - f_best)
        
    #     for solver in solvers:
    #         s_name = solver["name"]
    #         safe_s_name = sanitize_name(s_name)
    #         file_path = f"{history_dir}/{safe_s_name}_{prob_name}.npy"
            
    #         if os.path.exists(file_path):
    #             hist = np.load(file_path)
    #             for b_idx, b in enumerate(budgets):
    #                 allowed_evals = int(b * (n + 1))
    #                 visible_hist = hist[:allowed_evals]
                    
    #                 if len(visible_hist) > 0 and np.min(visible_hist) <= threshold:
    #                     data_profiles[s_name][b_idx] += 1

    # # Normalização e Plotagem
    # for solver in solvers:
    #     data_profiles[solver["name"]] /= valid_problems

    # plt.figure(figsize=(12, 8))
    # for solver in solvers:
    #     s_name = solver["name"]
    #     plt.step(budgets, data_profiles[s_name], label=s_name, 
    #              color=solver["color"], linestyle=solver["ls"], linewidth=2, where='post')

    # plt.xlim(0, max_simplex_gradients)
    # plt.ylim(0, 1.05)
    # plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
    # plt.ylabel("Proportion of problems solved", fontsize=13)
    # plt.title(f"Data Profile: Comparação de Modelos ({valid_problems} Problemas)", fontsize=15)
    # plt.legend(loc='lower right', fontsize=11, ncol=2)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()

    # final_file = f"{results_dir}/data_profile_{valid_problems}problems_LSTM_L2O_BlackBox_7dim.png"
    # plt.savefig(final_file, dpi=300)
    # plt.close()
    # print(f"Gráfico final salvo como '{final_file}'.")