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


from methodWithSurrogate import finiteDiffGradientWithSurrogate

# ==========================================
# 1. Configurações de Diretório e Ferramentas
# ==========================================
# Cria a pasta para salvar os dados se ela não existir
history_dir = "historicos_dimensoes_maiores"
results_dir = "results_dimensoes_maiores"
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
  {"name": "MLP-1C (+Sobolev)", "model": "NeuralNet",   "sobolev": True,  "color": "black",  "ls": "-"},

  #mlp com 2 camadas e 5*n em cada camada
{"name": "MLP-2C (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},

# mlp com 2 camadas e metade dos neuronios
# {"name": "MLP-2C metade neuronios (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},

#mlp com 4 camadas e 5*n em cada camada
  #{"name": "MLP-4C (+Sobolev)", "model": "NeuralNet4C", "sobolev": True,  "color": "purple", "ls": "-"},
  #{"name":"XNet3 Dimiminuida (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
#{"name":"XNet3 Diminuida 2 (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
#{"name":"XNet3 Diminuida 1 (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
#{"name":"XNetReformuled (+sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
#{'name':"XNetLinear (+sobolev)" , "model": "XNetLinear"  ,"sobolev":True , "color":"green" , "ls":"-"}
]


tol = 1e-4
max_simplex_gradients = 100
tau = 1e-4
dimension = 200
problems_limit = 134

# ==========================================
# 3. FASE DE SIMULAÇÃO (Execução e Salvamento)
# ==========================================
print("Buscando problemas no CUTEst...")
all_unconstrained = pycutest.find_problems(constraints='unconstrained', regular=True)
test_problems = sorted(all_unconstrained)

problem_info = {}
problem_count = 0

info_path = f"{history_dir}/problem_info_mlps_2_2_problemas_maiores.json"

def evaluate_single_problem(prob_name, problem_counter, dimension, max_simplex_gradients, tol, solvers, history_dir, return_dict):
    """Executa um único problema em um processo isolado para evitar vazamento de memória."""
   
    
    try:
        props = pycutest.problem_properties(prob_name)
        n_meta = props.get('n')
        if isinstance(n_meta, int) and (n_meta < 50 or n_meta > dimension):
            return
            
        p = pycutest.import_problem(prob_name)
        n = p.n
        if n < 50 or n > dimension:
            return
            
        x0 = p.x0
        f = lambda x: p.obj(x)
        f0_initial = f(x0)
        evalMax = max_simplex_gradients * (n + 1)
        
        print(f"[{problem_counter}] Resolvendo: {prob_name} (n={n})")
        
        for solver in solvers:
            np.random.seed(0)
            wrapped_func = HistoryWrapper(f)
            print(f"  -> {solver['name']} (Sobolev: {solver['sobolev']})")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                try:
                    finiteDiffGradientWithSurrogate(
                        wrapped_func, np.copy(x0), evalMax, tol, solver["model"], solver["sobolev"]
                    )
                except RuntimeWarning:
                    print(f"    -> Divergiu (NaN).")
                except Exception:
                    pass 
            
            # Salva no disco
            hist_array = np.array(wrapped_func.history)
            safe_s_name = sanitize_name(solver["name"])
            np.save(f"{history_dir}/{safe_s_name}_{prob_name}.npy", hist_array)
            
        # Salva o resultado no dicionário compartilhado para atualizar o JSON na thread principal
        return_dict[prob_name] = {'n': n, 'f0': f0_initial}
        
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
            
            with open(info_path, "w") as f:
                json.dump(problem_info, f)
            problem_count += 1
            
        gc.collect()

    # ==========================================
    # 4. FASE DE PLOTAGEM (Leitura do Disco)
    # ==========================================
  
    print("Lendo históricos do disco para gerar o Data Profile...")

    if not os.path.exists(info_path):
        print("Nenhum dado encontrado para plotar.")
        exit()

    with open(info_path, "r") as f:
        loaded_info = json.load(f)

    valid_problems = len(loaded_info)
    if valid_problems == 0:
        exit()

    budgets = np.linspace(0, max_simplex_gradients, 200)
    data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers}

    for prob_name, info in loaded_info.items():
        n = info['n']
        f0 = info['f0']
        
        f_best = np.inf
        for solver in solvers:
            safe_s_name = sanitize_name(solver["name"])
            file_path = f"{history_dir}/{safe_s_name}_{prob_name}.npy"
            if os.path.exists(file_path):
                hist = np.load(file_path)
                if len(hist) > 0:
                    f_best = min(f_best, np.min(hist))
                    
        threshold = f0 - (1 - tau) * (f0 - f_best)
        
        for solver in solvers:
            s_name = solver["name"]
            safe_s_name = sanitize_name(s_name)
            file_path = f"{history_dir}/{safe_s_name}_{prob_name}.npy"
            
            if os.path.exists(file_path):
                hist = np.load(file_path)
                for b_idx, b in enumerate(budgets):
                    allowed_evals = int(b * (n + 1))
                    visible_hist = hist[:allowed_evals]
                    
                    if len(visible_hist) > 0 and np.min(visible_hist) <= threshold:
                        data_profiles[s_name][b_idx] += 1

    # Normalização e Plotagem
    for solver in solvers:
        data_profiles[solver["name"]] /= valid_problems

    plt.figure(figsize=(12, 8))
    for solver in solvers:
        s_name = solver["name"]
        plt.step(budgets, data_profiles[s_name], label=s_name, 
                 color=solver["color"], linestyle=solver["ls"], linewidth=2, where='post')

    plt.xlim(0, max_simplex_gradients)
    plt.ylim(0, 1.05)
    plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
    plt.ylabel("Proportion of problems solved", fontsize=13)
    plt.title(f"Data Profile: Comparação de Modelos ({valid_problems} Problemas)", fontsize=15)
    plt.legend(loc='lower right', fontsize=11, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    final_file = f"{results_dir}/data_profile_{valid_problems}problems_mlps_2_2_problemas_maiores.png"
    plt.savefig(final_file, dpi=300)
    plt.close()
    print(f"Gráfico final salvo como '{final_file}'.")