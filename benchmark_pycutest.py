import numpy as np
import pycutest
import time
import warnings
import matplotlib
matplotlib.use('Agg') # Evita erro de interface gráfica no cluster/WSL
import matplotlib.pyplot as plt

from methodWithSurrogate import finiteDiffGradientWithSurrogate

# ==========================================
# 1. O "Espião" de Avaliações (Wrapper)
# ==========================================
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

# ==========================================
# 2. Configuração dos Otimizadores
# ==========================================
solvers = [
    {"name": "MLP-1C (+Sobolev)", "model": "NeuralNet",   "sobolev": True,  "color": "black",  "ls": "-"},
    
    {"name": "MLP-2C (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},
    # {"name": "MLP-2C (Std)",      "model": "NeuralNet2C", "sobolev": False, "color": "blue",   "ls": "--"},
    
    {"name": "MLP-4C (+Sobolev)", "model": "NeuralNet4C", "sobolev": True,  "color": "purple", "ls": "-"},
    # {"name": "MLP-4C (Std)",      "model": "NeuralNet4C", "sobolev": False, "color": "purple", "ls": "--"},
    
    #{"name": "XNet (+Sobolev)",   "model": "XNet",        "sobolev": True,  "color": "red",    "ls": "-"},
    # {"name": "XNet (Std)",        "model": "XNet",        "sobolev": False, "color": "red",    "ls": "--"},
    
    #{"name": "XNetLin (+Sobolev)","model": "XNetLinear",  "sobolev": True,  "color": "orange", "ls": "-"},
    # {"name": "XNetLin (Std)",     "model": "XNetLinear",  "sobolev": False, "color": "orange", "ls": "--"},

    #{"name": "XNetMLP (+Sobolev)","model": "XNetMLP",  "sobolev": True,  "color": "green", "ls": "-"}
    # {"name": "XNetMLP (Std)",     "model": "XNetMLP",  "sobolev": False, "color": "green", "ls": "--"}
]

tol = 1e-4
max_simplex_gradients = 100
tau = 1e-4

# ==========================================
# 3. Seleção de Problemas PyCUTEst
# ==========================================
print("Buscando problemas no CUTEst...")
all_unconstrained = pycutest.find_problems(constraints='unconstrained', regular=True)
test_problems = sorted(all_unconstrained)

# count = 0
# for prob_name in test_problems:
#     if count >= 60:
#         break
#     props = pycutest.problem_properties(prob_name)
#     n_meta = props.get('n')
#     if isinstance(n_meta, int) and (n_meta < 1 or n_meta > 110):
#         continue
#     print(prob_name, n_meta)
#     count += 1



dimension = 110
problems = 134
histories = {}
problem_info = {}
problem_count = 0
time_start = time.time()
print(f"Iniciando Benchmark PyCUTEst para {len(test_problems)} problemas potenciais...\n")
for prob_name in test_problems:
    if problem_count >= problems: 
        break

    if prob_name == 'AKIVA':
        continue

    try:
        # 1. FILTRO ULTRARRÁPIDO: Checa a dimensão lendo apenas o texto (sem compilar Fortran)
        props = pycutest.problem_properties(prob_name)
        n_meta = props.get('n')
        
        # Se for um tamanho fixo e fora do limite, pula em 1 milissegundo
        if isinstance(n_meta, int) and (n_meta < 1 or n_meta > dimension):
            continue

        # 2. Só compila o Fortran se o problema tiver o tamanho certo
        p = pycutest.import_problem(prob_name)
        n = p.n
        
        # Checagem final (garantia)
        if n < 1 or n > dimension:
            continue
            
        x0 = p.x0
        f = lambda x: p.obj(x)
        
        f0_initial = f(x0)
        evalMax = max_simplex_gradients * (n + 1)
        
        print(f"[{problem_count+1}] Resolvendo: {prob_name} (n={n})")
        
        histories[problem_count] = {}
        problem_info[problem_count] = {'n': n, 'f0': f0_initial, 'name': prob_name}
        
        for solver in solvers:
            np.random.seed(0)
            wrapped_func = HistoryWrapper(f)
            print(f"  -> {solver['name']} (Sobolev: {solver['sobolev']})")
            
            # --- BLINDAGEM EXTERNA (MANTÉM O CÓDIGO DOS AUTORES INTACTO) ---
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                
                try:
                    finiteDiffGradientWithSurrogate(
                        wrapped_func, np.copy(x0), evalMax, tol, solver["model"], solver["sobolev"]
                    )
                except RuntimeWarning:
                    print(f"    -> {solver['name']}: Divergiu matematicamente (NaN). Interrompido.")
                except Exception as e:
                    pass 
                
            histories[problem_count][solver["name"]] = wrapped_func.history
            
        problem_count += 1

        # Geração eventual de gráfico
        if problem_count % 15 == 0:
            # ==========================================
            # 4.Plotagem do Data Profile
            # ==========================================
            budgets = np.linspace(0, max_simplex_gradients, 200)
            data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers}

            for p_id in range(problem_count):
                n = problem_info[p_id]['n']
                f0 = problem_info[p_id]['f0']
                
                f_best = np.inf
                for s_name in histories[p_id]:
                    if len(histories[p_id][s_name]) > 0:
                        best_s = min(histories[p_id][s_name])
                        if best_s < f_best:
                            f_best = best_s
                            
                threshold = f0 - (1 - tau) * (f0 - f_best)
                
                for solver in solvers:
                    s_name = solver["name"]
                    hist = histories[p_id].get(s_name, [])
                    
                    for b_idx, b in enumerate(budgets):
                        allowed_evals = int(b * (n + 1))
                        visible_hist = hist[:allowed_evals]
                        
                        if len(visible_hist) > 0 and min(visible_hist) <= threshold:
                            data_profiles[s_name][b_idx] += 1

            for solver in solvers:
                data_profiles[solver["name"]] /= problem_count

            plt.figure(figsize=(12, 8))

            for solver in solvers:
                s_name = solver["name"]
                plt.step(budgets, data_profiles[s_name], label=s_name, 
                        color=solver["color"], linestyle=solver["ls"], linewidth=2, where='post')

            plt.xlim(0, max_simplex_gradients)
            plt.ylim(0, 1.05)
            plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
            plt.ylabel("Proportion of problems solved", fontsize=13)
            plt.title(f"Data Profile: Comparação de Modelos Substitutos ({problem_count} Problemas CUTEst)", fontsize=15)

            plt.legend(loc='lower right', fontsize=11, ncol=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            file_name = f"data_profile_pycutest_dimension{dimension}_{problem_count}problems_mlps.png"
            plt.savefig(file_name, dpi=300)
            plt.close()
            print(f"Gráfico salvo com sucesso como '{file_name}'.")
        
    except Exception as e:
        print(f"Falha ao carregar o problema {prob_name}. Pulando...")
        continue

time_end = time.time()

print(f"\nBenchmark concluído com {problem_count} problemas válidos. Gerando Data Profile...")

# ==========================================
# 4. Cálculo e Plotagem do Data Profile
# ==========================================
budgets = np.linspace(0, max_simplex_gradients, 200)
data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers}

for p_id in range(problem_count):
    n = problem_info[p_id]['n']
    f0 = problem_info[p_id]['f0']
    
    f_best = np.inf
    for s_name in histories[p_id]:
        if len(histories[p_id][s_name]) > 0:
            best_s = min(histories[p_id][s_name])
            if best_s < f_best:
                f_best = best_s
                
    threshold = f0 - (1 - tau) * (f0 - f_best)
    
    for solver in solvers:
        s_name = solver["name"]
        hist = histories[p_id].get(s_name, [])
        
        for b_idx, b in enumerate(budgets):
            allowed_evals = int(b * (n + 1))
            visible_hist = hist[:allowed_evals]
            
            if len(visible_hist) > 0 and min(visible_hist) <= threshold:
                data_profiles[s_name][b_idx] += 1

for solver in solvers:
    data_profiles[solver["name"]] /= problem_count

plt.figure(figsize=(12, 8))

for solver in solvers:
    s_name = solver["name"]
    plt.step(budgets, data_profiles[s_name], label=s_name, 
             color=solver["color"], linestyle=solver["ls"], linewidth=2, where='post')

plt.xlim(0, max_simplex_gradients)
plt.ylim(0, 1.05)
plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
plt.ylabel("Proportion of problems solved", fontsize=13)
plt.title(f"Data Profile: Comparação de Modelos Substitutos ({problem_count} Problemas CUTEst)", fontsize=15)

plt.legend(loc='lower right', fontsize=11, ncol=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

file_name = f"data_profile_pycutest_{problem_count}_mlps.png"
plt.savefig(file_name, dpi=300)
print(f"Gráfico salvo com sucesso como '{file_name}'.")
plt.close()