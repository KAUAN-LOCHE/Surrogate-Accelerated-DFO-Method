import numpy as np
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
# 2. Funções de Teste DFO (Puramente em NumPy)
# ==========================================
def rosenbrock(x): 
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def sphere(x): 
    return np.sum(x**2)

def rastrigin(x): 
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def sum_powers(x): 
    return np.sum(np.abs(x)**(np.arange(1, len(x)+1) + 1))

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e

base_functions = [
    ("Rosenbrock", rosenbrock),
    ("Sphere", sphere),
    ("Rastrigin", rastrigin),
    ("SumPowers", sum_powers),
    ("Ackley", ackley)
]

# Variando as dimensões para criar uma bateria massiva de problemas
dimensions = [2, 4, 6, 8, 10, 15, 20]

# ==========================================
# 3. Configuração dos Otimizadores
# ==========================================
solvers = [
    {"name": "NeuralNet", "model": "NeuralNet", "sobolev": True},
    {"name": "NeuralNet2C", "model": "NeuralNet2C", "sobolev": True},
    {"name": "NeuralNet2C", "model": "NeuralNet2C", "sobolev": False},
    {"name": "NeuralNet4C", "model": "NeuralNet4C", "sobolev": False}, # Rodando sem Sobolev devido à explosão do Hessiano
    {"name": "XNetLinear", "model": "XNetLinear", "sobolev": True},
    {"name": "XNetLinear", "model": "XNetLinear", "sobolev": False}
]

tol = 1e-4
max_simplex_gradients = 100

histories = {}
problem_info = {}
problem_id = 0

print("Iniciando Benchmark 100% Python...\n")

for func_name, func in base_functions:
    for n in dimensions:
        print(f"Resolvendo: {func_name} (n={n})")
        
        # Ponto inicial inicializado longe do mínimo global
        x0 = np.ones(n) * 2.0
        f0_initial = func(x0)
        
        # Orçamento máximo: 100 gradientes simplex (100 * (n+1) avaliações)
        evalMax = max_simplex_gradients * (n + 1)
        
        histories[problem_id] = {}
        problem_info[problem_id] = {'n': n, 'f0': f0_initial}
        
        for solver in solvers:
            np.random.seed(0)
            wrapped_func = HistoryWrapper(func)
            
            try:
                finiteDiffGradientWithSurrogate(
                    wrapped_func, np.copy(x0), evalMax, tol, solver["model"], solver["sobolev"]
                )
            except Exception as e:
                # Se um modelo quebrar em uma função específica, ignora e segue a vida
                pass 
                
            histories[problem_id][solver["name"]] = wrapped_func.history
            
        problem_id += 1

print("\nBenchmark concluído. Gerando Data Profile...")

# ==========================================
# 4. Cálculo e Plotagem do Data Profile
# ==========================================
tau = 1e-4
budgets = np.linspace(0, max_simplex_gradients, 200)
data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers}

for p_id in range(problem_id):
    n = problem_info[p_id]['n']
    f0 = problem_info[p_id]['f0']
    
    # Encontra o mínimo global descoberto entre todos os modelos para este problema
    f_best = np.inf
    for s_name in histories[p_id]:
        if len(histories[p_id][s_name]) > 0:
            best_s = min(histories[p_id][s_name])
            if best_s < f_best:
                f_best = best_s
                
    # Critério de convergência
    threshold = f0 - (1 - tau) * (f0 - f_best)
    
    for solver in solvers:
        s_name = solver["name"]
        hist = histories[p_id].get(s_name, [])
        
        for b_idx, b in enumerate(budgets):
            allowed_evals = int(b * (n + 1))
            visible_hist = hist[:allowed_evals]
            
            # Se a melhor função vista até aquele momento bateu a meta, pontua
            if len(visible_hist) > 0 and min(visible_hist) <= threshold:
                data_profiles[s_name][b_idx] += 1

# Converte de contagem absoluta para proporção (0 a 1)
for solver in solvers:
    data_profiles[solver["name"]] /= problem_id

plt.figure(figsize=(10, 6))
colors = ['orange', 'green', 'red', 'blue', 'purple', 'brown']
linestyles = ['-', '--', '-.', ':', '-', '--']

for idx, solver in enumerate(solvers):
    s_name = solver["name"]
    plt.step(budgets, data_profiles[s_name], label=s_name, 
             color=colors[idx], linestyle=linestyles[idx], linewidth=2, where='post')

plt.xlim(0, max_simplex_gradients)
plt.ylim(0, 1.05)
plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=12)
plt.ylabel("Proportion of problems solved", fontsize=12)
plt.title("Data Profile: Comparação de Arquiteturas DFO-Surrogate", fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("data_profile_dfo.png", dpi=300)
print("Gráfico salvo como 'data_profile_dfo.png'.")