import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuração dos Modelos para Comparação
# ==========================================
solvers_to_plot = [
    {"name": "MLP-1C (+Sobolev)", "color": "black",  "ls": "-"},
    # {"name": "MLP-2C (+Sobolev)", "color": "blue",   "ls": "-"},

    #Mlp com 2 camadas e metade dos neuronios
    {"name": "MLP-2C metade neuronios (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},

    # {"name": "MLP-4C (+Sobolev)", "color": "purple", "ls": "-"},
    # {"name": "XNet (+Sobolev)",   "color": "red",    "ls": "-"},
    #{"name":"XNet3 (+sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
    #{"name":"XNet3 Dimiminuida (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}

#   {"name":"XNet3 Diminuida 3 sem Sobolev" , "model": "XNet"  ,"sobolev":False , "color":"green" , "ls":"-"}
#   {"name":"XNet3 Diminuida 1 (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
    #{"name": "XNetReformuled (+Sobolev)", "color": "orange", "ls": "-"},
    # {"name": "XNetLinear (+Sobolev)", "color": "green", "ls": "-"}
]

tau = 1e-4
max_simplex_gradients = 100

history_dir = "historicos_dimensoes_maiores"
results_dir = "results_dimensoes_maiores"

# Adicione aqui os caminhos de TODOS os JSONs que você gerou
json_files = [
    # f"{history_dir}/problem_info_mlps_xnet_f.json",
    # f"{history_dir}/problem_info_xnet_3_diminuida.json",
    # f"{history_dir}/problem_info_xnet_3_diminuida_3_sem_sobolev.json",
    # f"{history_dir}/problem_info_xnet_3_diminuida_1_com_sobolev.json",
    # f"{history_dir}/problem_info_mlps.json", 
    #f"{history_dir}/problem_info_xnet.json", 
    #f"{history_dir}/problem_info_xnetreformuled.json",
    #f"{history_dir}/problem_info_xnetlinear.json",
    f"{history_dir}/problem_info_mlps_1_2_problemas_maiores.json",
    f"{history_dir}/problem_info_mlps_2_2_problemas_maiores.json",
]
output_image = "results_dimensoes_maiores/data_profile_unificado_mlp_2_diminuida_2_problemas_maiores.png"

# ==========================================
# 2. Funções Auxiliares
# ==========================================
def sanitize_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

# ==========================================
# 3. Mesclagem de Dados (Unificação dos JSONs)
# ==========================================
master_info = {}

for j_file in json_files:
    if os.path.exists(j_file):
        with open(j_file, "r") as f:
            data = json.load(f)
            master_info.update(data) # Funde os problemas no dicionário mestre
            print(f"Carregado: {j_file} ({len(data)} problemas)")
    else:
        print(f"Aviso: Arquivo {j_file} não encontrado. Será ignorado.")

total_problemas_mapeados = len(master_info)
if total_problemas_mapeados == 0:
    print("Nenhum problema encontrado nos arquivos JSON fornecidos.")
    exit()

print(f"\nTotal de problemas únicos mapeados para análise: {total_problemas_mapeados}")

# ==========================================
# 4. Cálculo do Perfil
# ==========================================
budgets = np.linspace(0, max_simplex_gradients, 200)
data_profiles = {solver["name"]: np.zeros(len(budgets)) for solver in solvers_to_plot}

problemas_comparados = 0

for prob_name, info in master_info.items():
    n = info['n']
    f0 = info['f0']
    
    # Passo A: Encontrar o f_best global para ESTE problema entre os modelos selecionados
    f_best = np.inf
    dados_encontrados = False
    
    for solver in solvers_to_plot:
        safe_s_name = sanitize_name(solver["name"])
        file_path = f"{history_dir}/{safe_s_name}_{prob_name}.npy"
        
        if os.path.exists(file_path):
            hist = np.load(file_path)
            if len(hist) > 0:
                f_best = min(f_best, np.min(hist))
                dados_encontrados = True
                
    if not dados_encontrados:
        continue # Pula se nenhum dos modelos escolhidos rodou este problema
        
    problemas_comparados += 1
    threshold = f0 - (1 - tau) * (f0 - f_best)
    
    # Passo B: Contabilizar a proporção baseada no threshold
    for solver in solvers_to_plot:
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

# ==========================================
# 5. Normalização e Plotagem
# ==========================================
if problemas_comparados == 0:
    print("Nenhum arquivo .npy correspondente aos modelos selecionados foi encontrado.")
    exit()

for solver in solvers_to_plot:
    data_profiles[solver["name"]] /= problemas_comparados

plt.figure(figsize=(12, 8))
for solver in solvers_to_plot:
    s_name = solver["name"]
    plt.step(budgets, data_profiles[s_name], label=s_name, 
             color=solver["color"], linestyle=solver["ls"], linewidth=2, where='post')

plt.xlim(0, max_simplex_gradients)
plt.ylim(0, 1.05)
plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
plt.ylabel("Proportion of problems solved", fontsize=13)
plt.title(f"Data Profile: Comparação de Modelos Substitutos ({problemas_comparados} Problemas CUTEst)", fontsize=15)
plt.legend(loc='lower right', fontsize=11, ncol=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(output_image, dpi=300)
plt.close()
print(f"\nGráfico comparativo final salvo com sucesso como '{output_image}'.")
