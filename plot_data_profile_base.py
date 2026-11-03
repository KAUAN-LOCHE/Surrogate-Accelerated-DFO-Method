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
    {"name": "L2O-LSTM v6", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # {"name": "L2O-LSTM simplex1", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # {"name": "MLP-2C (+Sobolev)", "color": "blue",   "ls": "-"},
    # {"name": "L2O-LSTM (Warmup 10)", "model": "L2O", "sobolev": False, "color": "red", "ls": "-"}

    #Mlp com 2 camadas e metade dos neuronios
    # {"name": "MLP-2C metade neuronios (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},
    # {"name": "MLP-2C metade neuronios (Sem Sobolev)", "model": "NeuralNet2C", "sobolev": False,  "color": "blue",   "ls": "-"},
    # {"name": "MLP-2C 3n neuronios (+Sobolev)", "model": "NeuralNet2C", "sobolev": True,  "color": "blue",   "ls": "-"},



    # LSTM 
    #{"name": "L2O-LSTM (Warmup 15)", "model": "L2O", "sobolev": False, "color": "red", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 7
    # {"name": "L2O-LSTM-BlackBox", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12
    # {"name": "L2O-LSTM-BlackBox 12", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12 sem pre-treino
    # {"name": "L2O-LSTM-BlackBox 12 sem pre-treino", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
     # Modelo LSTM BlackBox dimensão máxima de treino igual a 12 pre-treinado 25dim
    # {"name": "L2O-LSTM-BlackBox 12 pre-treinado 25dim", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 12 pre-treinado 10dim
    # {"name": "L2O-LSTM-BlackBox 12 pre-treinado 10dim", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}
    # LSTM BlackBox dimensão máxima de treino igual a 7 pre-treinado 10dim cutest 
    # {"name": "L2O-LSTM-BlackBox 7 pre-treinado 10dim cutest", "model": "L2O_BlackBox", "color": "purple", "ls": "-"},
    # LSTM BlackBox dimensão máxima de treino igual a 7 pre-treinado 10dim cutest 3 iterações
    # {"name": "L2O-LSTM-BlackBox 7 pre-treinado 10dim cutest 3 iteracoes", "model": "L2O_BlackBox", "color": "purple", "ls": "-"}



   
    # MLP 4 camadas
    # Modelo MLP com 5n neurônios em cada uma das suas 4 camadas (BASE)
    # {"name": "MLP-4C (+Sobolev)", "color": "purple", "ls": "-"},
# Modelo MLP com 1n neurônios em cada uma das suas 4 camadas
    # {"name": "MLP-4C 1n (+Sobolev)", "color": "purple", "ls": "-"},
    
    # XNET
    # {"name": "XNet (+Sobolev)",   "color": "red",    "ls": "-"},
    #{"name":"XNet3 (+sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
    #{"name":"XNet3 Dimiminuida (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}

#   {"name":"XNet3 Diminuida 3 sem Sobolev" , "model": "XNet"  ,"sobolev":False , "color":"green" , "ls":"-"}
#   {"name":"XNet3 Diminuida 1 (+Sobolev)" , "model": "XNet"  ,"sobolev":True , "color":"green" , "ls":"-"}
    #{"name": "XNetReformuled (+Sobolev)", "color": "orange", "ls": "-"},
    # {"name": "XNetLinear (+Sobolev)", "color": "green", "ls": "-"}
]

history_dir = "historicos_dimensao_base"
results_dir = "results_dimensao_base"

tau = 1e-4
max_simplex_gradients = 100

# Adicione aqui os caminhos de TODOS os JSONs que você gerou
json_files = [
    # "historicos/problem_info_mlps_xnet_f.json",
    # "historicos/problem_info_xnet_3_diminuida.json",
    # "historicos/problem_info_xnet_3_diminuida_3_sem_sobolev.json",
    # "historicos/problem_info_xnet_3_diminuida_1_com_sobolev.json",
    f"{history_dir}/problem_info_mlps.json", 
    # "historicos/problem_info_xnet.json", 
    # "historicos/problem_info_xnetreformuled.json",
    # "historicos/problem_info_xnetlinear.json"
    
    f"{history_dir}/problem_info_mlp_1_problemas_base.json",
    #LSTM com warmup 10
    f"{history_dir}/problem_info_LSTM.json",
    #LSTM com warmup 15
    f"{history_dir}/problem_info_LSTM_warmup_15.json",
    #MLP-2C metade dos neuronios
    f"{history_dir}/problem_info_mlp_2_diminuida_metade_neuronios.json",
    #MLP-2C metade dos neuronios sem Sobolev
    f"{history_dir}/problem_info_mlp_2_diminuida_metade_neuronios_sem_sobolev.json",
    #MLP-2C com 3n neuronios com e sem Sobolev
    f"{history_dir}/problem_info_mlp_2_diminuida_3n_neuronios_com_sobolev.json",
    #MLP-4C com 1n neuronios
    f"{history_dir}/problem_info_mlp_4c_1n_com_sobolev.json",
    # LSTM BlackBox 7dim
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_7dim.json",
    # LSTM BlackBox 12dim
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_12dim.json",
    # LSTM BlackBox 12dim sem pre-treino
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_12dim_sem_pre_treino.json",
    # LSTM BlackBox 12dim pre-treinado 25dim
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_12dim_pre_treinado_25dim.json",
    # LSTM BlackBox 12dim pre-treinado 10dim
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_12dim_pre_treinado_10dim.json",
    # LSTM BlackBox 7dim pre-treinado 10dim cutest
    f"{history_dir}/problem_info_LSTM_L2O_BlackBox_7_pre_treinado_10dim_cutest_DFO.json",
    # LSTM BlackBox 7dim pre-treinado 10dim cutest 5 iterações
    #  f"{history_dir}/problem_info_LSTM_L2O_BlackBox_7_pre_treinado_10dim_cutest_5_iteracoes.json"
    f"{history_dir}/problem_info_LSTM_L2O_v6.json",
    # f"{history_dir}/problem_info_LSTM_L2O_simplex1.json"

]
# output_image = f"{results_dir}/data_profile_unificado_mlp_4c_1n_com_sobolev.png"
output_image = f"{results_dir}/data_profile_unificado_test_LSTM.png"

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
    profile = np.asarray(data_profiles[s_name], dtype=float)
    plt.step(
        budgets, profile,
        label=s_name,
        color=solver["color"],
        linestyle=solver["ls"],
        linewidth=2,
        where='post'
    )
     # Posição final da curva
    x_final = float(budgets[-1])
    y_final = float(profile[-1])

    # Desloca cada rótulo verticalmente para evitar sobreposição
    deslocamento = (solvers_to_plot.index(solver) % 5 - 5) * 30

    plt.annotate(
        f"{y_final:.3f}",
        xy=(x_final, y_final),
        xytext=(8, deslocamento),
        textcoords="offset points",
        color=solver["color"],
        fontsize=10,
        va="center",
        ha="left",
        annotation_clip=False
    )

plt.xlim(0, max_simplex_gradients)
plt.ylim(0, 1.05)
plt.xlabel("Number of simplex gradients (n+1 function evaluations)", fontsize=13)
plt.ylabel("Proportion of problems solved", fontsize=13)
plt.title(f"Data Profile: Surrogate Models Comparison ({problemas_comparados} CUTEst Problems)", fontsize=15)
plt.legend(loc='lower right', fontsize=11, ncol=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(output_image, dpi=300)
plt.close()
print(f"\nGráfico comparativo final salvo com sucesso como '{output_image}'.")
