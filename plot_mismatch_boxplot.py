import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuração dos Modelos para o Boxplot
# ==========================================
version = 18
solvers_to_plot = [
    {"name": "MLP-1C (+Sobolev) boxplot",   "label": "NN surrogate\n(with Sobolev learning)", "color": "#2ca02c"},
    # {"name": "Simplex gradient v8", "label": "Simplex gradient\nsurrogate (pior caso)",           "color": "#9467bd"},
    # {"name": f"Simplex gradient v{version}", "label": "Simplex gradient\nsurrogate",           "color": "#ff7f0e"},
    {"name": f"L2O-LSTM v6", "label": "LSTM surrogate", "color": "#1f77b4"}
]

history_dir = "historicos_dimensao_base"
results_dir = "results_dimensao_base"
os.makedirs(results_dir, exist_ok=True)

json_files = [
    f"{history_dir}/problem_info_mlps.json",
    f"{history_dir}/problem_info_mlp_1_problemas_base.json",
    f"{history_dir}/problem_info_Simplex_v4.json",
    f"{history_dir}/problem_info_Simplex_v{version}.json",
    f"{history_dir}/problem_info_mlp_1c_com_sobolev.json"
]

output_image = f"{results_dir}/boxplot_surrogate_mismatch.png"

def sanitize_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

# ==========================================
# 2. Carregamento dos Problemas e Mismatches
# ==========================================
master_info = {}
for j_file in json_files:
    if os.path.exists(j_file):
        with open(j_file, "r") as f:
            master_info.update(json.load(f))

mismatch_data = {solver["name"]: [] for solver in solvers_to_plot}
problemas_validos = 0

for prob_name, info in master_info.items():
    # Garante que compara exatamente os mesmos problemas entre MLP e Simplex
    todos_existem = True
    temp_vals = {}
    
    for solver in solvers_to_plot:
        s_name = solver["name"]
        safe_s_name = sanitize_name(s_name)
        mismatch_path = f"{history_dir}/{safe_s_name}_{prob_name}_mismatch.npy"
        
        if os.path.exists(mismatch_path):
            arr = np.load(mismatch_path)
            eta_val = float(arr[0])
            if np.isfinite(eta_val):
                temp_vals[s_name] = eta_val
            else:
                todos_existem = False
        else:
            todos_existem = False
            
    if todos_existem:
        problemas_validos += 1
        for solver in solvers_to_plot:
            s_name = solver["name"]
            mismatch_data[s_name].append(temp_vals[s_name])

if problemas_validos == 0:
    print("Nenhum arquivo *_mismatch.npy encontrado em comum para os modelos selecionados.")
    exit()

print(f"Plotando Boxplot de Surrogate Mismatch para {problemas_validos} problemas em comum...\n")
for solver in solvers_to_plot:
    s_name = solver["name"]
    vals = np.array(mismatch_data[s_name])
    print(f"Modelo: {s_name}")
    print(f"  -> Mediana de eta(S(T_max)): {np.median(vals):.4f}")
    print(f"  -> Média de eta(S(T_max)):   {np.mean(vals):.4f}")
    print(f"  -> Quartis (25%, 75%):       [{np.percentile(vals, 25):.4f}, {np.percentile(vals, 75):.4f}]\n")

# ==========================================
# 3. Plotagem Idêntica à Figura 2 do Artigo
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

datasets = [mismatch_data[s["name"]] for s in solvers_to_plot]
labels = [s["label"] for s in solvers_to_plot]
colors = [s["color"] for s in solvers_to_plot]

# whis=(0, 100) faz os bigodes irem do mínimo ao máximo (sem pontos outliers soltos), igual à Fig. 2 do artigo
bp = ax.boxplot(
    datasets,
    tick_labels=labels,
    patch_artist=True,
    widths=0.45,
    whis=(0, 100),
    showfliers=False,
    boxprops=dict(linewidth=1.8, edgecolor='black'),
    medianprops=dict(linewidth=2.0, color='black'),
    whiskerprops=dict(linewidth=1.8, color='black'),
    capprops=dict(linewidth=1.8, color='black')
)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylim(-0.02, 1.05)
ax.set_ylabel(r"$\eta(S(T_{\max}))$", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, axis='y', linestyle='-', alpha=0.6)

plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.close()
print(f"Gráfico Boxplot salvo com sucesso em '{output_image}'.")