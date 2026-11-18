# Surrogate-Accelerated Derivative-Free Optimization (DFO)

## Project Overview

This repository investigates alternative acceleration strategies for **derivative-free optimization (DFO)**, building upon the frameworks proposed in [1] and [2].

Many scientific and engineering problems involve black-box objective functions whose analytical derivatives are unavailable, unreliable, or too expensive to compute. DFO methods address this setting by using function values, rather than explicitly supplied derivatives, to guide the optimization process.

The project studies ways to improve the efficiency of a finite-difference-based quadratic regularization framework by incorporating learned or history-based information. The central goal is to reduce the number of expensive objective-function evaluations while retaining acceptance and progress checks based on the original objective function.

Three acceleration paradigms are implemented:

- **Neural surrogate models:** Shallow and deeper multilayer perceptrons (MLPs), radial basis function (RBF) models, and a multivariate rational Cauchy-based model (XNet). Sobolev training can incorporate approximate gradient information into the surrogate fitting process.
- **Learning-to-Optimize (L2O):** A coordinate-wise Long Short-Term Memory (LSTM) network that predicts optimization displacement steps from recent optimization history. The model is trained offline using trajectories generated from optimization problems and is subsequently used within the optimization procedure.
- **History-based least-squares gradient:** A non-neural approach that estimates local gradient information from previously evaluated points using a distance-weighted least-squares procedure, inspired by simplex-gradient ideas.

The implementations are intended for experimentation and comparison. Their performance may depend on the objective function, dimension, available history, model configuration, and computational cost of evaluating the objective.

### Main files

| File | Description |
|---|---|
| `classModel.py` | Base class for surrogate models, including management of training data and loss minimization using L-BFGS. |
| `classNeuralNet.py` | Baseline shallow neural-network surrogate. |
| `classNeuralNet2C.py` | MLP architecture with two hidden layers and second-order Sobolev-related training support. |
| `classNeuralNet4C.py` | MLP architecture with four hidden layers and second-order Sobolev-related training support. |
| `classRadialBasis.py` | Radial basis function surrogate model. |
| `classXNet.py` | Multivariate rational Cauchy-based surrogate model (XNet). |
| `methodWithSurrogate.py` | DFO solver using neural and RBF surrogate models. |
| `l2o_optimizer.py` | PyTorch definition of the coordinate-wise LSTM meta-optimizer. |
| `train_l2o_offline_cutest.py` | Offline L2O training script using trajectories generated from CUTEst problems. |
| `methodWithSurrogateLSTM.py` | DFO solver accelerated by the L2O-LSTM model. |
| `methodWithSurrogateSimplex.py` | DFO solver using a history-based, distance-weighted least-squares gradient approximation. |

The repository tree above describes the principal files supplied for the project. Experimental outputs, local datasets, caches, and additional scripts may be present depending on the checkout.

## Requirements and Dependencies

The main Python dependencies and minimum versions are:

```text
numpy>=1.20.0
scipy>=1.6.0
torch>=1.10.0
matplotlib>=3.3.0
pycutest>=1.8.2
```

### Python libraries

| Library | Role |
|---|---|
| **NumPy** | Numerical arrays, vector operations, and linear algebra. |
| **SciPy** | Numerical routines and optimization-related utilities. |
| **PyTorch** | Neural-network models, automatic differentiation, training, and execution of the L2O model. |
| **Matplotlib** | Visualization of optimization trajectories, convergence, and experimental results. |
| **PyCUTEst** | Python interface used to access CUTEst benchmark problems. |

### External requirements for PyCUTEst

PyCUTEst requires more than installing a Python package. Its use depends on an external CUTEst toolchain, which typically includes:

- **CUTEst:** The underlying collection and infrastructure for optimization test problems.
- **SIFDecode:** Decodes problem descriptions in SIF format and supports the generation of problem-specific code.
- **MASTSIF:** Collection of SIF problem files.
- **Fortran and C compilers:** For example, `gfortran` and `gcc`, needed to compile the relevant components.

The environment variables `SIFDECODE` and `MASTSIF` must point to the corresponding local installation directories. The exact paths and setup commands depend on how CUTEst and SIFDecode were installed.

**Important:** A successful `pip install pycutest` does not confirm that the external CUTEst environment is correctly installed. Missing compilers, incorrect environment variables, or unavailable SIF files can prevent problems from being decoded, compiled, or loaded.

PyCUTEst is primarily needed for experiments that use CUTEst benchmark problems. Custom objective functions implemented directly in Python can be used without PyCUTEst, provided the selected solver does not otherwise depend on it.

## Installation and Configuration

The following is a general setup outline for Linux or Windows Subsystem for Linux (WSL). The CUTEst/SIFDecode installation is a separate system-level step and must be completed according to the official documentation.

### 1. Clone the repository

```bash
git clone https://github.com/KAUAN-LOCHE/Surrogate-Accelerated-DFO-Method.git
cd Surrogate-Accelerated-DFO-Method
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows Command Prompt, activation is typically:

```bat
venv\Scripts\activate
```

### 3. Install Python dependencies

Upgrade `pip` and install the packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyCUTEst is not needed for a particular experiment, it may be omitted from the installation and the remaining Python dependencies installed separately.

### 4. Configure CUTEst and PyCUTEst

Install and configure CUTEst, SIFDecode, MASTSIF, and the required compilers by following the official project documentation. Then define the environment variables required by PyCUTEst. For a Bash-based environment, these are commonly configured in `~/.bashrc`, for example:

```bash
export SIFDECODE=/path/to/sifdecode
export MASTSIF=/path/to/mastsif
```

Replace the example paths with the actual locations on the machine. Reload the shell configuration or open a new terminal after editing it.

#### Operating-system note

A Linux environment or WSL with Ubuntu is recommended for CUTEst-based experiments. Native Windows setups may require additional configuration and can encounter compatibility or compilation issues because the CUTEst toolchain relies on Unix-oriented build processes and compiled components. The exact behavior depends on the installed versions and environment.

Useful documentation:

- [PyCUTEst documentation](https://jfowkes.github.io/pycutest/)
- [CUTEst repository](https://github.com/ralna/CUTEst)
- [SIFDecode repository](https://github.com/ralna/SIFDecode)

## Running the Project

The solvers can be used with custom Python objective functions or, where configured, with benchmark problems accessed through PyCUTEst.

The main solver routines are located in:

- `1_Neural_Surrogates/methodWithSurrogate.py`
- `2_L2O_LSTM/methodWithSurrogateLSTM.py`
- `3_Least_Squares_Gradient/methodWithSurrogateSimplex.py`

The exact function signatures and optional parameters should be checked in the corresponding source files, since they may differ between solver variants.

### CUTEst benchmark experiments

For CUTEst experiments, the benchmark problem must be available through the configured PyCUTEst installation. The training script `train_l2o_offline_cutest.py` uses CUTEst problems to collect optimization trajectories for offline L2O training.

The benchmark list documented below records problems name, dimension (`n`), and initial objective value (`f0`) for each entry. These values describe the supplied experiment metadata. 

## CUTEst Problems in the Supplied Experiment

**134 problem entries**. The recorded dimensions range from **1 to 100**. The table reports problems informations:

| Problem | Dimension (`n`) | Initial objective (`f0`) |
|---|---:|---:|
| `ALLINITU` | 4 | 13 |
| `BA-L1LS` | 57 | 127388 |
| `BA-L1SPLS` | 57 | 127388 |
| `BARD` | 3 | 41.6817 |
| `BEALE` | 2 | 14.2031 |
| `BENNETT5LS` | 3 | 66022.4 |
| `BIGGS6` | 6 | 0.77907 |
| `BOX3` | 3 | 1.88457 |
| `BOXBODLS` | 2 | 186382 |
| `BRKMCC` | 2 | 5.99 |
| `BROWNBS` | 2 | 9.99998e+11 |
| `BROWNDEN` | 4 | 7.92669e+06 |
| `CERI651ALS` | 7 | 10321.8 |
| `CERI651BLS` | 7 | 448.979 |
| `CERI651CLS` | 7 | 119.885 |
| `CERI651DLS` | 7 | 30.7278 |
| `CERI651ELS` | 7 | 37.1343 |
| `CHNROSNB` | 50 | 7635.84 |
| `CHNRSNBM` | 50 | 8633.49 |
| `CHWIRUT1LS` | 3 | 50068.6 |
| `CHWIRUT2LS` | 3 | 14794.8 |
| `CLIFF` | 2 | 4.85165e+08 |
| `CLUSTERLS` | 2 | 1 |
| `COOLHANSLS` | 9 | 902930 |
| `CUBE` | 2 | 749.038 |
| `DANIWOODLS` | 2 | 149.719 |
| `DANWOODLS` | 2 | 149.719 |
| `DENSCHNA` | 2 | 7.95249 |
| `DENSCHNB` | 2 | 6 |
| `DENSCHNC` | 2 | 889.303 |
| `DENSCHND` | 3 | 8.321e+07 |
| `DENSCHNE` | 3 | 148.999 |
| `DENSCHNF` | 2 | 416 |
| `DEVGLA1` | 4 | 105127 |
| `DEVGLA2` | 5 | 10940.1 |
| `DIAMON2DLS` | 66 | 3.55755e+06 |
| `DIAMON3DLS` | 99 | 4.02894e+06 |
| `DIXMAANA1` | 3 | 29.5 |
| `DJTL` | 2 | -2641.36 |
| `DMN15102LS` | 66 | 3.55755e+06 |
| `DMN15103LS` | 99 | 4.02894e+06 |
| `DMN15332LS` | 66 | 491558 |
| `DMN15333LS` | 99 | 484133 |
| `DMN37142LS` | 66 | 263050 |
| `DMN37143LS` | 99 | 150236 |
| `ECKERLE4LS` | 3 | 0.722303 |
| `EGGCRATE` | 2 | 43.3724 |
| `ELATVIDU` | 2 | 16002 |
| `ENGVAL2` | 3 | 629 |
| `ENSOLS` | 9 | 1153.94 |
| `ERRINROS` | 50 | 110182 |
| `ERRINRSM` | 50 | 152873 |
| `EXP2` | 2 | 4.09974 |
| `EXPFIT` | 2 | 24.0625 |
| `FBRAIN3LS` | 6 | 10.2348 |
| `GAUSS1LS` | 8 | 7371.72 |
| `GAUSS2LS` | 8 | 9158.14 |
| `GAUSS3LS` | 8 | 18905.1 |
| `GAUSSIAN` | 3 | 3.88811e-06 |
| `GBRAINLS` | 2 | 41.0311 |
| `GROWTHLS` | 3 | 85962.4 |
| `GULF` | 3 | 12.1107 |
| `HAHN1LS` | 7 | 3.09756e+06 |
| `HAIRY` | 2 | 700.847 |
| `HATFLDD` | 3 | 25.243 |
| `HATFLDE` | 3 | 45.2065 |
| `HATFLDFL` | 3 | 0.000944198 |
| `HATFLDFLS` | 3 | 0.0779631 |
| `HATFLDGLS` | 25 | 27 |
| `HEART6LS` | 6 | 564.813 |
| `HEART8LS` | 8 | 181.62 |
| `HELIX` | 3 | 2500 |
| `HIELOW` | 3 | 1009.36 |
| `HILBERTA` | 2 | 10.5 |
| `HILBERTB` | 10 | 510.189 |
| `HIMMELBB` | 2 | 26656.1 |
| `HIMMELBCLS` | 2 | 106 |
| `HIMMELBF` | 4 | 29053 |
| `HIMMELBG` | 2 | 0.459849 |
| `HIMMELBH` | 2 | 2 |
| `HYDC20LS` | 99 | 1341.66 |
| `HYDCAR6LS` | 29 | 704.107 |
| `JENSMP` | 2 | 4171.31 |
| `JUDGE` | 2 | 2959.94 |
| `KIRBY2LS` | 5 | 373285 |
| `KOWOSB` | 4 | 0.00531362 |
| `LANCZOS1LS` | 6 | 269.75 |
| `LANCZOS2LS` | 6 | 269.75 |
| `LANCZOS3LS` | 6 | 269.751 |
| `LOGHAIRY` | 2 | 6.55252 |
| `LRCOVTYPE` | 54 | 4165.15 |
| `LRIJCNN1` | 22 | 1.52801 |
| `LSC1LS` | 3 | 63019.3 |
| `LSC2LS` | 3 | 235144 |
| `LUKSAN11LS` | 100 | 626.064 |
| `LUKSAN12LS` | 98 | 32160 |
| `LUKSAN13LS` | 98 | 64352 |
| `LUKSAN14LS` | 98 | 26880 |
| `LUKSAN15LS` | 100 | 27015.8 |
| `LUKSAN16LS` | 100 | 13068.5 |
| `LUKSAN17LS` | 100 | 1.68737e+06 |
| `LUKSAN21LS` | 100 | 99.9875 |
| `LUKSAN22LS` | 100 | 24876.9 |
| `MANCINO` | 100 | 1.10327e+12 |
| `MARATOSB` | 2 | 48401.1 |
| `METHANB8LS` | 31 | 1.0431 |
| `METHANL8LS` | 31 | 4345.1 |
| `MEXHAT` | 2 | 1.47548e+06 |
| `MEYER3` | 3 | 1.69361e+09 |
| `MGH09LS` | 4 | 897.545 |
| `MGH10LS` | 3 | 4.51524e+15 |
| `MGH10SLS` | 3 | 4.51524e+15 |
| `MGH17LS` | 5 | 87848.9 |
| `MGH17SLS` | 5 | 87848.9 |
| `MISRA1ALS` | 2 | 10780.2 |
| `MISRA1BLS` | 2 | 10994.3 |
| `MISRA1CLS` | 2 | 11603 |
| `MISRA1DLS` | 2 | 11202.7 |
| `MUONSINELS` | 1 | 61303 |
| `NELSONLS` | 3 | 63.0835 |
| `OSBORNEB` | 11 | 3.16571 |
| `PALMER1C` | 8 | 3.45295e+08 |
| `PALMER1D` | 7 | 2.87266e+07 |
| `PALMER2C` | 8 | 2.6894e+07 |
| `PALMER3C` | 8 | 8.12197e+06 |
| `PALMER4C` | 8 | 8.09445e+06 |
| `PALMER5C` | 6 | 25495 |
| `PALMER5D` | 4 | 22262.6 |
| `PALMER6C` | 8 | 772166 |
| `PALMER7C` | 8 | 3.20513e+06 |
| `PALMER8C` | 8 | 850271 |
| `PARKCH` | 15 | 2150.13 |
| `POWELLBSLS` | 2 | 1.13526 |
| `POWELLSQLS` | 2 | 217.362 |

## References

[1] T. Taminiau, E. Massart, and G. N. Grapiglia. *Enhancing finite-difference based derivative-free optimization methods with machine learning*. 2025. [arXiv:2502.07435](https://arxiv.org/abs/2502.07435).

[2] G. N. Grapiglia. *Worst-case evaluation complexity of a derivative-free quadratic regularization method*. **Optimization Letters**, 18, 1–19, 2024.
