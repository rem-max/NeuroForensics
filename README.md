# NeuroForensics: An Endogenous Defense Of Covert Federated Learning Backdoor Threats
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Focus](https://img.shields.io/badge/Focus-FL%20Security%20%26%20Defense-green.svg)

*A unified federated learning research platform integrating FL algorithms, adversarial attacks, and advanced defense mechanisms for comprehensive security analysis.*

This repository contains the official PyTorch implementation of **NeuroForensics**. NeuroForensics is designed to address critical security challenges in federated learning systems, providing researchers with a comprehensive toolkit for evaluating vulnerabilities and defense strategies. The framework supports end-to-end experimentation with backdoor attacks, model poisoning, and state-of-the-art defense mechanisms across heterogeneous data distributions.

## 🚀 Key Features

- **Advanced Attack Suite**: BadNets, Model Replacement, A3FL, Blended Attack
- **State-of-the-Art Defenses**: NeuroForensics (layer separation), FLAME (clustering-based), Gradient Clipping, Robust Aggregation (Krum, Trimmed Mean, Median)，AlignIns
- **Non-IID Data Support**: Dirichlet distribution with configurable α, label skew, quantity skew, and unbalanced partitioning
- **Flexible Model Zoo**: CNN, VGG16, ResNet18, MobileNetV2, AlexNet, LeNet, Transformer architectures
- **Comprehensive Evaluation**: Attack Success Rate (ASR), Main Task Accuracy, True Positive Rate, and False Positive Rate metrics

## � Repository Structure

The codebase is organized as follows:

```
NeuroForensics/
├── main.py                           # Entry point: Federated training orchestration
├── dataset/                          # Dataset generation and partitioning
│   ├── generate_Cifar10.py          # CIFAR-10 data preprocessing
│   ├── generate_MNIST.py            # MNIST data preprocessing
│   ├── generate_FashionMNIST.py     # FashionMNIST data preprocessing
│   └── utils/                        # Dataset utilities (Dirichlet split, etc.)
├── system/                           # Core federated learning system
│   ├── main.py                       # System entry point with argument parsing
│   ├── flcore/                       # FL core components
│   │   ├── clients/                  # Client-side implementations (40+ variants)
│   │   ├── servers/                  # Server-side algorithms (FedAvg, FedProx, etc.)
│   │   ├── trainmodel/               # Model architectures (CNN, VGG16, ResNet18, etc.)
│   │   └── optimizers/               # Custom optimizers
│   ├── security/                     # Security module
│   │   ├── attack/                   # Attack implementations
│   │   │   ├── BadNets.py            # BadNets backdoor attack
│   │   │   ├── model_replacement.py  # Model replacement attack
│   │   │   ├── A3FL.py               # Enhanced A3FL attack
│   │   │   └── blended.py            # Data-free backdoor attack
│   │   └── defense/                  # Defense mechanisms
│   │       ├── NeuroForensics.py     # Layer separation defense
│   │       ├── flame.py              # FLAME clustering defense
│   │       ├── gradient_clipping.py  # Gradient norm clipping
│   │       └── robust_aggregation.py # Robust aggregation methods
|   |       └── AlignIns.py           # AlignIns Defense methods
│   └── utils/                        # Utility functions
│       ├── data_utils.py             # Data loading utilities
│       ├── result_utils.py           # Results processing
│       └── ALA.py                    # Adaptive Local Aggregation
├── models/                           # Saved models and checkpoints
└── results/                          # Experimental results and logs
```

## 🛠️ Installation

### Prerequisites
- Python >= 3.8
- CUDA >= 11.1 (for GPU acceleration)
- PyTorch >= 1.8.0

### Clone the Repository
```bash
git clone https://github.com/rem-max/NeuroForensics.git
cd NeuroForensics
```

### Install Dependencies
It is recommended to use a virtual environment:
**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

### Prepare Datasets
Generate federated datasets with Non-IID splits:
```bash
# CIFAR-10 with Dirichlet(α=0.5) distribution
python dataset/generate_Cifar10.py

# MNIST with IID distribution
python dataset/generate_MNIST.py

# MNIST with Dirichlet(α=0.1) distribution (highly non-IID)
python dataset/generate_MNIST_dirichlet_0.1.py
```

## 🏃 Usage

### Quick Start
Run a basic federated learning experiment without attacks:

```bash
cd system
python main.py \
    --dataset Cifar10 \
    --model ResNet18 \
    --algorithm FedAvg \
    --num_clients 20 \
    --global_rounds 100 \
    --device_id 0
```

### Run with Backdoor Attack
Execute Model Replacement attack with 20% malicious clients:

```bash
python main.py \
    --dataset Cifar10 \
    --model ResNet18 \
    --local_batch_size 10 \
    --num_clients 20 \
    --join_ratio 0.5 \
    --attack_type model_replacement \
    --attack_ratio 0.2 \
    --model_replace_target_label 0 \
    --model_replace_trigger_size 4 \
    --model_replace_backdoor_epochs 15 \
    --defense_type none \
    --global_rounds 10
```

### Run with Defense Mechanism
Apply Gradient Clipping defense against attacks:

```bash
python main.py \
    --dataset Cifar10 \
    --model ResNet18 \
    --num_clients 20 \
    --attack_type model_replacement \
    --attack_ratio 0.2 \
    --defense_type gradient_clipping \
    --grad_clip_max_norm 1.0 \
    --grad_clip_detect_threshold 2.0
```

Apply NeuroForensics defense with layer separation:

```bash
python main.py \
    --dataset Cifar10 \
    --model ResNet18 \
    --num_clients 20 \
    --attack_type model_replacement \
    --attack_ratio 0.2 \
    --defense_type neuroforensics \
    --lsep 2
```


### Key Arguments Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `Cifar10` | Dataset name (Cifar10, MNIST, FashionMNIST) |
| `--model` | `CNN` | Model architecture (CNN, VGG16, ResNet18, etc.) |
| `--algorithm` | `FedAvg` | FL algorithm (FedAvg, FedProx, MOON, etc.) |
| `--num_clients` | `20` | Total number of clients |
| `--join_ratio` | `0.5` | Fraction of clients participating per round |
| `--global_rounds` | `100` | Number of communication rounds |
| `--local_batch_size` | `10` | Local training batch size |
| `--learning_rate` | `0.01` | Client learning rate |
| `--attack_type` | `none` | Attack method (model_replacement, badnets, etc.) |
| `--attack_ratio` | `0.0` | Fraction of malicious clients |
| `--defense_type` | `none` | Defense mechanism (neuroforensics, gradient_clipping, etc.) |
| `--lsep` | `2` | Layer separation point for NeuroForensics |

## 📊 Performance

NeuroForensics demonstrates robust defense capabilities across various attack scenarios on Non-IID data distributions.

### Defense Effectiveness Comparison

#### CIFAR-10 (ResNet18)

| Defense | BadNets<br>BA↓/ACC↑ | Blended<br>BA↓/ACC↑ | Model Replacement<br>BA↓/ACC↑ | A3FL<br>BA↓/ACC↑ |
|---------|---------------------|---------------------|-------------------------------|------------------|
| FedAvg | 99.75 / 68.99 | 99.33 / 68.13 | 71.56 / 69.96 | 87.69 / 70.18 |
| Gradient Clipping | 98.3 / 60.74 | 89.18 / 58.87 | 85.47 / 53.7 | 51.34 / 63.17 |
| Median | 99.65 / 67.76 | 95.72 / 67.12 | 7.74 / 66.74 | 81.87 / 69.13 |
| Flame | 24.98 / 68.29 | 32.37 / 68.67 | 10.62 / 66.27 | 55.07 / 68.62 |
| AlignIns | 1.13 / 68.16 | 1.55 / 67.02 | 11.87 / 69.37 | 10.03 / 69.32 |
| **NeuroForensics** | **1.03 / 68.73** | **1.35 / 68.84** | **0.16 / 69.59** | **2.82 / 69.74** |

#### MNIST (CNN)

| Defense | BadNets<br>BA↓/ACC↑ | Blended<br>BA↓/ACC↑ | Model Replacement<br>BA↓/ACC↑ | A3FL<br>BA↓/ACC↑ |
|---------|---------------------|---------------------|-------------------------------|------------------|
| FedAvg | 99.58 / 94.32 | 99.96 / 95.19 | 99.77 / 98.77 | 99.67 / 98.88 |
| Gradient Clipping | 91.91 / 97.78 | 99.93 / 97.86 | 99.87 / 98.26 | 99.95 / 97.53 |
| Median | 37.62 / 97.95 | 35.46 / 97.73 | 9.59 / 98.33 | 99.83 / 97.80 |
| Flame | 12.51 / 97.80 | 10.53 / 96.78 | 10.03 / 97.94 | 99.85 / 98.02 |
| AlignIns | 5.24 / 98.08 | 2.78 / 97.28 | 10.47 / 97.97 | 99.94 / 97.79 |
| **NeuroForensics** | **4.34 / 98.22** | **3.68 / 98.26** | **9.81 / 98.16** | **10.34 / 98.58** |

#### FashionMNIST (VGG16)

| Defense | BadNets<br>BA↓/ACC↑ | Blended<br>BA↓/ACC↑ | Model Replacement<br>BA↓/ACC↑ | A3FL<br>BA↓/ACC↑ |
|---------|---------------------|---------------------|-------------------------------|------------------|
| FedAvg | 99.89 / 87.75 | 99.69 / 89.77 | 99.96 / 90.42 | 98.70 / 89.65 |
| Gradient Clipping | 82.10 / 80.45 | 87.18 / 77.21 | 97.18 / 88.46 | 99.54 / 88.99 |
| Median | 9.12 / 87.70 | 40.67 / 87.06 | 99.81 / 90.38 | 51.68 / 87.85 |
| Flame | 6.42 / 87.68 | 5.47 / 88.54 | 44.00 / 90.04 | 99.17 / 88.92 |
| AlignIns | 5.09 / 87.01 | 4.53 / 88.98 | 10.06 / 89.98 | 12.81 / 88.33 |
| **NeuroForensics** | **4.30 / 89.49** | **4.38 / 89.76** | **10.18 / 90.13** | **8.29 / 89.38** |

**Legend:**
- **BA** (Backdoor Attack Success Rate): Lower is better ↓
- **ACC** (Main Task Accuracy): Higher is better ↑
- **Bold values** indicate NeuroForensics results
- Results based on experiments with 20% malicious clients under Dirichlet(α=0.7) distribution


### 🚀 Methodology Highlights

**NeuroForensics** defends against covert backdoor attacks in Federated Learning through a three-stage pipeline: **Extraction**, **Fusion**, and **Sanitization**.

#### 1. PFV Generation & Posterior Probability Matrix

To capture endogenous backdoor signatures without accessing private data, we decouple the model at a specific layer ($L_{Div}$) and synthesize **Pseudo Feature Vectors (PFV)**. 
First, we optimize a latent vector $x_{c}^{*}$ to maximize the confidence of a specific class $c$:

$$
x_{c}^{*} = \arg \min_{x} \mathcal{L}_{CE}(f_{L_{Div}\rightarrow output}(x), c) + \lambda ||x||_{2}^{2}
$$

We then generate the probability vector $p_{c}$ by passing the PFV through the classifier with **Temperature Scaling** ($T$) to reveal subtle inter-class correlations:

$$
p_{c} = \text{softmax}\left(\frac{f_{L_{Div}\rightarrow output}(x_{c}^{*})}{T}\right)
$$

By stacking these vectors, we construct the **Posterior Probability Matrix ($M_p$)**, which maps the model's directional decision boundaries.

#### 2. PCA-Based Multi-Metric Fusion 

To reveal latent triggers, we explicitly **zero out the diagonal elements** of $M_p$ to mitigate the masking effect of main-task accuracy. We then extract four statistical metrics (Entropy, Max Value, Kurtosis, Box-plot) and fuse them using **Principal Component Analysis (PCA)** to project anomalies into a unified score $M$:

$$
M = \left| w_{PCA}^{T} \cdot \frac{v - \mu}{\sigma} \right|
$$

Here, $w_{PCA}$ is the eigenvector corresponding to the largest eigenvalue. This projection automatically prioritizes the most discriminative features of malicious behavior.

#### 3. Fairness-Aware Adaptive Thresholding

To address Non-IID heterogeneity, we employ a dynamic threshold $\tau_t$ based on the variance ($\sigma^2$) of anomaly scores from the previous round. This prevents the false rejection of benign clients with unique data distributions:

$$
\tau_{t} = 
\begin{cases} 
\text{TopK-Mean}(\mathcal{M}_{t-1}, k), & \text{if } \sigma_{t-1}^{2} > \theta_{var} \quad \text{(High Variance Regime)}\\
\tau_{t-1}, & \text{if } \sigma_{t-1}^{2} \le \theta_{var} \quad \text{(Stable Regime)}
\end{cases}
$$

This mechanism anchors the decision boundary to the upper tail of the anomaly distribution, strictly isolating high-confidence outliers.

## 📜 Citation

If you use NeuroForensics in your research, please cite:


