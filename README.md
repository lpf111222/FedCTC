# FedCTC

This repository contains the source code of the paper:

Client-side Training Calibration in Federated Learning: Improving Global Model under Label Skew

The paper studies how to improve federated learning under label skew from the perspective of confidence calibration. In our method, calibration is performed during client-side local training, rather than as a post-hoc step. Three lightweight calibration strategies are considered:

- Lp Norm regularization
- Label Smoothing
- Mixup

These methods are used to suppress client drift and overconfident predictions caused by skewed local label distributions. The experiments show that calibrated federated training can achieve higher global test accuracy, lower calibration error, better failure prediction, and stronger OOD detection.

The paper is currently under review and has entered the first round of major revision. In this revision, we further expand the experiments by adding:

- a Vision Transformer variant (CCT) in addition to ResNet-18,
- IID and mild Dirichlet settings in addition to the original stronger skew settings,
- and a real-world medical dataset CAMELYON17-WILDS with naturally heterogeneous hospital clients.

--------------------------------------------------

## 1. Installation

Please install the required libraries with:

pip install -r requirements.txt

The requirements.txt file lists the main libraries needed for running the code.

In practice, the version requirements are not very strict: as long as these libraries are installed in a reasonably compatible Python/PyTorch environment, the code should run normally.

--------------------------------------------------

## 2. Supported datasets

This code currently supports the following datasets:

- CIFAR-10
- CIFAR-100
- CAMELYON17-WILDS  (added in the major revision)

### 2.1 CIFAR-10 / CIFAR-100

These datasets will be automatically downloaded by PyTorch to: ./data/

### 2.2 CAMELYON17-WILDS

For the medical dataset used in the revision experiments, please manually download it from:

Baidu Netdisk:
https://pan.baidu.com/s/1tPdu_4Y5x4QS9USJUxGx7w?pwd=fnyv

Extraction code:
fnyv

After downloading, please place the dataset under: ./data/

--------------------------------------------------

## 3. How to run

In the revised codebase, each method is run from its own independent script.
You can directly execute the corresponding file for the method you want.

Examples:

python FL_FedAvg.py
python FL_FedCL_LogitNorm.py
python FL_FedCL_LabelSmooth.py
python FL_FedCL_MixUp.py
python FL_FedCL_3In1.py

Where:

- FL_FedAvg.py : vanilla FedAvg baseline
- FL_FedCL_LogitNorm.py : federated training with Lp Norm regularization
- FL_FedCL_LabelSmooth.py : federated training with Label Smoothing
- FL_FedCL_MixUp.py : federated training with Mixup
- FL_FedCL_3In1.py : combination of all three calibration strategies

--------------------------------------------------

## 4.Code structure

The code is organized in a modular way.

Main files include:

- client_base.py : base class for FL clients
- server_base.py : base class for FL servers
- dataset_division.py : dataset loading and client partitioning
- model.py : model definitions
- util.py : utility functions
- FL_FedAvg.py : FedAvg baseline
- FL_FedCL_LogitNorm.py : Lp Norm regularization
- FL_FedCL_LabelSmooth.py : Label Smoothing
- FL_FedCL_MixUp.py : Mixup
- FL_FedCL_3In1.py : combined calibration method

Each calibration method only modifies the necessary part of the standard FL training pipeline, so the framework is easy to extend.

--------------------------------------------------

## 5.Hyperparameter settings

All experimental hyperparameters are defined in the configuration file:

conf.json

Below is a brief explanation of the main options.

### 5.1 Model

- model_name
  Selects the backbone model.
  - "resnet18" : ResNet-18
  - "vit" : Vision Transformer-style model (CCT)

### 5.2 Dataset

- dataset_name
  - "cifar10"
  - "cifar100"
  - "CAMELYON17-WILDS"

Note:
when using CAMELYON17-WILDS, the number of FL clients must be 3, because the revised experiments treat the three hospitals/centers as three natural federated clients.

### 5.3 Federated learning settings

- num_client
  Total number of FL clients.

- k
  Number of participating clients in each communication round.

- global_epochs
  Total number of communication rounds.

- local_epochs
  Number of local training epochs per selected client in each round.

- batch_size
  Local training batch size.

- optimizer
  Local optimizer, e.g. "SGD" or "Adagrad".

- lr
  Learning rate.

### 5.4 Data partition settings

- data_distribution
  - "IID" : IID client data
  - "Dirichlet" : Dirichlet-based label skew
  - "LongTail" : long-tailed class distribution

- dirichlet_alpha_min, dirichlet_alpha_max
  These two hyperparameters control the degree of label skew in the Dirichlet setting.
  Smaller values mean stronger heterogeneity / stronger label skew.
  Larger values mean milder skew, closer to IID.

- longtail_exp
  Controls the decay factor in the Long-Tail setting.
  For CIFAR-10, the paper uses 0.7.
  For CIFAR-100, the paper uses 0.99.

### 5.5 Calibration-specific hyperparameters

- lable_smooth_p
  Ground-truth class probability used in Label Smoothing.

- mixup_alpha
  Beta distribution parameter for Mixup.

- lp
  Weight of the Lp Norm regularization term.

--------------------------------------------------

## 6.Experimental notes

The original experiments in the paper mainly use:

- 100 clients
- 20 participating clients per round
- 2000 global rounds

The revised experiments additionally include:

- IID
- mild Dirichlet
- Dirichlet
- Long-Tail
- ResNet-18
- CCT
- CAMELYON17-WILDS

For CAMELYON17-WILDS, the setup is different from CIFAR:

- it is a binary classification medical dataset,
- the clients are natural hospitals/centers,
- and the total number of clients is fixed to 3.

--------------------------------------------------

## 7.Remarks

This repository is being updated together with the first-round major revision of the paper.

If you find any issue in running the code, please first check:

1. whether all required libraries are installed,
2. whether the dataset path is correct,
3. whether the chosen hyperparameters are compatible with the selected dataset and model.
