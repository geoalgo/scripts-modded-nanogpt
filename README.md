# Script for Modded NanoGPT Speed Runs

Script to launch speed-runs of NanoGPT.

## 🎯 Objective

The goal of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/)
is to train a language model to achieve **3.28 cross-entropy loss** on the FineWeb validation set using 8×H100 GPUs 
in minimal time. 
Current records have improved from 45 minutes down to **under 3 minutes**.

This repository shows how to reproduce this result using [Skypilot](https://github.com/skypilot-org/skypilot), 
by default Lambdalabs is used due to its good price and overall performance but you can swap to any cloud provider.


## 🚀 Quick Start

These instructions use `uv` but are easily adaptable to conda or pip.

### Prerequisites
- Access to 8×H100 GPUs configured to some cloud provider, see [Skypilot instructions](https://docs.skypilot.co/en/latest/getting-started/installation.html)
- `uv` installed, see [uv doc](https://docs.astral.sh/uv/getting-started/installation/) to install it or adapt instructions to conda/pip

### Setup to launch with Slurmpilot

```bash
# Clone the repository
git clone https://github.com/geoalgo/scripts-modded-nanogpt.git
cd scripts-modded-nanogpt

# Initialize environment and install dependencies
uv init .
uv add skypilot

# Launch training instance
uv run sky launch modded-125M.yaml
```

### Setup to launch directly in a GPU node

```bash
# Clone the repository
git clone https://github.com/geoalgo/scripts-modded-nanogpt.git
cd scripts-modded-nanogpt

git checkout hpo

# Initialize environment and install dependencies
uv init . --python 3.11
uv sync
uv pip install -r requirements.txt
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

# Launch training instance
uv run sky launch modded-125M.yaml
```




## 📊 Performance

- **Training time**: 2.856 minutes
- **Hardware**: 8×H100 GPUs via [Lambda Labs](https://cloud.lambda.ai/)
- **Estimated cost**: ~$3 ($1.2 for training the rest for installing dependencies and downloading dataset)

## 🛠 Configuration

The `modded-125M.yaml` file contains the optimized configuration for:
- Model architecture (125M parameters)
- Training hyperparameters
- Hardware specifications
- Cloud provider settings

## 📈 Results

Upon completion, the training run will:
- Achieve the target 3.28 cross-entropy loss
- Generate training logs and metrics
- Save the final model checkpoint

## 🤝 Contributing

Contributions to improve training speed and efficiency are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with performance benchmarks

## 📄 License

This project builds upon the original [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/) work. Please refer to the original repository for licensing details.