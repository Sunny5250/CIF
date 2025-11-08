# Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory

Offical implementation of CIF (AAAI 2026).

## News

* **[2025.11.08]**  🔥🔥 Accepted by **AAAI 2026**!

## Abstract
Few-shot multimodal industrial anomaly detection is a critical yet underexplored task, offering the ability to quickly adapt to complex industrial scenarios. In few-shot settings, insufficient training samples often fail to cover the diverse patterns present in test samples. This challenge can be mitigated by extracting structural commonality from a small number of training samples. In this paper, we propose a novel few-shot unsupervised multimodal industrial anomaly detection method based on structural commonality, **CIF** (**C**ommonality **I**n **F**ew). To extract intra-class structural information, we employ hypergraphs, which are capable of modeling higher-order correlations, to capture the structural commonality within training samples, and use a memory bank to store this intra-class structural prior. Firstly, we design a semantic-aware hypergraph construction module tailored for single-semantic industrial images, from which we extract common structures to guide the construction of the memory bank. Secondly, we use a training-free hypergraph message passing module to update the visual features of test samples, reducing the distribution gap between test features and features in the memory bank. We further propose a hyperedge-guided memory search module, which utilizes structural information to assist the memory search process and reduce the false positive rate. Experimental results on the MVTec 3D-AD dataset and the Eyecandies dataset show that our method outperforms the state-of-the-art (SOTA) methods in few-shot settings.


## Datasets

We use [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and [Eyecandies](https://eyecan-ai.github.io/eyecandies/) for experiments.

## Installation

We implement this repo with the following environment:
- Python 3.8.11
- Pytorch 1.13.1
- CUDA 11.7

To run experiments, first clone the repository and install the dependency packages:

```bash
# create virtual environment
conda create -n cif python==3.8.11
conda activate cif

# install pytorch (1.13.1+cu117)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install other packages
pip install -r requirements.txt
# install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# install pointnet2_ops_lib
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# if you can't install pointnet2_ops_lib using the command above, try this:
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/
pip install pointnet2_ops_lib/
```

## Training and Testing

### 1. Checkpoints

Download pre-trained models [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth) and [Point-MAE](https://drive.google.com/file/d/1-wlRIz0GM8o6BuPTJz4kTt6c_z1Gh6LX/view?usp=sharing), then put them in `checkpoints` folder.

### 2. Data preprocessing

Only if you use Eyecandies, you need to preprocess the data with the following command:

```bash
python data/preprocess_eyecandies.py --dataset_path [dataset_path] --target_dir [target_dir]
```

### 3. Training

Run the following command to train our model on all categories of MVTec 3D-AD and Eyecandies dataset:

```bash
./scripts/train.sh
```

### 4. Testing

Run the following command to evaluate the models on all categories of MVTec 3D-AD and Eyecandies dataset:

```bash
./scripts/eval.sh
```
