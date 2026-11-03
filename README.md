# Structured Sparsity in Spiking Neural Networks

G2G (group-to-group) connectivity, Dynamic Sparse Training and edge pruning in spiking MLPs and spiking CNNs.

Alexandros Papafragkakis

Full report: [REPORT.pdf](REPORT.pdf)

## Highlights

- **ResNet–SNN on CIFAR-10:** sparse G2G classifiers match the dense SNN (93.4–93.5%) with 2–3.7× lower firing rate for p′ ≤ 0.5. At p′ = 0, Index reaches 93.47% at a firing rate of 0.059, against 93.44% at 0.216 for Dense.
- **Spiking CNN on CIFAR-10:** dynamic pruning beats pre-training pruning at every sparsity level (79.99% vs 79.04% at 70%, 69.11% vs 66.76% at 95%). Post-training pruning collapses above 50% sparsity (16.80% at 70%).
- **Dynamic Sparse Training:** improves static sparsity by up to +2.7 points and keeps Random from collapsing at p′ = 0.75 on CIFAR-10 (29.6% → 39.7%).
- **Fashion-MNIST:** every G2G variant beats a parameter-matched dense baseline (Index 89.52% vs Dense 87.04%).

## Part I: G2G Connectivity in Spiking MLPs

The classifier is a spiking MLP with three hidden LIF layers. In the G2G models each layer is split into 8 groups. Connections inside a group are always active (p = 1). Connections between groups are kept with probability **p′**, so a higher p′ means more inter-group connections and a denser network.

![Spiking MLP with index-based G2G connectivity](docs/figures/spiking_mlp.png)

Three ways to assign neurons to groups:

- **Index:** contiguous blocks of neurons, which keeps spatial locality across layers
- **Random:** random assignment, which keeps the sparsity but removes the structure
- **Mixer:** index-based groups in the first and last hidden layer and interleaved groups (neuron l in group l mod G) in the middle one, so features mix across groups in a controlled way

![Index, Random and Mixer grouping](docs/figures/grouping_strategies.png)

**Dense** is the fully connected baseline. For comparisons it is capacity-matched to the sparse models.

### ResNet–SNN (CIFAR-10)

A frozen, truncated ResNet-32 pretrained on CIFAR extracts features (cut after `layer1` with 4×4 pooling, a 256-D embedding). The features are min–max normalized per sample, rate-encoded into spike trains and classified by the G2G spiking MLP. Accuracy saturates, so firing rate becomes the main difference between models.

| p′ | Dense* | Index | Mixer | Random |
|---|---|---|---|---|
| 0.00 | 93.44 / 0.2159 | **93.47 / 0.0590** | 93.43 / 0.0924 | 93.39 / 0.0892 |
| 0.15 | 93.48 / 0.1881 | 93.49 / 0.0766 | **93.50** / 0.0852 | 93.45 / **0.0632** |
| 0.50 | **93.51** / 0.1600 | 93.37 / 0.0823 | 93.43 / 0.0952 | **93.51 / 0.0816** |
| 0.75 | 93.42 / 0.1283 | 93.46 / 0.1048 | **93.49 / 0.1019** | 93.36 / 0.1093 |

Best test accuracy (%) / overall firing rate. \*Dense is fully connected and capacity-matched to the sparse models at each p′.

### Static sparsity vs Dynamic Sparse Training (CIFAR-10 / CIFAR-100)

DST keeps the number of connections fixed and rewires them during training. Every 1,000 steps it prunes 10% of the active connections and grows the same number of new ones.

| Pruning (`--cp`) | Removes | Growth (`--cg`) | Adds |
|---|---|---|---|
| `set` | smallest-magnitude weights | `random` | random inactive edges |
| `random` | random edges | `hebb` | edges with high pre/post co-activation |
| `hebb` | edges with low pre/post co-activation | | |

![Accuracy vs p′ for Dense, Static and DST](docs/figures/dst_vs_static.png)

- Sparse models beat the dense SNN (35.7% on CIFAR-10, 11.1% on CIFAR-100) everywhere except static sparsity at p′ = 0.75 (Random on CIFAR-10, Index on CIFAR-100).
- With DST, all architectures peak at p′ = 0.05. On CIFAR-10: Index 47.10%, Mixer 47.07%, Random 45.33%. On CIFAR-100: Mixer 21.57%, Random 19.68%, Index 19.27%.
- DST helps most where static sparsity struggles, e.g. Random at p′ = 0.75 on CIFAR-10 goes from 29.6% to 39.7%.

### Fashion-MNIST

| Strategy | Best test accuracy | at p′ |
|---|---|---|
| Index | **89.52%** | 0.05 |
| Mixer | 89.42% | 0.10 |
| Random | 88.73% | 0.25 |
| Dense (447 hidden units, ~754k parameters) | 87.04% | – |

Three hidden layers of 1024 neurons, 8 groups, T = 50, 20 epochs. The dense baseline is matched to the parameter count of the sparse models.

## Part II: Edge Pruning in Spiking CNNs

A convolutional layer can be seen as a graph. Input channel c and output channel c′ are nodes, and the kernel K(c′,c) is the edge between them. Removing an edge means setting its kernel to zero. Edge importance is the L1 norm of the kernel.

<img src="docs/figures/spiking_cnn.png" alt="Spiking CNN with seven convolutional spiking layers" width="320" align="right">

The network (`KS32_FullySpiking_Small`) has seven convolutional spiking layers (conv + batch norm + spiking neurons) with residual connections, 4×4 adaptive average pooling and two dense spiking layers.

Three pruning strategies:

- **Pre:** a fixed percentage of edges is removed at random before training
- **Dynamic:** starts like Pre, then periodically removes the least important edges and re-activates the same number at random
- **Post:** the dense network is trained first, then low-importance edges are removed and the network is fine-tuned

CIFAR-10, T = 4, 50 epochs, SGD with cosine annealing. Dense: **83.37%**.

| Sparsity | Pre | Dynamic | Post |
|---|---|---|---|
| 30% | 82.19 | **82.46** | 75.90 |
| 50% | 80.68 | **81.98** | 60.30 |
| 70% | 79.04 | **79.99** | 16.80 |
| 90% | 71.44 | **72.50** | 12.57 |
| 95% | 66.76 | **69.11** | 9.96 |

<br clear="right">

<img src="docs/figures/cnn_pruning.png" alt="Accuracy and firing rate vs sparsity for Pre, Dynamic and Post pruning" width="600">

Dynamic pruning works best because it decides which edges to remove after the network has started learning, and it changes the topology a little at a time. Post-training pruning removes too much at once at high sparsity, and fine-tuning cannot recover.

## Code

- `src/train.py`: Part I, the Dense, Index, Random and Mixer spiking MLPs with static or dynamic sparsity, with or without the ResNet feature extractor
- `src/train_cnn.py`: Part II, edge pruning in the spiking CNN
- `src/crossval.py`: k-fold cross-validation for the spiking MLPs
- `src/run_ablation_one.py`: intra-group ablation of a trained spiking MLP

### Installation

Requires Python 3.12+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

On Windows with an AMD GPU you can use DirectML (`pip install torch-directml`), but it needs a PyTorch version that DirectML supports. The device is picked automatically: DirectML, then CUDA, then CPU.

Datasets are downloaded to `data/` the first time you run.

### Training

Run from the repository root:

```bash
python src/train.py --dataset DATASET --model MODEL [OPTIONS]
```

| Argument | Description | Default |
|---|---|---|
| `--dataset` | `fashionmnist`, `cifar10`, `cifar100` | `fashionmnist` |
| `--model` | `dense`, `index`, `random`, `mixer` | `dense` |
| `--p_inter` | inter-group connection probability p′ (sparse models) | `0.15` |
| `--hidden_dim` | neurons per hidden layer | `2048` |
| `--epochs` | training epochs | `20` |
| `--T` | simulation timesteps | `50` |
| `--batch_size` | batch size | `256` |
| `--enc` | input encoding without ResNet: `current` (analog input) or `rate` (Bernoulli spikes) | `current` |
| `--enc_scale`, `--enc_bias` | scale and bias applied to the input before encoding | `1.0`, `0.0` |
| `--use_resnet` | frozen ResNet-32 feature extractor (CIFAR only) | off |
| `--sparsity_mode` | `static` or `dynamic` (DST) | `static` |
| `--cp` | DST pruning rule: `set`, `random`, `hebb` | `set` |
| `--cg` | DST growth rule: `hebb`, `random` | `hebb` |
| `--enforce_sparsity` | firing-rate regularization | off |
| `--warmup_epochs` | epochs before the regularization starts | `10` |

Training uses Adam (lr 1e-3, weight decay 1e-4) and cross-entropy on the output spike counts. At the end it prints the per-layer firing rates. No random seed is fixed, so results vary slightly between runs.

### Examples

```bash
# Fashion-MNIST, Index with p' = 0.05
python src/train.py --dataset fashionmnist --model index --p_inter 0.05 --hidden_dim 1024 --enc rate

# CIFAR-10, Mixer with static sparsity
python src/train.py --dataset cifar10 --model mixer --p_inter 0.05

# CIFAR-10, Index with DST (Hebbian pruning and growth)
python src/train.py --dataset cifar10 --model index --p_inter 0.05 --sparsity_mode dynamic --cp hebb --cg hebb

# ResNet–SNN on CIFAR-10
python src/train.py --dataset cifar10 --model index --use_resnet --p_inter 0.15 --hidden_dim 512
```

The sweeps used for the report are in `scripts/` (Windows batch files).

### Spiking CNN pruning

```bash
python src/train_cnn.py --strategy STRATEGY --sparsity SPARSITY [OPTIONS]
```

| Argument | Description | Default |
|---|---|---|
| `--strategy` | `dense`, `pre`, `dynamic`, `post` | `dense` |
| `--sparsity` | fraction of edges removed in each pruned layer | `0.5` |
| `--dataset` | `cifar10`, `cifar100` | `cifar10` |
| `--epochs`, `--T`, `--batch_size` | training epochs, timesteps, batch size | `50`, `4`, `256` |
| `--lr`, `--momentum`, `--weight_decay` | SGD with cosine annealing | `0.1`, `0.9`, `5e-4` |
| `--update_every`, `--update_frac`, `--update_until` | dynamic: epochs between updates, fraction of active edges replaced, fraction of training with updates | `1`, `0.1`, `0.75` |
| `--finetune_epochs`, `--finetune_lr` | post: fine-tuning after pruning | `10`, `0.01` |
| `--prune_first_layer` | also prune CSL1 (kept dense by default, it has only 3 input channels) | off |
| `--enc` | `direct` (image repeated T times) or `rate` | `direct` |

```bash
python src/train_cnn.py --strategy dense
python src/train_cnn.py --strategy dynamic --sparsity 0.7
```

The defaults follow the training setup in the report. At the end it prints the test accuracy, the overall firing rate and the firing rate of each CSL. Use `--seed` for repeatable runs.

### Cross-validation

`src/crossval.py` takes the same arguments as `train.py`, plus `--k_folds` (default 5). Each fold trains for `--epochs` epochs (default 5 here). The output can be collected with `tools/cv_results.py`:

```bash
mkdir results_cv_5fold_5ep_p_sweep
python src/crossval.py --dataset fashionmnist --model index --p_inter 0.05 --hidden_dim 1024 --enc rate > results_cv_5fold_5ep_p_sweep/index_p0.05.txt
python tools/cv_results.py
```

### Checkpoints

Each run saves a checkpoint to `checkpoints/` after every epoch and resumes from it if interrupted. When a run finishes, a `.DONE` file is created and the same run is skipped next time. Delete the `.DONE` file to run it again.

### Intra-group ablation

`src/run_ablation_one.py` loads a trained checkpoint, keeps only the intra-group connections and reports the accuracy drop. It takes the same arguments as `train.py`:

```bash
python src/run_ablation_one.py --dataset fashionmnist --model index --p_inter 0.05 --hidden_dim 1024 --enc rate
```

## Repository Layout

```
models/     Dense, Index, Random and Mixer SNNs, spiking CNN, frozen ResNet feature extractor
src/        training scripts, cross-validation, intra-group ablation
data/       dataset loaders (datasets are downloaded here)
utils/      input encoding
tools/      scripts that collected result tables from the original sweep logs
scripts/    sweep scripts used for the report
docs/       figures used in this README
REPORT.pdf  full report
```

## License

The code is released under the [MIT License](LICENSE).
