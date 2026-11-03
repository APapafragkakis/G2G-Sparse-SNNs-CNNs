"""k-fold cross-validation for the spiking MLPs.

Splits the training set into k folds, trains a fresh model on k-1 folds and
reports the validation accuracy on the held-out fold. Takes the same model
arguments as train.py, plus --k_folds.

The last line has the format read by tools/cv_results.py:
    Val accuracy over folds: 88.12% ± 0.35%
"""

import argparse
import statistics
import sys

import torch
from torch.utils.data import DataLoader, Subset

import train
from data.fashionmnist import get_fashion_loaders
from data.cifar10_100 import get_cifar10_loaders, get_cifar100_loaders


def parse_args():
    # Reuse train.py's arguments, with a shorter default for --epochs
    argv = sys.argv[1:]
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=42, help="seed for the fold split")
    cv_args, rest = p.parse_known_args(argv)

    if "--epochs" not in rest:
        rest += ["--epochs", "5"]
    sys.argv = [sys.argv[0]] + rest
    args = train.parse_args()
    args.k_folds = cv_args.k_folds
    args.split_seed = cv_args.split_seed
    return args


def setup_globals(args):
    """Set the train.py globals that main() normally sets."""
    train.T = args.T
    train.batch_size = args.batch_size
    train.hidden_dim = args.hidden_dim
    train.cp_mode, train.cg_mode = args.cp, args.cg
    train.enc_mode = args.enc
    train.enc_scale, train.enc_bias = float(args.enc_scale), float(args.enc_bias)

    if args.dataset == "fashionmnist":
        train.input_dim, train.num_classes = 28 * 28, 10
    else:
        train.input_dim = 3 * 32 * 32
        train.num_classes = 10 if args.dataset == "cifar10" else 100


def get_train_set(args):
    normalize = bool(args.use_resnet)
    if args.dataset == "fashionmnist":
        loader, _ = get_fashion_loaders(args.batch_size)
    elif args.dataset == "cifar10":
        loader, _ = get_cifar10_loaders(args.batch_size, normalize=normalize)
    else:
        loader, _ = get_cifar100_loaders(args.batch_size, normalize=normalize)
    return loader.dataset


def main():
    args = parse_args()
    setup_globals(args)
    device = train.select_device()

    dataset = get_train_set(args)
    g = torch.Generator().manual_seed(args.split_seed)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    folds = [indices[i::args.k_folds] for i in range(args.k_folds)]

    use_dst = args.sparsity_mode == "dynamic"
    accs = []

    print(f"[CV] dataset={args.dataset} | model={args.model} | p_inter={args.p_inter} | "
          f"k={args.k_folds} | epochs per fold={args.epochs}")

    for k in range(args.k_folds):
        val_idx = folds[k]
        train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)

        train.global_step = 0
        model = train.build_model(args.model, args.p_inter, args.dataset, use_resnet=args.use_resnet).to(device)
        if model.feature_extractor is not None:
            model.feature_extractor = model.feature_extractor.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=train.lr, weight_decay=1e-4)

        for epoch in range(1, args.epochs + 1):
            train.train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)

        acc = train.evaluate(model, val_loader, device) * 100
        accs.append(acc)
        print(f"Fold {k + 1}/{args.k_folds}: val accuracy = {acc:.2f}%", flush=True)

    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"Val accuracy over folds: {mean:.2f}% ± {std:.2f}%")


if __name__ == "__main__":
    main()
