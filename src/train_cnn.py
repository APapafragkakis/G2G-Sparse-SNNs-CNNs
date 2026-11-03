"""Edge pruning in a spiking CNN (Part II).

Strategies:
  dense    no pruning
  pre      random edges removed before training, topology fixed
  dynamic  random edges removed before training, then every few epochs the
           least important active edges are removed and the same number of
           inactive edges is re-activated at random
  post     dense training, then the least important edges are removed and
           the network is fine-tuned

An edge is one kernel K(c', c) of a convolution, its importance is ||K(c', c)||_1.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.cifar10_100 import DATA_ROOT, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from models.spiking_cnn import KS32FullySpikingSmall
from train import select_device


def parse_args():
    p = argparse.ArgumentParser(description="Edge pruning in a spiking CNN.")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--strategy", default="dense", choices=["dense", "pre", "dynamic", "post"])
    p.add_argument("--sparsity", type=float, default=0.5, help="fraction of edges removed in each pruned layer")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--enc", default="direct", choices=["direct", "rate"],
                   help="direct: image repeated T times, rate: Bernoulli spikes")
    p.add_argument("--prune_first_layer", action="store_true",
                   help="also prune CSL1 (it only has 3 input channels, so it is kept dense by default)")
    p.add_argument("--update_every", type=int, default=1, help="dynamic: epochs between updates")
    p.add_argument("--update_frac", type=float, default=0.1, help="dynamic: fraction of active edges replaced per update")
    p.add_argument("--update_until", type=float, default=0.75, help="dynamic: stop updating after this fraction of training")
    p.add_argument("--finetune_epochs", type=int, default=10, help="post: fine-tuning epochs after pruning")
    p.add_argument("--finetune_lr", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def get_loaders(dataset, batch_size, num_workers):
    if dataset == "cifar10":
        cls, mean, std, n_classes = datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10
    else:
        cls, mean, std, n_classes = datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = cls(DATA_ROOT, train=True, download=True, transform=train_tf)
    test_set = cls(DATA_ROOT, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, n_classes


def encode(images, T, enc):
    x = images.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    if enc == "rate":
        # Bernoulli spikes from pixel values rescaled to [0, 1]
        mn = x.amin(dim=(2, 3, 4), keepdim=True)
        mx = x.amax(dim=(2, 3, 4), keepdim=True)
        x = torch.bernoulli((x - mn) / (mx - mn + 1e-8))
    return x


def prunable_convs(model, prune_first_layer):
    convs = model.masked_convs()
    return convs if prune_first_layer else [c for c in convs if c is not model.csl1.conv]


@torch.no_grad()
def random_masks(convs, sparsity):
    for conv in convs:
        n_edges = conv.edge_mask.numel()
        n_off = int(round(sparsity * n_edges))
        mask = torch.ones(n_edges, device=conv.edge_mask.device)
        mask[torch.randperm(n_edges, device=mask.device)[:n_off]] = 0
        conv.edge_mask.copy_(mask.view_as(conv.edge_mask))


@torch.no_grad()
def magnitude_masks(convs, sparsity):
    """Keep the edges with the largest L1 norm in each layer."""
    for conv in convs:
        importance = conv.edge_importance().flatten()
        n_off = int(round(sparsity * importance.numel()))
        mask = torch.ones_like(importance)
        if n_off > 0:
            mask[torch.topk(importance, n_off, largest=False).indices] = 0
        conv.edge_mask.copy_(mask.view_as(conv.edge_mask))


@torch.no_grad()
def dynamic_update(convs, frac, optimizer):
    """Remove the least important active edges and re-activate the same number at random."""
    for conv in convs:
        mask = conv.edge_mask.flatten()
        active = mask.nonzero().flatten()
        inactive = (mask == 0).nonzero().flatten()
        n = min(int(frac * active.numel()), inactive.numel())
        if n == 0:
            continue

        importance = conv.edge_importance().flatten()[active]
        drop = active[torch.topk(importance, n, largest=False).indices]
        grow = inactive[torch.randperm(inactive.numel(), device=mask.device)[:n]]

        mask[drop] = 0
        mask[grow] = 1
        conv.edge_mask.copy_(mask.view_as(conv.edge_mask))

        # Re-activated edges start from zero, with no momentum left from before
        out_idx, in_idx = grow // conv.edge_mask.size(1), grow % conv.edge_mask.size(1)
        conv.weight[out_idx, in_idx] = 0
        state = optimizer.state.get(conv.weight, {})
        if "momentum_buffer" in state and state["momentum_buffer"] is not None:
            state["momentum_buffer"][out_idx, in_idx] = 0


def edge_sparsity(convs):
    total = sum(c.edge_mask.numel() for c in convs)
    off = sum((c.edge_mask == 0).sum().item() for c in convs)
    return off / total if total else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, T, enc):
    model.train()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(encode(images, T, enc))
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device, T, enc):
    model.eval()
    correct = total = 0
    layer_sums = [0.0] * 7
    sizes = None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out, rates, sizes = model(encode(images, T, enc), return_layer_spikes=True)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        layer_sums = [a + b for a, b in zip(layer_sums, rates)]

    layer_rates = [s / total for s in layer_sums]
    overall = sum(r * n for r, n in zip(layer_rates, sizes)) / sum(sizes)
    return correct / total, overall, layer_rates


def fit(model, train_loader, test_loader, device, args, epochs, lr, convs=None, dynamic=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.T, args.enc)
        scheduler.step()

        if dynamic and epoch % args.update_every == 0 and epoch <= args.update_until * epochs:
            dynamic_update(convs, args.update_frac, optimizer)

        test_acc, _, _ = evaluate(model, test_loader, device, args.T, args.enc)
        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}", flush=True)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = select_device()
    train_loader, test_loader, n_classes = get_loaders(args.dataset, args.batch_size, args.num_workers)
    model = KS32FullySpikingSmall(num_classes=n_classes).to(device)
    convs = prunable_convs(model, args.prune_first_layer)

    print(f"[CONFIG] dataset={args.dataset} | strategy={args.strategy} | sparsity={args.sparsity} | "
          f"T={args.T} | epochs={args.epochs} | batch={args.batch_size} | enc={args.enc}")

    if args.strategy in ("pre", "dynamic"):
        random_masks(convs, args.sparsity)

    fit(model, train_loader, test_loader, device, args, args.epochs, args.lr,
        convs=convs, dynamic=(args.strategy == "dynamic"))

    if args.strategy == "post":
        magnitude_masks(convs, args.sparsity)
        acc, _, _ = evaluate(model, test_loader, device, args.T, args.enc)
        print(f"[Post] pruned to {edge_sparsity(convs):.2%} edge sparsity | test_acc before fine-tuning={acc:.4f}")
        fit(model, train_loader, test_loader, device, args, args.finetune_epochs, args.finetune_lr)

    acc, overall, layer_rates = evaluate(model, test_loader, device, args.T, args.enc)
    print("\n" + "=" * 70)
    print(f"Strategy: {args.strategy} | edge sparsity: {edge_sparsity(convs):.2%}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Overall firing rate: {overall:.4f}")
    print("Layerwise firing rates: " + " ".join(f"CSL{i + 1}={r:.4f}" for i, r in enumerate(layer_rates)))
    print("=" * 70)

    os.makedirs("checkpoints", exist_ok=True)
    name = f"cnn_{args.dataset}_{args.strategy}_s{args.sparsity:.2f}".replace(".", "_") + ".pth"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "test_acc": acc},
               os.path.join("checkpoints", name))


if __name__ == "__main__":
    main()
