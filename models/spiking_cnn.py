import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


def lif():
    # Same LIF settings as the spiking MLPs
    return snn.Leaky(
        beta=0.9,
        spike_grad=surrogate.atan(),
        init_hidden=True,
        learn_beta=True,
        learn_threshold=True,
        threshold=1.0,
    )


class MaskedConv2d(nn.Conv2d):
    """Conv2d where each edge c -> c' (one k x k kernel) can be switched off.

    edge_mask has shape [out_channels, in_channels]. A zero removes the whole
    kernel K(c', c), so input channel c no longer contributes to output channel c'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("edge_mask", torch.ones(self.out_channels, self.in_channels // self.groups))

    def forward(self, x):
        weight = self.weight * self.edge_mask[:, :, None, None]
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def edge_importance(self):
        """L1 norm of each kernel: importance(c -> c') = ||K(c', c)||_1."""
        return self.weight.detach().abs().sum(dim=(2, 3))


class CSL(nn.Module):
    """Convolutional Spiking Layer: conv -> batch norm -> LIF."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = MaskedConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = lif()

    def forward(self, x):
        return self.lif(self.bn(self.conv(x)))


class ResidualStage(nn.Module):
    """Two CSLs with a skip connection around them (spikes are added)."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.a = CSL(in_ch, out_ch, stride)
        self.b = CSL(out_ch, out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        skip = x if self.shortcut is None else self.shortcut(x)
        out_a = self.a(x)
        out_b = self.b(out_a)
        return out_b + skip, (out_a, out_b)


class KS32FullySpikingSmall(nn.Module):
    """Spiking CNN: CSL1, three residual stages (CSL2-CSL7), 4x4 adaptive
    average pooling and two dense spiking layers (DSL1-DSL2).

    Channels: 32 (CSL1-3), 64 (CSL4-5), 128 (CSL6-7).
    """

    def __init__(self, num_classes=10, in_channels=3, width=32, hidden=256):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        self.csl1 = CSL(in_channels, c1)
        self.stage1 = ResidualStage(c1, c1, stride=1)  # CSL2, CSL3
        self.stage2 = ResidualStage(c1, c2, stride=2)  # CSL4, CSL5
        self.stage3 = ResidualStage(c2, c3, stride=2)  # CSL6, CSL7
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(c3 * 4 * 4, hidden)
        self.lif_fc1 = lif()
        self.fc2 = nn.Linear(hidden, num_classes)
        self.lif_fc2 = lif()

    def masked_convs(self):
        return [m for m in self.modules() if isinstance(m, MaskedConv2d)]

    def _reset_states(self):
        for m in self.modules():
            if isinstance(m, snn.Leaky):
                m.reset_mem()

    def forward(self, x_seq, return_layer_spikes=False):
        """x_seq: [T, B, C, H, W]. Returns output spike counts [B, num_classes].
        With return_layer_spikes=True also returns, per CSL, the spike count per
        neuron and timestep summed over the batch, and the number of neurons."""
        self._reset_states()
        T_steps = x_seq.size(0)

        out_sum = 0.0
        spike_sums = [0.0] * 7
        sizes = [0] * 7

        for t in range(T_steps):
            s1 = self.csl1(x_seq[t])
            x, (s2, s3) = self.stage1(s1)
            x, (s4, s5) = self.stage2(x)
            x, (s6, s7) = self.stage3(x)

            x = self.pool(x).flatten(1)
            x = self.lif_fc1(self.fc1(x))
            out = self.lif_fc2(self.fc2(x))
            out_sum = out_sum + out

            if return_layer_spikes:
                for i, s in enumerate((s1, s2, s3, s4, s5, s6, s7)):
                    spike_sums[i] += s.detach().sum().item()
                    sizes[i] = s[0].numel()

        if return_layer_spikes:
            # spikes per neuron per timestep, summed over the batch (divide by batch size later)
            return out_sum, [sp / (T_steps * n) for sp, n in zip(spike_sums, sizes)], sizes
        return out_sum
