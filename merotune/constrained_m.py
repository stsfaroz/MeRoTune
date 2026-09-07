"""
RoPE-commutant transform used to align two models' attention subspaces
before merging.

RoPE rotates each (query, key) pair by a position-dependent angle. If
you merge two models' K/Q projections by plain weight averaging, you're
implicitly assuming both models already agree on how their RoPE pairs
are oriented -- usually false, since nothing forces two independently
fine-tuned checkpoints to share that convention. The fix has to commute
with the RoPE rotation itself, or you break the relative-position
property RoPE is there for. The only 2x2 real matrices that commute
with every rotation angle are matrices of the form [[a, b], [-b, a]] --
a scaled rotation. That's what this class represents: one such matrix
per (layer, kv-head), with a and b as the only learnable parameters.

Uses HF's split-half RoPE convention (pair j = (x[j], x[j+half]), not
the interleaved (x[2j], x[2j+1]) convention some other implementations
use).
"""
import torch
import torch.nn as nn


class ConstrainedM(nn.Module):
    def __init__(self, n_layers, n_kv_heads, half_head_dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_layers, n_kv_heads, half_head_dim))
        self.b = nn.Parameter(torch.zeros(n_layers, n_kv_heads, half_head_dim))
        self.half = half_head_dim

    def get_M(self, layer, kv_head):
        a, b = self.a[layer, kv_head], self.b[layer, kv_head]
        h = self.half
        idx = torch.arange(h, device=a.device)
        M = torch.zeros(2 * h, 2 * h, dtype=a.dtype, device=a.device)
        M = M.index_put((idx, idx), a)
        M = M.index_put((idx, idx + h), b)
        M = M.index_put((idx + h, idx), -b)
        M = M.index_put((idx + h, idx + h), a)
        return M

    def n_params(self):
        return self.a.numel() + self.b.numel()
