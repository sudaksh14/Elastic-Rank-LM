import torch
import torch.nn.functional as F
from torch import nn


class DynamicLowRankLinear(nn.Module):
    """
    A linear layer with a dynamically adjustable low-rank factorization.
    The weight matrix W is approximated as W = B @ A, where A projects the
    input to a low-rank space and B projects back to the output space.
    The active rank can be changed at any time without reallocation.
    """
    def __init__(self, in_features, out_features, max_rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.active_rank = max_rank

        self.A = nn.Linear(in_features, max_rank, bias=False)
        self.B = nn.Linear(max_rank, out_features, bias=bias)

    def set_rank(self, rank):
        self.active_rank = max(1, min(rank, self.max_rank))

    def forward(self, x):
        r = self.active_rank
        A_weight = self.A.weight[:r, :]   # (r, in_features)
        B_weight = self.B.weight[:, :r]   # (out_features, r)
        hidden = F.linear(x, A_weight)
        return F.linear(hidden, B_weight, self.B.bias)

    def forward_with_hidden(self, x):
        """Forward pass that also returns the intermediate low-rank representation."""
        r = self.active_rank
        h = F.linear(x, self.A.weight[:r], bias=None)
        out = F.linear(h, self.B.weight[:, :r], bias=self.B.bias)
        return out, h

    def __repr__(self):
        return (
            f"DynamicLowRankLinear(in={self.in_features}, out={self.out_features}, "
            f"max_rank={self.max_rank}, active_rank={self.active_rank})"
        )


def replace_linear_with_lowrank(model: nn.Module, max_rank: int, target_substring: str):
    """
    Replaces nn.Linear layers whose name contains `target_substring`
    with DynamicLowRankLinear layers (structure only, no weight init).
    """
    for full_name, module in list(model.named_modules()):
        if target_substring in full_name and isinstance(module, nn.Linear):
            name_parts = full_name.split('.')
            parent_name = '.'.join(name_parts[:-1])
            local_name = name_parts[-1]
            parent_module = model.get_submodule(parent_name)

            new_layer = DynamicLowRankLinear(
                module.in_features, module.out_features, max_rank,
                bias=(module.bias is not None),
            )
            setattr(parent_module, local_name, new_layer)
    return model


def decompose_and_replace(model: nn.Module, max_rank: int, target_substring: str = "mlp"):
    """
    Replaces nn.Linear layers with DynamicLowRankLinear layers, initialising
    the factors via truncated SVD of the original weight matrix.
    """
    for full_name, module in list(model.named_modules()):
        if target_substring in full_name and isinstance(module, nn.Linear):
            W = module.weight.data.to(torch.float32)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            r = min(max_rank, S.size(0))
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
            S_sqrt = torch.sqrt(S)

            B_weight = U @ torch.diag(S_sqrt)    # (out_features, r)
            A_weight = torch.diag(S_sqrt) @ Vh   # (r, in_features)

            new_layer = DynamicLowRankLinear(
                module.in_features, module.out_features, max_rank,
                bias=(module.bias is not None),
            ).to(module.weight.device).to(module.weight.dtype)

            new_layer.A.weight.data = A_weight.to(new_layer.A.weight.dtype)
            new_layer.B.weight.data = B_weight.to(new_layer.B.weight.dtype)
            if module.bias is not None:
                new_layer.B.bias.data = module.bias.data.to(new_layer.B.bias.dtype)

            name_parts = full_name.split('.')
            parent_name = '.'.join(name_parts[:-1])
            local_name = name_parts[-1]
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, local_name, new_layer)

    return model
