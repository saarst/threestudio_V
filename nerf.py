import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vjp, jvp
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import Callable


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2. ** (num_freqs - 1), num_freqs)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (..., 2) for 2D input.
        Returns:
            Positional encoded tensor of shape (..., encoded_dim)
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            for fn in [torch.sin, torch.cos]:
                out.append(fn(x * freq))
        return torch.cat(out, dim=-1)

    @property
    def out_dim(self):
        return 2 * (1 + 2 * self.num_freqs) if self.include_input else 2 * 2 * self.num_freqs

class NeRF2D(nn.Module):
    def __init__(self, D=8, W=256, num_freqs=10, output_ch=32, skips=[4]):
        super().__init__()
        self.embedder = PositionalEncoding2D(num_freqs=num_freqs)
        input_ch = self.embedder.out_dim
        self.skips = skips

        self.layers = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.feature_linear = nn.Linear(W, W)

        self.views_linears = nn.ModuleList([nn.Linear(W, W//2)])

        self.rgb_linear = nn.Linear(W//2, output_ch)

    def forward(self, coords, height=None, width=None):
        x_embed = self.embedder(coords)
        h = x_embed
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_embed, h], -1)

        h = self.feature_linear(h)
        for view_layer in self.views_linears:
            h = view_layer(h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)  # (H*W, C)
        # Optional reshaping (safe for jvp/vjp)
        if height is not None and width is not None:
            rgb = rgb.view(height, width, -1).permute(2, 0, 1).unsqueeze(0)
        return rgb
    
# ---------------------
# Step 2: Flatten params for functional use
# ---------------------
def get_flat_params(model):
    param_dict = dict(model.named_parameters())
    flat_params, param_spec = tree_flatten(param_dict)
    return flat_params, param_spec

def rebuild_params(flat_params, param_spec):
    return tree_unflatten(flat_params, param_spec)

# ---------------------
# Step 3: Functional version of the model
# ---------------------
# def forward_with_params(flat_params, param_spec, model, coords, height, width):
#     param_dict = rebuild_params(flat_params, param_spec)
#     return functional_call(model, param_dict, (coords, height, width))
def forward_with_params(param_spec, model, coords, height, width) -> Callable[[torch.Tensor], torch.Tensor]:
    def forward_fun(p):
        param_dict = rebuild_params(p, param_spec)
        return functional_call(model, param_dict, (coords, height, width))
    return forward_fun


def get_image_coords(H, W, device='cuda'):
    """
    Returns (H*W, 2) coordinates sampled at pixel centers in [-1, 1] range.
    """
    y = torch.linspace(0, H - 1, H, device=device) + 0.5
    x = torch.linspace(0, W - 1, W, device=device) + 0.5

    # Normalize to [-1, 1]
    y = 2 * (y / H) - 1
    x = 2 * (x / W) - 1

    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
    coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    coords = coords.view(-1, 2)
    return coords


def matvec_A(u, flat_params, forward_fun, lambda_reg):
    # Compute Ju
    _, Ju = jvp(forward_fun, (flat_params,), (u,))
    
    # Compute JTJu
    _, vjp_fn = vjp(forward_fun, flat_params)
    JTJu = vjp_fn(Ju)[0]

    # Add regularization
    return [jtju_i + lambda_reg * u_i for jtju_i, u_i in zip(JTJu, u)]

def compute_rhs_b(flat_params, forward_fun, flow_pred):
    _, vjp_fn = vjp(forward_fun, flat_params)
    rhs_b = vjp_fn(flow_pred)[0]
    return rhs_b

def cg_solve_tuple(A_fn, b, u_init, max_iter=20, tol=1e-6):
    """
    Solves A(u) = b for u using Conjugate Gradient, where u and b are tuples of tensors.
    
    Args:
        A_fn: function that takes a tuple of tensors u and returns A(u)
        b: tuple of tensors, same shape as u
        u_init: initial guess (tuple of tensors, same shape as b)
        max_iter: max number of CG iterations
        tol: convergence tolerance (based on residual norm)
    
    Returns:
        u: solution tuple of tensors
    """
    def dot(xs, ys):
        return sum(torch.sum(x * y) for x, y in zip(xs, ys))
    
    def add(xs, ys, alpha=1.0):
        return list(x + alpha * y for x, y in zip(xs, ys))
    
    def sub(xs, ys, alpha=1.0):
        return list(x - alpha * y for x, y in zip(xs, ys))

    # Total element count for MSE normalization
    def numel_tuple(xs):
        return sum(x.numel() for x in xs)
    
    u = u_init
    r = sub(b, A_fn(u))  # initial residual: r = b - A u
    p = r
    rs_old = dot(r, r)

    for i in range(max_iter):
        Ap = A_fn(p)
        alpha = rs_old / (dot(p, Ap) + 1e-8)
        u = add(u, p, alpha)
        r = sub(r, Ap, alpha)
        rs_new = dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break

        beta = rs_new / (rs_old + 1e-8)
        p = add(r, p, beta)
        rs_old = rs_new

    # Final MSE
    Au_minus_b = sub(A_fn(u), b)
    mse = (dot(Au_minus_b, Au_minus_b) / numel_tuple(b)).item()

    return u, mse