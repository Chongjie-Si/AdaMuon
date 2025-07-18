import os
import torch
import torch.distributed as dist
from torch import Tensor

## Original Muon

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:

    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class AdaMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, beta2=0.999, eps=1e-8, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, beta2=beta2, eps=eps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            buf = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=buf, 
                update_buffer_views=[buf[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * 0.2 * max(p_world.size(-2), p_world.size(-1))**0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None

                    state = self.state[p]

                    # First-order Momentum
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                    if g.ndim == 4:
                        g = g.view(len(g), -1)
                    
                    # NS Approximation
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()

                    # Second-order Momentum
                    if "v_buffer" not in state:
                        state["v_buffer"] = torch.zeros_like(g)
                        state["step"] = 0
                    v = state["v_buffer"]
                    state["step"] += 1
                    beta2 = group["beta2"]
                    eps = group["eps"]

                    # v_t = β2 * v_{t-1} + (1 - β2) * g^2
                    v.mul_(beta2).addcmul_(1 - beta2, g, g)
                    bias_correction2 = 1 - beta2 ** state["step"]
                    v_hat = v.div(bias_correction2)
                    g = g.div(v_hat.view_as(g).sqrt().add(eps))

                    # RMS-aligned Rescaling
                    scale = min(p.shape)**0.5 / (g.norm() + eps)
                    g.mul_(scale)

                    g = g.to(update_buffer.dtype)

                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
