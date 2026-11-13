"""Physics-informed neural network model.

The network is a plain MLP mapping (t, x) -> u. The training loop shuffles the
PDE collocation points each epoch, uses Adam with a learning-rate scheduler,
and records the loss histories. An optional random Fourier-feature input layer
can be enabled to lift the network's spectral bias.

Besides fixed loss weights, ``fit`` supports two adaptive weighting schemes:

  * ``gradnorm`` : the learning-rate-annealing scheme of Wang et al. (2021),
    which sets each weight to the ratio of that term's gradient norm to the
    mean gradient norm.
  * ``adaptive`` : inverse-loss balancing, which sets each weight to
    ``sqrt(L_pde / L_term)`` so that no single term dominates the total loss.
"""

import math

import numpy as np
import torch


class _Sin(torch.nn.Module):
    """Sine activation with a fixed frequency, sin(omega * x)."""

    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x):
        return torch.sin(self.omega * x)


class Pinn(torch.nn.Module):
    def __init__(
        self,
        in_size=2,
        hidden_size=50,
        out_size=1,
        num_hidden_layers=4,
        activation="tanh",
        fourier_features=0,
        fourier_sigma=1.0,
        fourier_trainable=False,
        seed=0,
        arch="mlp",
        adaptive_activation=False,
        activation_scale=1.0,
        siren_omega=3.0,
        hard_ic=False,
    ):
        super().__init__()
        self.arch = arch
        self.hard_ic = bool(hard_ic)
        self.adaptive_activation = bool(adaptive_activation)
        self.siren_omega = float(siren_omega)
        if activation == "sin":
            self.act = _Sin(self.siren_omega)
        else:
            act = {
                "tanh": torch.nn.Tanh,
                "relu": torch.nn.ReLU,
                "gelu": torch.nn.GELU,
                "silu": torch.nn.SiLU,
            }[activation]
            self.act = act()

        # Optional random Fourier feature map gamma(v) = [cos(2 pi B v),
        # sin(2 pi B v)] with B ~ N(0, sigma^2) held fixed (Tancik et al.,
        # 2020). This lets the MLP represent high-frequency / sharp features
        # that a plain coordinate input cannot.
        self.fourier_features = int(fourier_features)
        self.fourier_trainable = bool(fourier_trainable)
        if self.fourier_features > 0:
            gen = torch.Generator().manual_seed(seed)
            B = torch.randn(in_size, self.fourier_features, generator=gen)
            if self.fourier_trainable:
                self.fourier_B = torch.nn.Parameter(B * fourier_sigma)
            else:
                self.register_buffer("fourier_B", B * fourier_sigma)
            net_in = 2 * self.fourier_features
        else:
            self.fourier_B = None
            net_in = in_size

        if arch == "modified":
            # Gated "modified MLP" (Wang, Teng & Perdikaris, 2021): two
            # learned input encodings are mixed into every hidden layer by a
            # learned gate.
            self.encoder_u = torch.nn.Linear(net_in, hidden_size)
            self.encoder_v = torch.nn.Linear(net_in, hidden_size)
            self.hidden = torch.nn.ModuleList(
                [torch.nn.Linear(net_in, hidden_size)]
                + [
                    torch.nn.Linear(hidden_size, hidden_size)
                    for _ in range(num_hidden_layers - 1)
                ]
            )
            self.out = torch.nn.Linear(hidden_size, out_size)
        elif arch == "mlp":
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(net_in, hidden_size)]
                + [
                    torch.nn.Linear(hidden_size, hidden_size)
                    for _ in range(num_hidden_layers)
                ]
                + [torch.nn.Linear(hidden_size, out_size)]
            )
        else:
            raise ValueError(f"unknown arch {arch!r}")

        if self.adaptive_activation:
            n_act = num_hidden_layers + (2 if arch == "modified" else 1)
            self.act_scale = torch.nn.Parameter(
                torch.full((n_act,), float(activation_scale))
            )

        if activation == "sin":
            self._siren_init()

        self.loss_pde_history = []
        self.loss_ic_history = []
        self.loss_bc_history = []
        self.loss_total_history = []
        self.weight_history = []

    def _siren_init(self, first_bound=None):
        """SIREN-style initialization (Sitzmann et al., 2020)."""
        for i, lin in enumerate(
            m for m in self.modules() if isinstance(m, torch.nn.Linear)
        ):
            n = lin.weight.shape[1]
            with torch.no_grad():
                if i == 0:
                    bound = 1.0 / n if first_bound is None else first_bound
                else:
                    bound = math.sqrt(6.0 / n) / self.siren_omega
                lin.weight.uniform_(-bound, bound)
                if lin.bias is not None:
                    lin.bias.zero_()

    def _apply_act(self, z, idx):
        if self.adaptive_activation:
            z = z * self.act_scale[idx]
        return self.act(z)

    def raw(self, tx):
        """Network output before any hard initial-condition constraint."""
        if self.fourier_B is not None:
            proj = 2.0 * math.pi * (tx @ self.fourier_B)
            tx = torch.cat((torch.cos(proj), torch.sin(proj)), dim=-1)

        if self.arch == "modified":
            u = self._apply_act(self.encoder_u(tx), 0)
            v = self._apply_act(self.encoder_v(tx), 1)
            h = tx
            for i, lin in enumerate(self.hidden):
                z = self._apply_act(lin(h), i + 2)
                h = (1.0 - z) * u + z * v
            return self.out(h)

        h = tx
        for i, lin in enumerate(self.linears[:-1]):
            h = self._apply_act(lin(h), i)
        return self.linears[-1](h)

    def forward(self, tx):
        out = self.raw(tx)
        if not self.hard_ic:
            return out
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        u0 = torch.where(
            x < 0.5 * math.pi,
            torch.full_like(x, 0.5),
            torch.full_like(x, 1.5),
        )
        return torch.where(t <= 0.0, u0, out)

    def plot_loss_history(self, save_path=None):
        import matplotlib.pyplot as plt

        plt.plot(self.loss_pde_history, label="Loss PDE")
        plt.plot(self.loss_ic_history, label="Loss IC")
        plt.plot(self.loss_bc_history, label="Loss BC")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def fit(
        self,
        num_epochs,
        loss_fn,
        pde_data,
        batch_size,
        optimizer,
        scheduler=None,
        loss_weights=None,
        log_every=100,
        callback=None,
        verbose=True,
        weighting="fixed",
        weight_update_every=100,
        weight_alpha=0.1,
        **loss_kwargs,
    ):
        """Train the network.

        Named `fit` (not `train`) to avoid shadowing nn.Module.train, which
        would break `.eval()` and `.train(mode)`.

        Parameters
        ----------
        loss_fn : callable(model, pde_batch, **loss_kwargs) -> dict of losses
        pde_data : (N_PDE, 2) collocation points [t, x]
        loss_weights : dict mapping loss names to weights
        callback : optional callable(epoch, self) invoked every log_every
        weighting : "fixed", "gradnorm", or "adaptive" loss weighting
        weight_update_every : epochs between adaptive weight updates
        weight_alpha : exponential moving-average rate for adaptive weights
        **loss_kwargs : extra tensors (ic_data, bc_left_data, ...) forwarded
            to loss_fn
        """
        if loss_weights is None:
            loss_weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
        loss_weights = dict(loss_weights)
        if weighting not in ("fixed", "gradnorm", "adaptive"):
            raise ValueError(f"unknown weighting scheme {weighting!r}")

        n_pde = pde_data.shape[0]
        iterator = range(num_epochs)
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(iterator)

        for epoch in iterator:
            pde_losses, ic_losses, bc_losses = [], [], []

            perm = torch.randperm(n_pde, device=pde_data.device)
            for start in range(0, n_pde, batch_size):
                X = pde_data[perm[start : start + batch_size]]
                optimizer.zero_grad(set_to_none=True)

                losses = loss_fn(self, X, **loss_kwargs)
                total_loss = sum(loss_weights[k] * losses[k] for k in loss_weights)
                total_loss.backward()
                optimizer.step()

                pde_losses.append(losses["pde"].item())
                ic_losses.append(losses["ic"].item())
                bc_losses.append(losses["bc"].item())

            if scheduler is not None:
                scheduler.step()

            means = {
                "pde": float(np.mean(pde_losses)),
                "ic": float(np.mean(ic_losses)),
                "bc": float(np.mean(bc_losses)),
            }
            self.loss_pde_history.append(means["pde"])
            self.loss_ic_history.append(means["ic"])
            self.loss_bc_history.append(means["bc"])
            self.loss_total_history.append(
                sum(loss_weights[k] * means[k] for k in loss_weights)
            )

            if (
                weighting != "fixed"
                and weight_update_every > 0
                and (epoch + 1) % weight_update_every == 0
            ):
                self._update_weights(
                    weighting,
                    loss_weights,
                    means,
                    loss_fn,
                    pde_data,
                    batch_size,
                    weight_alpha,
                    loss_kwargs,
                )
            self.weight_history.append(dict(loss_weights))

            if callback is not None and (epoch % log_every == 0):
                callback(epoch, self)

        return self

    def _update_weights(
        self,
        mode,
        loss_weights,
        means,
        loss_fn,
        pde_data,
        batch_size,
        alpha,
        loss_kwargs,
    ):
        """Update ``loss_weights`` in place according to the chosen scheme."""
        keys = list(loss_weights)
        if mode == "adaptive":
            ref = max(means["pde"], 1e-30)
            target = {
                k: 1.0 if k == "pde" else math.sqrt(ref / max(means[k], 1e-30))
                for k in keys
            }
        else:  # gradnorm
            n = pde_data.shape[0]
            idx = torch.randperm(n, device=pde_data.device)[:batch_size]
            losses = loss_fn(self, pde_data[idx], **loss_kwargs)
            params = list(self.parameters())
            norms = {}
            for i, k in enumerate(keys):
                grads = torch.autograd.grad(
                    losses[k],
                    params,
                    retain_graph=(i < len(keys) - 1),
                    allow_unused=True,
                )
                norms[k] = max(
                    (g.detach().abs().max().item() for g in grads if g is not None),
                    default=0.0,
                )
            avg = sum(norms.values()) / len(norms)
            target = {k: (norms[k] / avg if avg > 0 else 1.0) for k in keys}

        for k in keys:
            loss_weights[k] = (1.0 - alpha) * loss_weights[k] + alpha * target[k]
