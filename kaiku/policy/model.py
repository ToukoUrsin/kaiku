"""The policy: a small residual MLP from one observation to the next ``HORIZON`` commands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from kaiku.policy.features import ACT_DIM, HORIZON


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class Net(nn.Module):
    def __init__(self, dim_in: int, width: int = 512, depth: int = 4, dim_out: int = HORIZON * ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, width), *[Block(width) for _ in range(depth)],
                                 nn.LayerNorm(width), nn.Linear(width, dim_out))

    def forward(self, x):
        return self.net(x)


class Policy:
    """A trained net with the scaling of its inputs and outputs."""

    def __init__(self, net: Net, x_mean, x_std, y_mean, y_std, info: dict | None = None):
        self.net = net.eval()
        self.x_mean, self.x_std = np.asarray(x_mean, np.float32), np.asarray(x_std, np.float32)
        self.y_mean, self.y_std = np.asarray(y_mean, np.float32), np.asarray(y_std, np.float32)
        self.info = info or {}

    @torch.no_grad()
    def act(self, x: np.ndarray) -> np.ndarray:
        """(B, dim) observations -> (B, HORIZON, ACT_DIM) commands."""
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)      # a small net: more threads only wait on each other
        z = torch.from_numpy(((x - self.x_mean) / self.x_std).astype(np.float32))
        return self.net(z).numpy().reshape(len(x), HORIZON, ACT_DIM) * self.y_std + self.y_mean

    def save(self, path: Path | str) -> None:
        first = self.net.net[0]
        torch.save({"state": self.net.state_dict(), "dim_in": first.in_features, "width": first.out_features,
                    "depth": sum(isinstance(m, Block) for m in self.net.net),
                    "x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std,
                    "info": self.info}, path)

    @classmethod
    def load(cls, path: Path | str) -> "Policy":
        d = torch.load(path, map_location="cpu", weights_only=False)
        net = Net(d["dim_in"], d["width"], d["depth"])
        net.load_state_dict(d["state"])
        return cls(net, d["x_mean"], d["x_std"], d["y_mean"], d["y_std"], d.get("info"))
