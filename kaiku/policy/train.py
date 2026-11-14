"""Behaviour cloning on the recorded demonstrations.

    python -m kaiku.policy.train DATA --out policy.pt [--humans K] [--plain] [--steps 20000]

``--humans K`` uses the operator's first K episodes; ``--plain`` leaves their echoes out.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from kaiku.policy.dataset import Loader, Pairs
from kaiku.policy.features import Featurizer
from kaiku.policy.model import Net, Policy


def jitter(x: torch.Tensor, y: torch.Tensor, sl: dict, sigma: float, sigma_asked: float
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two things the demonstrations never show, added to each batch.

    A hand a little away from where it was: what it was told stays the same, so the policy
    learns that its path does not hang on a few millimetres of tracking error.
    A last command a little off: the next ones still go where the demonstration's went, so the
    policy learns to come back to the path instead of carrying its own error along."""
    B = len(x)
    d = sigma * torch.randn(B, 6, device=x.device)
    e = sigma_asked * torch.randn(B, 6, device=x.device)
    x = x.clone()
    has = x[:, sl["has"]]
    x[:, sl["pinch"]] += d
    x[:, sl["to_part"]] -= d * has
    x[:, sl["to_bin"]] -= d * has
    x[:, sl["lead"]] += e - d
    y = y.clone()
    y[:, :, :6] -= e[:, None]
    return x, y


def train(data: Pairs, feat: Featurizer, steps: int = 20000, batch: int = 512, lr: float = 1e-3, width: int = 512,
          depth: int = 4, sigma: float = 0.004, sigma_asked: float = 0.002, seed: int = 0, threads: int = 8,
          log=print, every: int = 2000, on_step=None, device: str | None = None) -> Policy:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x, y = torch.from_numpy(data.x).to(dev), torch.from_numpy(data.y).to(dev)
    M = len(x)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-3)
    # One scale per kind of output, so that a millimetre of hand position weighs the same on every axis.
    y_mean = y.mean((0, 1))
    y_std = torch.cat([y[..., :6].std().expand(6), y[..., 6:18].std().expand(12), y[..., 18:].std().expand(14)])
    y_std = y_std.clamp_min(1e-3)
    net = Net(x.shape[1], width, depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator(device=dev).manual_seed(seed)
    t0, run = time.time(), 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(M, (batch,), generator=g, device=dev)
        xb, yb = jitter(x[idx], y[idx], feat.slices, sigma, sigma_asked) if sigma else (x[idx], y[idx])
        pred = net((xb - x_mean) / x_std).view(batch, y.shape[1], y.shape[2])
        loss = torch.nn.functional.smooth_l1_loss(pred, (yb - y_mean) / y_std, beta=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1 or step == steps:
            run = loss.item() if step == 1 else 0.9 * run + 0.1 * loss.item()
            if on_step:
                on_step(step, steps, run)
        if log and (step % every == 0 or step == steps):
            log(f"  step {step:6d}/{steps}  loss {run:.4f}  {time.time() - t0:5.0f} s")
    cpu = lambda t: t.detach().cpu().numpy()                      # noqa: E731
    return Policy(net.cpu(), cpu(x_mean), cpu(x_std), cpu(y_mean), cpu(y_std),
                  {"pairs": M, "steps": steps, "loss": round(run, 5), "seconds": round(time.time() - t0, 1)})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="recorder folder")
    ap.add_argument("--out", default="policy.pt")
    ap.add_argument("--humans", type=int, default=None, help="use the operator's first K episodes")
    ap.add_argument("--plain", action="store_true", help="the operator's own demonstrations only, no echoes")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args(argv)
    loader = Loader()
    data = loader.load(a.data, a.humans, a.stride, log=print)
    if a.plain:
        data = data.pick(np.unique(data.episode), echoes=False)
    print(f"{len(data.x)} pairs from {len(np.unique(data.episode))} operator episodes"
          f" ({'no echoes' if a.plain else 'with echoes'})")
    policy = train(data, loader.feat, a.steps, a.batch, width=a.width, depth=a.depth, seed=a.seed, threads=a.threads)
    policy.info.update(humans=int(len(np.unique(data.episode))), plain=bool(a.plain), data=str(Path(a.data)))
    policy.save(a.out)
    print(f"-> {a.out}  {policy.info}")


if __name__ == "__main__":
    main()
