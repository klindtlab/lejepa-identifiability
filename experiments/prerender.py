"""
Pre-render all Reacher datasets to disk.

Three modes:
  eval  — 10k Gaussian samples, rendered once, shared by all runs
  ou    — 100k OU pairs for a given rho
  traj  — 100k pairs subsampled from LeWM trajectories at a given delta

Usage:
    python prerender.py eval
    python prerender.py ou   --rho 0.95
    python prerender.py traj --delta 16 --h5_path data/reacher.h5

Saves images as uint8 (3, 64, 64) to keep disk usage ~1.2 GB per 100k images.
Normalization stats computed and saved; applied at training time.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, shapiro, skew, kurtosis
from tqdm import tqdm
from dm_control import suite


# ═════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═════════════════════════════════════════════════════════════════════════════

TARGET = np.array([0.1, 0.1])
IMG_SIZE = 64


def make_env():
    return suite.load(domain_name="reacher", task_name="hard")


def render_at(env, qpos, height=IMG_SIZE, width=IMG_SIZE):
    """Render → (3, H, W) uint8."""
    env.physics.data.qpos[:2] = qpos
    env.physics.data.qvel[:] = 0
    env.physics.named.model.geom_pos['target', :2] = TARGET
    env.physics.forward()
    rgb = env.physics.render(height=height, width=width, camera_id=0)
    return rgb.transpose(2, 0, 1)  # uint8, (3, H, W)


def render_batch(env, qpos_batch):
    """Render → (N, 3, H, W) uint8."""
    N = len(qpos_batch)
    imgs = np.empty((N, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in tqdm(range(N), desc="Rendering"):
        imgs[i] = render_at(env, qpos_batch[i])
    return imgs


def compute_norm_stats(imgs_uint8):
    """Compute per-channel mean/std from uint8 images. Returns float32 arrays."""
    imgs = imgs_uint8.astype(np.float32) / 255.0
    mean = imgs.mean(axis=(0, 2, 3))   # (3,)
    std = imgs.std(axis=(0, 2, 3))     # (3,)
    return mean.astype(np.float32), std.astype(np.float32)


def save_dataset(out_dir, z_t, z_tp1, img_t, img_tp1, meta):
    """Save arrays + metadata to directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_t.npy", z_t)
    np.save(out_dir / "z_tp1.npy", z_tp1)
    np.save(out_dir / "img_t.npy", img_t)
    np.save(out_dir / "img_tp1.npy", img_tp1)

    # Norm stats from img_t
    mean, std = compute_norm_stats(img_t)
    np.save(out_dir / "img_mean.npy", mean)
    np.save(out_dir / "img_std.npy", std)

    meta["img_mean"] = mean.tolist()
    meta["img_std"] = std.tolist()
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_gb = sum(
        (out_dir / fn).stat().st_size for fn in
        ["img_t.npy", "img_tp1.npy", "z_t.npy", "z_tp1.npy"]
    ) / 1e9
    print(f"  Saved to {out_dir} ({size_gb:.2f} GB)")


# ═════════════════════════════════════════════════════════════════════════════
# EVAL
# ═════════════════════════════════════════════════════════════════════════════

def prerender_eval(args):
    """10k i.i.d. Gaussian samples + rendered images."""
    out_dir = Path(args.data_root) / "eval"
    if (out_dir / "img.npy").exists() and not args.force:
        print(f"Eval data already exists at {out_dir}, skipping (use --force)")
        return

    rng = np.random.default_rng(args.eval_seed)
    z = rng.standard_normal((args.n_eval, 2)).astype(np.float32)

    env = make_env()
    print(f"Rendering {args.n_eval} eval images...")
    imgs = render_batch(env, z)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z.npy", z)
    np.save(out_dir / "img.npy", imgs)

    mean, std = compute_norm_stats(imgs)
    np.save(out_dir / "img_mean.npy", mean)
    np.save(out_dir / "img_std.npy", std)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"n_eval": args.n_eval, "seed": args.eval_seed,
                    "img_mean": mean.tolist(), "img_std": std.tolist()}, f, indent=2)
    print(f"  Saved to {out_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# OU
# ═════════════════════════════════════════════════════════════════════════════

def prerender_ou(args):
    """100k OU pairs for a given rho."""
    rho = args.rho
    out_dir = Path(args.data_root) / "ou" / f"rho={rho:.2f}"
    if (out_dir / "img_t.npy").exists() and not args.force:
        print(f"OU data for rho={rho} already exists, skipping (use --force)")
        return

    N = args.n_train
    rng = np.random.default_rng(args.render_seed)
    z_t = rng.standard_normal((N, 2)).astype(np.float32)
    eps = rng.standard_normal((N, 2)).astype(np.float32)
    z_tp1 = rho * z_t + np.sqrt(1 - rho**2) * eps

    env = make_env()
    print(f"OU rho={rho}: rendering {2 * N} images...")
    img_t = render_batch(env, z_t)
    img_tp1 = render_batch(env, z_tp1)

    meta = {"type": "ou", "rho": rho, "n": N, "seed": args.render_seed}
    save_dataset(out_dir, z_t, z_tp1, img_t, img_tp1, meta)


# ═════════════════════════════════════════════════════════════════════════════
# TRAJECTORY
# ═════════════════════════════════════════════════════════════════════════════

def load_episodes(h5_path):
    """Load qpos grouped by episode → (n_episodes, T, 2)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        qpos = np.array(f["qpos"])
        ep_len = np.array(f["ep_len"])
    T = ep_len[0]
    assert (ep_len == T).all(), f"Non-uniform episode lengths"
    episodes = qpos.reshape(len(ep_len), T, 2)
    print(f"Loaded {len(episodes)} episodes, {T} steps each")
    return episodes


def subsample_pairs(episodes, delta, n_per_episode, seed):
    """Sample n_per_episode (t, t+delta) pairs from each episode."""
    rng = np.random.default_rng(seed)
    n_ep, T, d = episodes.shape
    max_start = T - delta
    z_t_list, z_tp1_list = [], []
    for ep in episodes:
        starts = rng.choice(max_start, size=n_per_episode, replace=False)
        z_t_list.append(ep[starts])
        z_tp1_list.append(ep[starts + delta])
    return (np.concatenate(z_t_list).astype(np.float32),
            np.concatenate(z_tp1_list).astype(np.float32))


def traj_diagnostics(episodes, delta):
    """Compute autocorrelation + normality stats."""
    n_ep, T, d = episodes.shape
    ms = T - delta
    z_t = episodes[:, :ms].reshape(-1, d)
    z_tp1 = episodes[:, delta:delta+ms].reshape(-1, d)

    diag = {"delta": delta}
    for i, name in enumerate(["shoulder", "wrist"]):
        r, _ = pearsonr(z_t[:, i], z_tp1[:, i])
        diag[f"rho_{name}"] = float(r)
        diag[f"skew_{name}"] = float(skew(z_t[:, i]))
        diag[f"kurtosis_{name}"] = float(kurtosis(z_t[:, i]))
        sub = z_t[np.random.choice(len(z_t), 5000, replace=False), i]
        _, p = shapiro(sub)
        diag[f"shapiro_p_{name}"] = float(p)
    diag["rho_mean"] = (diag["rho_shoulder"] + diag["rho_wrist"]) / 2
    return diag


def prerender_traj(args):
    """100k pairs subsampled from LeWM trajectories at a given delta."""
    delta = args.delta
    out_dir = Path(args.data_root) / "traj" / f"delta={delta}"
    if (out_dir / "img_t.npy").exists() and not args.force:
        print(f"Traj data for delta={delta} already exists, skipping")
        return

    episodes = load_episodes(args.h5_path)
    n_episodes = len(episodes)
    n_per_episode = args.n_train // n_episodes
    N_actual = n_per_episode * n_episodes
    print(f"delta={delta}: {n_per_episode} pairs/episode × {n_episodes} = {N_actual}")

    # Diagnostics
    diag = traj_diagnostics(episodes, delta)
    print(f"  rho: shoulder={diag['rho_shoulder']:.4f}, "
          f"wrist={diag['rho_wrist']:.4f}")
    print(f"  skew: {diag['skew_shoulder']:.3f}, {diag['skew_wrist']:.3f}")

    # Subsample
    z_t, z_tp1 = subsample_pairs(episodes, delta, n_per_episode, args.render_seed)

    # Render
    env = make_env()
    print(f"  Rendering {2 * len(z_t)} images...")
    img_t = render_batch(env, z_t)
    img_tp1 = render_batch(env, z_tp1)

    meta = {"type": "traj", "delta": delta, "n": len(z_t),
            "n_per_episode": n_per_episode, "seed": args.render_seed,
            **diag}
    save_dataset(out_dir, z_t, z_tp1, img_t, img_tp1, meta)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    # Shared
    for name in ["eval", "ou", "traj"]:
        sp = sub.add_parser(name)
        sp.add_argument("--data_root", type=str, default="data/reacher")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--render_seed", type=int, default=9999)

    # eval
    sub.choices["eval"].add_argument("--n_eval", type=int, default=10000)
    sub.choices["eval"].add_argument("--eval_seed", type=int, default=8888)

    # ou
    sub.choices["ou"].add_argument("--rho", type=float, required=True)
    sub.choices["ou"].add_argument("--n_train", type=int, default=100000)

    # traj
    sub.choices["traj"].add_argument("--delta", type=int, required=True)
    sub.choices["traj"].add_argument("--h5_path", type=str, required=True)
    sub.choices["traj"].add_argument("--n_train", type=int, default=100000)

    args = p.parse_args()

    if args.mode == "eval":
        prerender_eval(args)
    elif args.mode == "ou":
        prerender_ou(args)
    elif args.mode == "traj":
        prerender_traj(args)


if __name__ == "__main__":
    main()
