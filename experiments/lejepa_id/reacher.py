"""DMC Reacher rendering and dataset utilities."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from dm_control import suite
from tqdm import tqdm

from .data import ou_augment


def make_env():
    return suite.load(domain_name="reacher", task_name="hard")


def render_at(env, qpos, target, height=64, width=64):
    """Set joint angles and render → (3, H, W) float32 in [0, 1]."""
    env.physics.data.qpos[:2] = qpos
    env.physics.data.qvel[:] = 0
    env.physics.named.model.geom_pos['target', :2] = target
    env.physics.forward()
    rgb = env.physics.render(height=height, width=width, camera_id=0)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def render_batch(env, qpos_batch, target, height=64, width=64):
    """Render batch → (N, 3, H, W)."""
    N = len(qpos_batch)
    imgs = np.empty((N, 3, height, width), dtype=np.float32)
    for i in tqdm(range(N), desc="Rendering"):
        imgs[i] = render_at(env, qpos_batch[i], target, height, width)
    return imgs


def generate_ou_image_pairs(env, N, rho, target, seed=9999):
    """
    Sample OU latent pairs, render both → (img_t, img_tp1, z_t, z_tp1).

    Uses the same OU process as the rest of the repo but renders through MuJoCo.
    """
    rng = np.random.default_rng(seed)
    z_t = rng.standard_normal((N, 2)).astype(np.float32)
    eps = rng.standard_normal((N, 2)).astype(np.float32)
    z_tp1 = rho * z_t + np.sqrt(1 - rho**2) * eps

    print(f"Rendering {2 * N} images (rho={rho})...")
    img_t = render_batch(env, z_t, target)
    img_tp1 = render_batch(env, z_tp1, target)
    return img_t, img_tp1, z_t, z_tp1


def normalize_images(img_t, img_tp1, img_eval=None):
    """Per-channel mean/std normalization. Returns normalized arrays + stats."""
    mean = img_t.mean(axis=(0, 2, 3), keepdims=True)
    std = img_t.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    img_t = (img_t - mean) / std
    img_tp1 = (img_tp1 - mean) / std
    if img_eval is not None:
        img_eval = (img_eval - mean) / std
        return img_t, img_tp1, img_eval, mean, std
    return img_t, img_tp1, mean, std


def solve_ik_grid(env, target, n_grid=200):
    """Find joint angles that place fingertip at target via grid search."""
    best_dist, best_qpos = np.inf, None
    for q0 in np.linspace(-np.pi, np.pi, n_grid):
        for q1 in np.linspace(-np.pi, np.pi, n_grid):
            env.physics.data.qpos[:2] = [q0, q1]
            env.physics.named.model.geom_pos['target', :2] = target
            env.physics.forward()
            tip = env.physics.named.data.geom_xpos['finger'][:2]
            d = np.linalg.norm(tip - target)
            if d < best_dist:
                best_dist = d
                best_qpos = np.array([q0, q1])
    return best_qpos, best_dist


class ReacherOUDataset(torch.utils.data.Dataset):
    """Prerendered OU image pairs with ground-truth latents."""

    def __init__(self, img_t, img_tp1, z_t, z_tp1):
        self.img_t = torch.from_numpy(img_t)
        self.img_tp1 = torch.from_numpy(img_tp1)
        self.z_t = torch.from_numpy(z_t)
        self.z_tp1 = torch.from_numpy(z_tp1)

    def __len__(self):
        return len(self.img_t)

    def __getitem__(self, i):
        return self.img_t[i], self.img_tp1[i], self.z_t[i], self.z_tp1[i]
