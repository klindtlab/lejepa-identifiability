"""
Paper figures for Reacher experiment.

Produces four figures:
    1. reacher_annotated.png      — schematic of the two latent angles
    2. planning_demo.png          — 3-row image grid: true / OU retrieval /
                                    traj retrieval, with true-frame ghost
                                    overlay on the two model rows
    3. planning_scatter.png       — 3x3 scatter: embeddings, straight-in-true
                                    trajectories, straight-in-model trajectories
    4. planning_quantitative.png  — boxplots: path length (log y) and
                                    control effort over K random (start, goal)
                                    pairs, with kNN decoder θ̂ = f^{-1}(ẑ).

Usage (needs GPU + MuJoCo + rendered gallery with z.npy):
    python -m analysis.make_reacher_figures \
        --results_dir results/reacher \
        --data_root   data/reacher \
        --out_dir     figures/reacher
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import argparse
import colorsys
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from sklearn.neighbors import KNeighborsRegressor

from lejepa_id.reacher import make_env, render_at, solve_ik_grid
from lejepa_id.models import make_cnn_encoder
from run_reacher import normalize_uint8


TARGET = np.array([0.1, 0.1])


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def find_best_checkpoint(results_dir, condition):
    """Highest-R² checkpointed run for 'ou' or 'traj'."""
    best_r2, best_dir, best_res = -np.inf, None, None
    for p in Path(results_dir).rglob("result.json"):
        if not (p.parent / "checkpoint.pt").exists():
            continue
        with open(p) as f:
            r = json.load(f)
        is_traj = "delta" in r and r.get("rho") is None
        if condition == "ou" and is_traj:
            continue
        if condition == "traj" and not is_traj:
            continue
        if r.get("r2_hz", -1) > best_r2:
            best_r2, best_dir, best_res = r["r2_hz"], p.parent, r
    if best_res is None:
        raise RuntimeError(f"No {condition} results with checkpoint in {results_dir}")
    print(f"Best {condition}: R²={best_r2:.4f}  run={best_res['run_name']}")
    return best_dir, best_res


def load_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = make_cnn_encoder(d_latent=ckpt["d_latent"], device=device)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    return enc, ckpt["train_mean"], ckpt["train_std"], ckpt["d_latent"]


@torch.no_grad()
def encode_batched(enc, imgs_norm, device, bs=512):
    outs = []
    for i in range(0, len(imgs_norm), bs):
        outs.append(enc(imgs_norm[i:i + bs].to(device)).cpu())
    return torch.cat(outs).numpy()


def encode_images(enc, imgs, mn, sd, device):
    """Encode a small list of (3,H,W) float images."""
    mn_b, sd_b = mn[:, None, None], sd[:, None, None]
    arr = np.stack([(im - mn_b) / (sd_b + 1e-6) for im in imgs]).astype(np.float32)
    with torch.no_grad():
        return enc(torch.from_numpy(arr).to(device)).cpu().numpy()


def project_to_2d(z_gallery, z_points):
    """d=2: identity. Else PCA fit on gallery, applied to both."""
    if z_gallery.shape[1] == 2:
        return z_gallery, z_points, None
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(z_gallery)
    return pca.transform(z_gallery), pca.transform(z_points), pca


def try_load_true_angles(eval_dir, gallery_size):
    for fname in ("z.npy", "angles.npy", "qpos.npy"):
        p = os.path.join(eval_dir, fname)
        if os.path.exists(p):
            arr = np.load(p)[:gallery_size]
            print(f"Loaded true angles from {fname}  shape={arr.shape}")
            return arr
    return None


def make_colors(z):
    """Polar color map: hue = angle, lightness = radius."""
    if hasattr(z, "cpu"):
        z = z.cpu().numpy()
    x, y = z[:, 0], z[:, 1]
    angles = np.arctan2(y, x)
    radii = np.sqrt(x ** 2 + y ** 2)
    hue = (angles + np.pi) / (2 * np.pi)
    lightness = 0.3 + 0.4 * (radii / (radii.max() + 1e-8))
    saturation = np.full_like(hue, 0.85)
    return np.array([colorsys.hls_to_rgb(h, l, s)
                     for h, l, s in zip(hue, lightness, saturation)])


def square_extent(*arrays, pad=0.08):
    """Shared square xlim/ylim covering all input (N, 2) arrays."""
    pts = np.vstack([a for a in arrays if a is not None and len(a) > 0])
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin) * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def show_img(ax, img):
    if img.ndim == 3 and img.shape[0] == 3:
        ax.imshow(img.transpose(1, 2, 0))
    else:
        ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def border(ax, color, width=4):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(width); sp.set_visible(True)


# ═════════════════════════════════════════════════════════════════════════════
# Shared context (loaded once, passed to every figure function)
# ═════════════════════════════════════════════════════════════════════════════

def load_planning_context(results_dir, data_root, device, gallery_size):
    """Load both encoders, the gallery, and per-encoder gallery embeddings."""
    gallery_u8 = np.load(os.path.join(data_root, "eval", "img.npy"))[:gallery_size]
    gallery_angles = try_load_true_angles(os.path.join(data_root, "eval"),
                                          gallery_size)
    if gallery_angles is None:
        raise RuntimeError("Need eval angles (z.npy / angles.npy / qpos.npy)")

    run_ou,   res_ou   = find_best_checkpoint(results_dir, "ou")
    run_traj, res_traj = find_best_checkpoint(results_dir, "traj")
    enc_ou,   mn_ou,   sd_ou,   _ = load_encoder(run_ou   / "checkpoint.pt", device)
    enc_traj, mn_traj, sd_traj, _ = load_encoder(run_traj / "checkpoint.pt", device)

    gnorm_ou   = torch.from_numpy(normalize_uint8(gallery_u8, mn_ou,   sd_ou))
    gnorm_traj = torch.from_numpy(normalize_uint8(gallery_u8, mn_traj, sd_traj))
    gallery_z_ou   = encode_batched(enc_ou,   gnorm_ou,   device)
    gallery_z_traj = encode_batched(enc_traj, gnorm_traj, device)
    gallery_2d_ou,   _, _ = project_to_2d(gallery_z_ou,   gallery_z_ou)
    gallery_2d_traj, _, _ = project_to_2d(gallery_z_traj, gallery_z_traj)

    return {
        "device": device,
        "gallery_u8": gallery_u8,
        "gallery_display": [im.astype(np.float32) / 255.0 for im in gallery_u8],
        "gallery_angles": gallery_angles,
        "gallery_colors": make_colors(gallery_angles),
        "enc_ou":   enc_ou,   "mn_ou":   mn_ou,   "sd_ou":   sd_ou,
        "enc_traj": enc_traj, "mn_traj": mn_traj, "sd_traj": sd_traj,
        "gallery_z_ou":   gallery_z_ou,
        "gallery_z_traj": gallery_z_traj,
        "gallery_2d_ou":   gallery_2d_ou,
        "gallery_2d_traj": gallery_2d_traj,
        "result_ou":   res_ou,
        "result_traj": res_traj,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: annotated Reacher frame
# ═════════════════════════════════════════════════════════════════════════════

def make_annotated_frame(save_path, img_size=256):
    env = make_env()
    qpos = np.array([-np.pi / 2, -np.pi / 2])
    img = render_at(env, qpos, TARGET, height=img_size, width=img_size)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.transpose(1, 2, 0))

    sh = (128, 128)
    el = (128, 178)

    arc1 = patches.Arc((sh[0] + 4, sh[1]), 46, 46, angle=0,
                       theta1=0, theta2=90, color="#22cc22", linewidth=3)
    ax.add_patch(arc1)
    ax.annotate(r"$z_0$", xy=(sh[0] + 32, sh[1] + 32),
                fontsize=20, fontweight="bold", color="#22cc22")

    arc2 = patches.Arc((el[0] - 4, el[1] - 4), 46, 46, angle=0,
                       theta1=180, theta2=270, color="#ff8800", linewidth=3)
    ax.add_patch(arc2)
    ax.annotate(r"$z_1$", xy=(el[0] - 40, el[1] - 30),
                fontsize=20, fontweight="bold", color="#ff8800")

    ax.plot(*sh, "o", color="#22cc22", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(*el, "o", color="#ff8800", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: planning_demo.png — 3-row image grid with ghost overlay
# ═════════════════════════════════════════════════════════════════════════════

def _straight_latent_plan(z_endpoints, n_steps):
    alphas = np.linspace(0, 1, n_steps)
    return np.stack([(1 - a) * z_endpoints[0] + a * z_endpoints[1]
                     for a in alphas])


def _nn_retrieve(plan_z, gallery_z, gallery_display):
    out = []
    for pz in plan_z:
        idx = int(np.linalg.norm(gallery_z - pz, axis=1).argmin())
        out.append(gallery_display[idx])
    return out


def make_planning_figure(env, ctx, save_path, n_steps=8,
                         qpos_start=None, ghost_alpha=0.3):
    """True, OU retrieval, traj retrieval. Rows 2-3 blend each retrieved frame
    with the corresponding true frame at weight `ghost_alpha`."""
    qpos_goal, _ = solve_ik_grid(env, TARGET)
    if qpos_start is None:
        qpos_start = np.array([-3 / 4 * np.pi, 1 / 4 * np.pi])

    alphas = np.linspace(0, 1, n_steps)
    qpos_traj = np.array([(1 - a) * qpos_start + a * qpos_goal for a in alphas])
    true_imgs = [render_at(env, q, TARGET) for q in qpos_traj]

    def retrieval_row(enc, mn, sd, gallery_z):
        z_ends = encode_images(enc, [true_imgs[0], true_imgs[-1]],
                               mn, sd, ctx["device"])
        plan_z = _straight_latent_plan(z_ends, n_steps)
        retrieved = _nn_retrieve(plan_z, gallery_z, ctx["gallery_display"])
        # Pin endpoints so Start/Goal columns are identical across rows.
        retrieved[0]  = true_imgs[0]
        retrieved[-1] = true_imgs[-1]
        return retrieved

    ou_imgs   = retrieval_row(ctx["enc_ou"],   ctx["mn_ou"],   ctx["sd_ou"],
                              ctx["gallery_z_ou"])
    traj_imgs = retrieval_row(ctx["enc_traj"], ctx["mn_traj"], ctx["sd_traj"],
                              ctx["gallery_z_traj"])

    def blend(retrieved, alpha=ghost_alpha):
        return [np.clip(alpha * t + (1 - alpha) * r, 0.0, 1.0)
                for r, t in zip(retrieved, true_imgs)]

    rows = [
        ("True\ntrajectory",                                                  true_imgs),
        (f"Gaussian\n(R²={ctx['result_ou']['r2_hz']:.2f})",                   blend(ou_imgs)),
        (f"Trajectory\n(R²={ctx['result_traj']['r2_hz']:.2f})",               blend(traj_imgs)),
    ]

    fig, axes = plt.subplots(3, n_steps, figsize=(2.0 * n_steps, 6.0))
    for r, (label, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            show_img(axes[r, c], im)
        axes[r, 0].text(-0.25, 0.5, label,
                        transform=axes[r, 0].transAxes,
                        fontsize=12, fontweight="bold",
                        ha="right", va="center")
        border(axes[r, 0],  "#22cc22")
        border(axes[r, -1], "#dd2222")

    axes[0, 0].set_title("Start", color="#22cc22", fontsize=13, fontweight="bold")
    axes[0, -1].set_title("Goal", color="#dd2222", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: planning_scatter.png — 3x3 scatter grid
# ═════════════════════════════════════════════════════════════════════════════

TRAJ_COLORS = ["#1a1a1a", "#0072b2", "#cc79a7"]
SPACES = ["true", "ou", "traj"]
SPACE_TITLES = ["True (θ-space)", "Gaussian latent", "Trajectory latent"]


def make_scatter_figure(env, ctx, save_path, n_steps=8):
    """Three rows:
       0. Gallery embedding in each space (polar-colored).
       1. Three straight-in-θ trajectories, as they appear in each space.
       2. Straight-in-OU and straight-in-traj plans (decoded via kNN for the
          True panel), as they appear in each space.
    """
    qpos_goal, _ = solve_ik_grid(env, TARGET)
    qpos_starts = [
        np.array([-3 / 4 * np.pi,  1 / 4 * np.pi]),
        np.array([ 1 / 4 * np.pi,  1 / 2 * np.pi]),
        np.array([-1 / 2 * np.pi, -3 / 4 * np.pi]),
    ]
    alphas = np.linspace(0, 1, n_steps)

    # Row 1 data: straight θ-line → encoded in each model.
    multi_trajs = []
    for qs in qpos_starts:
        qpos_path = np.array([(1 - a) * qs + a * qpos_goal for a in alphas])
        imgs = [render_at(env, q, TARGET) for q in qpos_path]
        z_ou   = encode_images(ctx["enc_ou"],   imgs,
                                ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
        z_traj = encode_images(ctx["enc_traj"], imgs,
                                ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
        multi_trajs.append({"theta": qpos_path, "z_ou": z_ou, "z_traj": z_traj})

    # Row 2 data: plan straight in each model, decode to θ via kNN.
    dec_ou   = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
        ctx["gallery_z_ou"],   ctx["gallery_angles"])
    dec_traj = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
        ctx["gallery_z_traj"], ctx["gallery_angles"])

    multi_modelplan = []
    for traj in multi_trajs:
        plan_ou_z   = _straight_latent_plan(
            np.array([traj["z_ou"][0],   traj["z_ou"][-1]]),   n_steps)
        plan_traj_z = _straight_latent_plan(
            np.array([traj["z_traj"][0], traj["z_traj"][-1]]), n_steps)

        theta_from_ou   = dec_ou.predict(plan_ou_z)
        theta_from_traj = dec_traj.predict(plan_traj_z)
        theta_from_ou[0],   theta_from_ou[-1]   = traj["theta"][0], traj["theta"][-1]
        theta_from_traj[0], theta_from_traj[-1] = traj["theta"][0], traj["theta"][-1]

        # To display the OU plan in the traj panel (and vice versa), re-render
        # the decoded θ and re-encode.
        imgs_from_ou   = [render_at(env, q, TARGET) for q in theta_from_ou]
        imgs_from_traj = [render_at(env, q, TARGET) for q in theta_from_traj]
        z_traj_from_ou   = encode_images(ctx["enc_traj"], imgs_from_ou,
                                          ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
        z_ou_from_traj   = encode_images(ctx["enc_ou"],   imgs_from_traj,
                                          ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])

        multi_modelplan.append({
            "true_from_ou":   theta_from_ou,
            "true_from_traj": theta_from_traj,
            "ou_from_ou":     plan_ou_z,         # literally straight in OU
            "traj_from_traj": plan_traj_z,       # literally straight in traj
            "traj_from_ou":   z_traj_from_ou,
            "ou_from_traj":   z_ou_from_traj,
        })

    # Accessors
    def gallery(space):
        return {"true": ctx["gallery_angles"],
                "ou":   ctx["gallery_2d_ou"],
                "traj": ctx["gallery_2d_traj"]}[space]

    def row1_coords(space, traj):
        return {"true": traj["theta"],
                "ou":   traj["z_ou"],
                "traj": traj["z_traj"]}[space]

    def row2_coords(space, mp):
        """Return (solid, dashed) = (planned-in-OU, planned-in-traj), in `space`."""
        if space == "true":
            return mp["true_from_ou"], mp["true_from_traj"]
        if space == "ou":
            return mp["ou_from_ou"],   mp["ou_from_traj"]
        if space == "traj":
            return mp["traj_from_ou"], mp["traj_from_traj"]

    # Per-column extent (shared across all 3 rows of that column)
    col_extents = []
    for space in SPACES:
        g = gallery(space)
        row1 = [row1_coords(space, t) for t in multi_trajs]
        row2_flat = [p for mp in multi_modelplan
                       for p in row2_coords(space, mp)]
        col_extents.append(square_extent(g, *row1, *row2_flat))

    # Compose
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.08,
                          top=0.95, bottom=0.03, left=0.07, right=0.99)
    faint = 0.35 * ctx["gallery_colors"] + 0.65
    row_titles = ["Embedding", "Straight in true", "Straight in model"]

    for col_idx, space in enumerate(SPACES):
        g = gallery(space)
        xlim, ylim = col_extents[col_idx]

        # Row 0
        ax = fig.add_subplot(gs[0, col_idx])
        ax.scatter(g[:, 0], g[:, 1], c=ctx["gallery_colors"],
                   s=5, alpha=0.6, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(SPACE_TITLES[col_idx], fontsize=13, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(row_titles[0], fontsize=13, fontweight="bold")

        # Row 1
        ax = fig.add_subplot(gs[1, col_idx])
        ax.scatter(g[:, 0], g[:, 1], c=faint, s=4, alpha=0.5,
                   linewidths=0, zorder=1)
        for t_idx, traj in enumerate(multi_trajs):
            c = row1_coords(space, traj)
            color = TRAJ_COLORS[t_idx]
            ax.plot(c[:, 0], c[:, 1], "-", color=color, lw=2.2, zorder=3)
            ax.scatter(c[:, 0], c[:, 1], c=color, s=22,
                       ec="white", lw=0.7, zorder=4)
            ax.scatter(c[0, 0], c[0, 1], c=color, s=110, marker="o",
                       ec="white", lw=1.5, zorder=5)
        g_goal = row1_coords(space, multi_trajs[0])[-1]
        ax.scatter(g_goal[0], g_goal[1], c="#dd2222", s=180, marker="*",
                   ec="white", lw=1.5, zorder=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(row_titles[1], fontsize=13, fontweight="bold")

        # Row 2: two lines per start (solid = from OU, dashed = from traj)
        ax = fig.add_subplot(gs[2, col_idx])
        ax.scatter(g[:, 0], g[:, 1], c=faint, s=4, alpha=0.5,
                   linewidths=0, zorder=1)
        for t_idx, mp in enumerate(multi_modelplan):
            c_ou, c_traj = row2_coords(space, mp)
            color = TRAJ_COLORS[t_idx]
            for c_path, ls, alpha in [(c_ou,   "-",  1.0),
                                       (c_traj, "--", 0.85)]:
                ax.plot(c_path[:, 0], c_path[:, 1], ls, color=color,
                        lw=2.0, alpha=alpha, zorder=3)
                ax.scatter(c_path[:, 0], c_path[:, 1], c=color, s=18,
                           ec="white", lw=0.6, alpha=alpha, zorder=4)
            ax.scatter(c_ou[0, 0], c_ou[0, 1], c=color, s=110,
                       marker="o", ec="white", lw=1.5, zorder=5)
        g_goal = row2_coords(space, multi_modelplan[0])[0][-1]
        ax.scatter(g_goal[0], g_goal[1], c="#dd2222", s=180, marker="*",
                   ec="white", lw=1.5, zorder=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(row_titles[2], fontsize=13, fontweight="bold")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: planning_quantitative.png — box plots
# ═════════════════════════════════════════════════════════════════════════════

def _action_ratio(theta):
    """(N-1) · Σ‖Δθ‖² / ‖θ_N − θ_0‖²  ≥ 1  (Cauchy–Schwarz)."""
    chord_sq = float(np.sum((theta[-1] - theta[0]) ** 2))
    step_sq  = float(np.sum(np.diff(theta, axis=0) ** 2))
    return (len(theta) - 1) * step_sq / max(chord_sq, 1e-12)


def _tracking_error(theta, theta_opt):
    chord = float(np.linalg.norm(theta_opt[-1] - theta_opt[0]))
    return float(np.linalg.norm(theta - theta_opt, axis=1).mean()
                 / max(chord, 1e-12))


def make_quantitative_figure(env, ctx, save_path, n_steps=8,
                             K=30, k_nn=5, margin=0.25, seed=0):
    """For K random (start, goal) pairs well inside [−π, π]:
          - plan straight in each latent between encoded endpoints
          - decode to θ̂ via kNN on (gallery_z, gallery_angles)
          - compare θ̂ to θ_opt = straight θ-line.
    """
    gallery_angles = ctx["gallery_angles"]
    alphas = np.linspace(0, 1, n_steps)
    rng = np.random.default_rng(seed)

    inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
    inside_idx = np.where(inside)[0]
    print(f"Endpoint pool: {len(inside_idx)} / {len(gallery_angles)} "
          f"within ±{np.pi - margin:.2f}")
    pair_idx = np.stack([
        rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
    ])

    dec_ou   = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
        ctx["gallery_z_ou"],   gallery_angles)
    dec_traj = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
        ctx["gallery_z_traj"], gallery_angles)

    action_data   = {k: [] for k in SPACES}
    tracking_data = {k: [] for k in SPACES}

    for i, j in pair_idx:
        theta_0, theta_N = gallery_angles[i], gallery_angles[j]
        if np.linalg.norm(theta_N - theta_0) < 1e-4:
            continue

        theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])
        img_0 = render_at(env, theta_0, TARGET)
        img_N = render_at(env, theta_N, TARGET)

        z_ou   = encode_images(ctx["enc_ou"],   [img_0, img_N],
                                ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
        z_traj = encode_images(ctx["enc_traj"], [img_0, img_N],
                                ctx["mn_traj"], ctx["sd_traj"], ctx["device"])

        plan_ou   = _straight_latent_plan(z_ou,   n_steps)
        plan_traj = _straight_latent_plan(z_traj, n_steps)
        theta_hat_ou   = dec_ou.predict(plan_ou)
        theta_hat_traj = dec_traj.predict(plan_traj)
        theta_hat_ou[0],   theta_hat_ou[-1]   = theta_0, theta_N
        theta_hat_traj[0], theta_hat_traj[-1] = theta_0, theta_N

        for name, theta_hat in [("true", theta_opt),
                                 ("ou",   theta_hat_ou),
                                 ("traj", theta_hat_traj)]:
            action_data[name].append(_action_ratio(theta_hat))
            tracking_data[name].append(_tracking_error(theta_hat, theta_opt))

    # Plot
    fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=0.4 * np.array((10, 5)))
    labels = ["Optimum", "Gaussian", "Trajectory"]
    colors = ["#888888", "#0072b2", "#cc79a7"]

    for ax, data, title, ideal in [
        (ax_a, action_data,   "Path length",   1.0),
        (ax_t, tracking_data, "Control effort", 0.0),
    ]:
        values = [data[k] for k in SPACES]
        bp = ax.boxplot(values, positions=np.arange(3), widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=1.8),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="#444",
                                        markeredgecolor="none", alpha=0.6))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        ax.axhline(ideal, ls="--", color="gray", lw=1, alpha=0.7)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid()
        if title == "Path length":
            ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}  (K={K} pairs, kNN decoder with k={k_nn})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  type=str, default="results/reacher")
    parser.add_argument("--data_root",    type=str, default="data/reacher")
    parser.add_argument("--out_dir",      type=str, default="figures/reacher")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--gallery_size", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotated frame: no models needed
    make_annotated_frame(out_dir / "reacher_annotated.png")

    # Shared resources for the other three figures
    ctx = load_planning_context(args.results_dir, args.data_root,
                                args.device, args.gallery_size)
    env = make_env()

    make_planning_figure(env, ctx, out_dir / "planning_demo.png")
    make_scatter_figure(env, ctx, out_dir / "planning_scatter.png")
    make_quantitative_figure(env, ctx, out_dir / "planning_quantitative.png")


if __name__ == "__main__":
    main()