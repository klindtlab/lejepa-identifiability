"""
Paper figures for the Reacher experiment.

Produces the following figures:

    reacher_annotated.png     Schematic of the two joint angles overlaid on
                              a rendered frame.

    planning_demo.png         3-row image grid: the true straight-in-θ
                              trajectory between a (start, goal) pair, and
                              the corresponding kNN-retrieved frames for the
                              best OU encoder and the best Trajectory encoder,
                              with a faint overlay of the true frame so
                              deviations are visible.

    planning_scatter.png      3×3 scatter grid. Columns are coordinate
                              systems (true θ-space, OU latent, Traj latent);
                              row 0 shows the gallery embedding, row 1 shows
                              three straight-in-θ paths as they appear in
                              each space, row 2 shows straight-in-model plans
                              (decoded via kNN for the θ panel).

    control_cost.png          (Cor. 4.4, main text) Two panels.
                              Left:  boxplot of control cost divided by
                                     oracle cost for the best OU and best
                                     Traj encoders over K random (start,
                                     goal) pairs.
                              Right: same quantity vs R²(h→z) across ALL
                                     reacher runs, colored by OU vs Traj,
                                     with Pearson r annotated.

    lqr_equivalence.png       (Cor. 4.4, appendix) Synthetic-LQR test:
                              solves a discrete algebraic Riccati equation
                              in true θ-space and in each encoder's latent
                              space, and compares the resulting value
                              functions V*(z₀) vs V̂*(h(z₀)) pointwise.

    planning_quantitative.png (Appendix) Boxplots of path length and
                              control effort over K random pairs, same
                              setup as make_planning_figure but scalar.

Usage (needs GPU + MuJoCo + prerendered gallery with z.npy):

    python -m analysis.make_reacher_figures \\
        --results_dir results/reacher \\
        --data_root   data/reacher \\
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
import matplotlib.patches as patches

from scipy.linalg import solve_discrete_are
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from lejepa_id.reacher import make_env, render_at, solve_ik_grid
from lejepa_id.models import make_cnn_encoder
from run_reacher import normalize_uint8


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

TARGET       = np.array([0.1, 0.1])          # fixed target position (x, y)

OU_COLOR     = "#0072b2"                     # shared across figures
TRAJ_COLOR   = "#cc79a7"
OPT_COLOR    = "#888888"
GOAL_COLOR   = "#dd2222"
START_COLOR  = "#22cc22"

# For the 3×3 scatter figure
SPACES       = ["true", "ou", "traj"]
SPACE_TITLES = ["True (θ-space)", "Gaussian latent", "Trajectory latent"]
TRAJ_COLORS  = ["#1a1a1a", OU_COLOR, TRAJ_COLOR]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def find_best_checkpoint(results_dir, condition):
    """Return (run_dir, result_dict) for the highest-R² checkpointed run of
    the requested condition ('ou' or 'traj')."""
    best_r2, best_dir, best_res = -np.inf, None, None
    for p in Path(results_dir).rglob("result.json"):
        if not (p.parent / "checkpoint.pt").exists():
            continue
        with open(p) as f:
            r = json.load(f)
        is_traj = "delta" in r and r.get("rho") is None
        if condition == "ou"   and is_traj:      continue
        if condition == "traj" and not is_traj:  continue
        if r.get("r2_hz", -1) > best_r2:
            best_r2, best_dir, best_res = r["r2_hz"], p.parent, r
    if best_res is None:
        raise RuntimeError(
            f"No {condition} results with checkpoint in {results_dir}")
    print(f"Best {condition}: R²={best_r2:.4f}  run={best_res['run_name']}")
    return best_dir, best_res


def load_encoder(ckpt_path, device):
    """Load a saved CNN encoder plus its per-channel normalization stats."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = make_cnn_encoder(d_latent=ckpt["d_latent"], device=device)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    return enc, ckpt["train_mean"], ckpt["train_std"], ckpt["d_latent"]


@torch.no_grad()
def encode_batched(enc, imgs_norm, device, bs=512):
    """Encode a large stack of already-normalized images in batches."""
    outs = []
    for i in range(0, len(imgs_norm), bs):
        outs.append(enc(imgs_norm[i:i + bs].to(device)).cpu())
    return torch.cat(outs).numpy()


def encode_images(enc, imgs, mn, sd, device):
    """Encode a small list of (3, H, W) float images — normalizes inline."""
    mn_b, sd_b = mn[:, None, None], sd[:, None, None]
    arr = np.stack([(im - mn_b) / (sd_b + 1e-6) for im in imgs]).astype(np.float32)
    with torch.no_grad():
        return enc(torch.from_numpy(arr).to(device)).cpu().numpy()


def project_to_2d(z_gallery, z_points):
    """For d=2, identity; otherwise PCA fit on gallery and applied to both."""
    if z_gallery.shape[1] == 2:
        return z_gallery, z_points, None
    pca = PCA(n_components=2).fit(z_gallery)
    return pca.transform(z_gallery), pca.transform(z_points), pca


def try_load_true_angles(eval_dir, gallery_size):
    """Find the ground-truth angles file regardless of its exact name."""
    for fname in ("z.npy", "angles.npy", "qpos.npy"):
        p = os.path.join(eval_dir, fname)
        if os.path.exists(p):
            arr = np.load(p)[:gallery_size]
            print(f"Loaded true angles from {fname}  shape={arr.shape}")
            return arr
    return None


def make_colors(z):
    """Polar color map: hue = angle(z), lightness = ||z||."""
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
    """Display a (3, H, W) or (H, W, 3) image without axes or spines."""
    if img.ndim == 3 and img.shape[0] == 3:
        ax.imshow(img.transpose(1, 2, 0))
    else:
        ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def border(ax, color, width=4):
    """Draw a colored border around the axes (used to mark Start/Goal)."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(width); sp.set_visible(True)


def sample_endpoint_pairs(gallery_angles, K, margin=0.25, seed=0):
    """Sample K random (start, goal) index pairs well inside [−π, π]."""
    rng = np.random.default_rng(seed)
    inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
    inside_idx = np.where(inside)[0]
    return np.stack([
        rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
    ]), inside_idx


def _straight_latent_plan(z_endpoints, n_steps):
    """Linear interpolation in latent space between two endpoints."""
    alphas = np.linspace(0, 1, n_steps)
    return np.stack([(1 - a) * z_endpoints[0] + a * z_endpoints[1]
                     for a in alphas])


def _nn_retrieve(plan_z, gallery_z, gallery_display):
    """1-NN retrieval of gallery images closest to each point in plan_z."""
    out = []
    for pz in plan_z:
        idx = int(np.linalg.norm(gallery_z - pz, axis=1).argmin())
        out.append(gallery_display[idx])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Shared planning context (loaded once, passed to every figure function)
# ═════════════════════════════════════════════════════════════════════════════

def load_planning_context(results_dir, data_root, device, gallery_size):
    """Load both encoders, the evaluation gallery, and the two encoders'
    per-gallery embeddings. Everything needed for the planning figures."""
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
# Figure: annotated Reacher frame
# ═════════════════════════════════════════════════════════════════════════════

def make_annotated_frame(save_path, img_size=256):
    env = make_env()
    qpos = np.array([-np.pi / 2, -np.pi / 2])
    img = render_at(env, qpos, TARGET, height=img_size, width=img_size)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.transpose(1, 2, 0))

    sh = (128, 128)
    el = (128, 178)

    ax.add_patch(patches.Arc((sh[0] + 4, sh[1]), 46, 46, angle=0,
                             theta1=0, theta2=90, color="#22cc22", linewidth=3))
    ax.annotate(r"$z_0$", xy=(sh[0] + 32, sh[1] + 32),
                fontsize=20, fontweight="bold", color="#22cc22")

    ax.add_patch(patches.Arc((el[0] - 4, el[1] - 4), 46, 46, angle=0,
                             theta1=180, theta2=270, color="#ff8800", linewidth=3))
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
# Figure: planning_demo.png — 3-row image grid with ghost overlay
# ═════════════════════════════════════════════════════════════════════════════

def make_planning_figure(env, ctx, save_path, n_steps=8,
                         qpos_start=None, ghost_alpha=0.3):
    """Three rows: the true straight-in-θ trajectory, the kNN retrieval of
    a straight-line interpolant in the OU latent, and the same for the
    Trajectory latent. Rows 2 and 3 blend each retrieved frame with the
    corresponding true frame at weight `ghost_alpha`, so deviations from
    the true motion are visible."""
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
        # Pin endpoints so Start/Goal columns match across rows exactly.
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
        ("True\ntrajectory",                                    true_imgs),
        (f"Gaussian\n(R²={ctx['result_ou']['r2_hz']:.2f})",     blend(ou_imgs)),
        (f"Trajectory\n(R²={ctx['result_traj']['r2_hz']:.2f})", blend(traj_imgs)),
    ]

    fig, axes = plt.subplots(3, n_steps, figsize=(2.0 * n_steps, 6.0))
    for r, (label, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            show_img(axes[r, c], im)
        axes[r, 0].text(-0.25, 0.5, label,
                        transform=axes[r, 0].transAxes,
                        fontsize=12, fontweight="bold",
                        ha="right", va="center")
        border(axes[r, 0],  START_COLOR)
        border(axes[r, -1], GOAL_COLOR)

    axes[0, 0].set_title("Start", color=START_COLOR, fontsize=13,
                         fontweight="bold")
    axes[0, -1].set_title("Goal", color=GOAL_COLOR,  fontsize=13,
                          fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure: planning_scatter.png — 3×3 embedding / path scatter grid
# ═════════════════════════════════════════════════════════════════════════════

def make_scatter_figure(env, ctx, save_path, n_steps=8):
    """Columns = coordinate systems (true θ, OU latent, Traj latent).
    Row 0: gallery embedding, polar-colored.
    Row 1: three straight-in-θ trajectories, as they appear in each space.
    Row 2: for each start, two plans — straight in the OU latent (solid) and
           straight in the Traj latent (dashed) — as they appear in each
           space. For the θ-column we decode via kNN."""
    qpos_goal, _ = solve_ik_grid(env, TARGET)
    qpos_starts = [
        np.array([-3 / 4 * np.pi,  1 / 4 * np.pi]),
        np.array([ 1 / 4 * np.pi,  1 / 2 * np.pi]),
        np.array([-1 / 2 * np.pi, -3 / 4 * np.pi]),
    ]
    alphas = np.linspace(0, 1, n_steps)

    # Row 1: straight θ-line, then encode.
    multi_trajs = []
    for qs in qpos_starts:
        qpos_path = np.array([(1 - a) * qs + a * qpos_goal for a in alphas])
        imgs = [render_at(env, q, TARGET) for q in qpos_path]
        z_ou   = encode_images(ctx["enc_ou"],   imgs,
                               ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
        z_traj = encode_images(ctx["enc_traj"], imgs,
                               ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
        multi_trajs.append({"theta": qpos_path, "z_ou": z_ou, "z_traj": z_traj})

    # Row 2: plan straight in each model, decode to θ via kNN, then re-encode
    # in the *other* model's space so we can display cross-coordinate paths.
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

        imgs_from_ou   = [render_at(env, q, TARGET) for q in theta_from_ou]
        imgs_from_traj = [render_at(env, q, TARGET) for q in theta_from_traj]
        z_traj_from_ou = encode_images(ctx["enc_traj"], imgs_from_ou,
                                       ctx["mn_traj"], ctx["sd_traj"],
                                       ctx["device"])
        z_ou_from_traj = encode_images(ctx["enc_ou"],   imgs_from_traj,
                                       ctx["mn_ou"],   ctx["sd_ou"],
                                       ctx["device"])

        multi_modelplan.append({
            "true_from_ou":   theta_from_ou,
            "true_from_traj": theta_from_traj,
            "ou_from_ou":     plan_ou_z,      # straight in OU by construction
            "traj_from_traj": plan_traj_z,    # straight in Traj by construction
            "traj_from_ou":   z_traj_from_ou,
            "ou_from_traj":   z_ou_from_traj,
        })

    def gallery(space):
        return {"true": ctx["gallery_angles"],
                "ou":   ctx["gallery_2d_ou"],
                "traj": ctx["gallery_2d_traj"]}[space]

    def row1_coords(space, traj):
        return {"true": traj["theta"],
                "ou":   traj["z_ou"],
                "traj": traj["z_traj"]}[space]

    def row2_coords(space, mp):
        """Returns (solid, dashed) paths = (planned-in-OU, planned-in-traj),
        rendered in the requested coordinate space."""
        if space == "true":
            return mp["true_from_ou"], mp["true_from_traj"]
        if space == "ou":
            return mp["ou_from_ou"],   mp["ou_from_traj"]
        return     mp["traj_from_ou"], mp["traj_from_traj"]

    # Per-column extent (shared across all 3 rows of that column).
    col_extents = []
    for space in SPACES:
        g = gallery(space)
        row1 = [row1_coords(space, t) for t in multi_trajs]
        row2_flat = [p for mp in multi_modelplan
                       for p in row2_coords(space, mp)]
        col_extents.append(square_extent(g, *row1, *row2_flat))

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.08,
                          top=0.95, bottom=0.03, left=0.07, right=0.99)
    faint = 0.35 * ctx["gallery_colors"] + 0.65
    row_titles = ["Embedding", "Straight in true", "Straight in model"]

    for col_idx, space in enumerate(SPACES):
        g = gallery(space)
        xlim, ylim = col_extents[col_idx]

        # Row 0: gallery embedding
        ax = fig.add_subplot(gs[0, col_idx])
        ax.scatter(g[:, 0], g[:, 1], c=ctx["gallery_colors"],
                   s=5, alpha=0.6, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(SPACE_TITLES[col_idx], fontsize=13, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(row_titles[0], fontsize=13, fontweight="bold")

        # Row 1: straight-in-θ trajectories
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
        ax.scatter(g_goal[0], g_goal[1], c=GOAL_COLOR, s=180, marker="*",
                   ec="white", lw=1.5, zorder=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(row_titles[1], fontsize=13, fontweight="bold")

        # Row 2: straight-in-model plans (solid = OU, dashed = Traj)
        ax = fig.add_subplot(gs[2, col_idx])
        ax.scatter(g[:, 0], g[:, 1], c=faint, s=4, alpha=0.5,
                   linewidths=0, zorder=1)
        for t_idx, mp in enumerate(multi_modelplan):
            c_ou, c_traj = row2_coords(space, mp)
            color = TRAJ_COLORS[t_idx]
            for c_path, ls, alpha in [(c_ou, "-", 1.0),
                                       (c_traj, "--", 0.85)]:
                ax.plot(c_path[:, 0], c_path[:, 1], ls, color=color,
                        lw=2.0, alpha=alpha, zorder=3)
                ax.scatter(c_path[:, 0], c_path[:, 1], c=color, s=18,
                           ec="white", lw=0.6, alpha=alpha, zorder=4)
            ax.scatter(c_ou[0, 0], c_ou[0, 1], c=color, s=110,
                       marker="o", ec="white", lw=1.5, zorder=5)
        g_goal = row2_coords(space, multi_modelplan[0])[0][-1]
        ax.scatter(g_goal[0], g_goal[1], c=GOAL_COLOR, s=180, marker="*",
                   ec="white", lw=1.5, zorder=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(row_titles[2], fontsize=13, fontweight="bold")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")



# ═════════════════════════════════════════════════════════════════════════════
# Cor. 4.4 — shared cost function
# ═════════════════════════════════════════════════════════════════════════════

def _quadratic_cost(theta_path, theta_goal, w_state=1.0, w_action=1.0):
    """LQR-style quadratic cost,
       J(path) = Σ w_state ‖θ_t - θ_goal‖² + Σ w_action ‖θ_{t+1} - θ_t‖².
    Both terms are O(n)-invariant (‖Rθ - Rθ*‖ = ‖θ - θ*‖ for R ∈ O(n)),
    so Cor. 4.4 predicts equal J-values across encoders that differ only
    by an orthogonal rotation."""
    theta_path = np.asarray(theta_path)
    theta_goal = np.asarray(theta_goal)
    state_cost  = float(np.sum(np.sum((theta_path - theta_goal) ** 2, axis=1)))
    action_cost = float(np.sum(np.sum(np.diff(theta_path, axis=0) ** 2, axis=1)))
    return w_state * state_cost + w_action * action_cost


def _cost_ratios_for_pairs(enc_mn_sd, decoder, pair_idx, gallery_angles, env,
                           device, n_steps=8, w_state=1.0, w_action=1.0):
    """Given (encoder, mean, std) and a kNN decoder, compute per-pair
    control-cost ratio (encoder / oracle) over a fixed pair_idx list.
    Factored out so best-run boxplot and per-run scatter share the logic.
    Returns (ratios array, oracle costs array)."""
    enc, mn, sd = enc_mn_sd
    alphas = np.linspace(0, 1, n_steps)
    oracle_costs, encoder_costs = [], []
    for i, j in pair_idx:
        theta_0, theta_N = gallery_angles[i], gallery_angles[j]
        if np.linalg.norm(theta_N - theta_0) < 1e-4:
            continue
        theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])
        img_0 = render_at(env, theta_0, TARGET)
        img_N = render_at(env, theta_N, TARGET)
        z_ends = encode_images(enc, [img_0, img_N], mn, sd, device)
        plan_z = _straight_latent_plan(z_ends, n_steps)
        theta_hat = decoder.predict(plan_z)
        theta_hat[0], theta_hat[-1] = theta_0, theta_N
        oracle_costs .append(_quadratic_cost(theta_opt, theta_N, w_state, w_action))
        encoder_costs.append(_quadratic_cost(theta_hat, theta_N, w_state, w_action))
    oracle_costs  = np.array(oracle_costs)
    encoder_costs = np.array(encoder_costs)
    ratios = encoder_costs / np.maximum(oracle_costs, 1e-9)
    return ratios, oracle_costs


# ═════════════════════════════════════════════════════════════════════════════
# Figure: control_cost.png — Cor. 4.4 two-panel main-text figure
# ═════════════════════════════════════════════════════════════════════════════

def _collect_control_cost_across_runs(results_dir, data_root, device,
                                      gallery_size=10000, K=30, n_steps=8,
                                      k_nn=5, margin=0.25, seed=0,
                                      cache_path=None):
    """For every reacher run with a checkpoint, compute the mean control-cost
    ratio over K random start-goal pairs and pair it with R² values from
    result.json. Returns a list of dicts. Cached to JSON when cache_path is
    set, so repeat calls skip the loop entirely.

    IMPORTANT: the cache is keyed only by path, not by K. If you change K,
    point cache_path at a different file (or pass None)."""
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cached control-cost entries from {cache_path}")
        return out

    gallery_u8 = np.load(
        os.path.join(data_root, "eval", "img.npy"))[:gallery_size]
    gallery_angles = try_load_true_angles(
        os.path.join(data_root, "eval"), gallery_size)
    env = make_env()

    pair_idx, _ = sample_endpoint_pairs(gallery_angles, K, margin, seed)

    out = []
    result_paths = sorted(Path(results_dir).rglob("result.json"))
    for rp in tqdm(result_paths, desc="control-cost across runs"):
        with open(rp) as f:
            r = json.load(f)
        ckpt = rp.parent / "checkpoint.pt"
        if not ckpt.exists():
            continue

        enc, mean, std, _ = load_encoder(ckpt, device)
        gnorm = torch.from_numpy(normalize_uint8(gallery_u8, mean, std))
        gallery_z = encode_batched(enc, gnorm, device)
        decoder = KNeighborsRegressor(
            n_neighbors=k_nn, weights="distance"
        ).fit(gallery_z, gallery_angles)

        ratios, _ = _cost_ratios_for_pairs(
            (enc, mean, std), decoder, pair_idx, gallery_angles, env, device,
            n_steps=n_steps)

        per_dim = r.get("r2_hz_per_dim", [None, None])
        out.append({
            "run_name":    r["run_name"],
            "type":        r.get("type",
                                 "ou" if r.get("rho") is not None else "traj"),
            "rho":         r.get("rho"),
            "delta":       r.get("delta"),
            "lamb":        r.get("lamb"),
            "seed":        r.get("seed"),
            "r2_zh":       r.get("r2_zh"),
            "r2_hz":       r.get("r2_hz"),
            "r2_hz_dim0":  per_dim[0] if len(per_dim) > 0 else None,
            "r2_hz_dim1":  per_dim[1] if len(per_dim) > 1 else None,
            "control_cost_ratio_mean": float(np.mean(ratios)),
        })

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Cached {len(out)} entries to {cache_path}")
    return out


def make_control_cost_figure(env, ctx, save_path,
                             results_dir, data_root, device,
                             n_steps=8, K=30, k_nn=5, margin=0.25, seed=0,
                             w_state=1.0, w_action=1.0,
                             cache_path=None, gallery_size=10000):
    """Two-panel figure for main text.

    Left  — boxplot of control cost / oracle for the best OU and best Traj
            encoders, over K random (start, goal) pairs.
    Right — the same mean ratio per run, vs linear identifiability R²(h→z)
            across ALL reacher runs, colored by OU vs Traj."""
    gallery_angles = ctx["gallery_angles"]
    pair_idx, _ = sample_endpoint_pairs(gallery_angles, K, margin, seed)

    # ── Left panel: best-encoder boxplot ────────────────────────────────
    dec_ou   = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
        ctx["gallery_z_ou"],   gallery_angles)
    dec_traj = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
        ctx["gallery_z_traj"], gallery_angles)

    ratio_ou, _   = _cost_ratios_for_pairs(
        (ctx["enc_ou"],   ctx["mn_ou"],   ctx["sd_ou"]),
        dec_ou,   pair_idx, gallery_angles, env, ctx["device"],
        n_steps=n_steps, w_state=w_state, w_action=w_action)
    ratio_traj, _ = _cost_ratios_for_pairs(
        (ctx["enc_traj"], ctx["mn_traj"], ctx["sd_traj"]),
        dec_traj, pair_idx, gallery_angles, env, ctx["device"],
        n_steps=n_steps, w_state=w_state, w_action=w_action)

    print(f"\n[Control-cost]  ratio_ou   median={np.median(ratio_ou):.3f}  "
          f"mean={np.mean(ratio_ou):.3f}")
    print(f"[Control-cost]  ratio_traj median={np.median(ratio_traj):.3f}  "
          f"mean={np.mean(ratio_traj):.3f}")

    # ── Right panel: across-all-runs scatter (uses cache) ───────────────
    all_runs = _collect_control_cost_across_runs(
        results_dir, data_root, device,
        gallery_size=gallery_size, K=K, n_steps=n_steps, k_nn=k_nn,
        margin=margin, seed=seed, cache_path=cache_path)

    # ── Compose figure ──────────────────────────────────────────────────
    fig, (ax_box, ax_sc) = plt.subplots(1, 2, figsize=0.6 * np.array((9, 3.8)))

    labels = ["Optimum", "Gaussian", "Trajectory"]
    colors = [OPT_COLOR,  OU_COLOR,   TRAJ_COLOR]
    values = [np.ones_like(ratio_ou), ratio_ou, ratio_traj]
    bp = ax_box.boxplot(values, positions=np.arange(3), widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=1.8),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="#444",
                                        markeredgecolor="none", alpha=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    ax_box.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.7)
    ax_box.set_xticks(np.arange(3))
    ax_box.set_xticklabels(labels, rotation=30, fontsize=9)
    ax_box.set_ylabel("Control Cost")
    ax_box.set_yscale("log")
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.grid(alpha=0.3)

    ou_runs   = [r for r in all_runs if r["type"] == "ou"
                 and r["r2_hz"] is not None
                 and r["control_cost_ratio_mean"] is not None]
    traj_runs = [r for r in all_runs if r["type"] == "traj"
                 and r["r2_hz"] is not None
                 and r["control_cost_ratio_mean"] is not None]

    def _plot_group(runs, color, marker, label):
        if not runs:
            return
        xs = np.array([r["r2_hz"] for r in runs])
        ys = np.array([r["control_cost_ratio_mean"] for r in runs])
        ax_sc.scatter(xs, ys, s=32, alpha=0.75, c=color, marker=marker,
                      edgecolors="black", linewidths=0.3, label=label)

    _plot_group(ou_runs,   OU_COLOR,   "o", "OU")
    _plot_group(traj_runs, TRAJ_COLOR, "s", "Trajectory")
    ax_sc.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.6)
    ax_sc.set_xlabel(r"Linear Identifiability [$R^2$]")
    ax_sc.set_ylabel("Control Cost")
    ax_sc.set_yscale("log")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure: lqr_equivalence.png — Cor. 4.4 LQR covariance (appendix)
# ═════════════════════════════════════════════════════════════════════════════
#
# Cor. 4.4 predicts that for an O(n)-invariant quadratic cost, the Riccati
# equation transforms covariantly under the residual Q, so the optimal value
# V*(z_0) equals V̂*(h(z_0)). This test uses SYNTHETIC linear dynamics so we
# can solve the DARE analytically in both coordinate systems and compare V*
# pointwise. The linear dynamics are a stand-in: we are testing whether the
# ENCODER's residual rotation preserves LQR value, not whether the real
# reacher is linear. This isolates the covariance claim of Cor. 4.4 cleanly.

def _linear_regress_encoder(z_gallery, h_gallery):
    """Fit ẑ = M z + b via ordinary least squares. For an ideal Cor. 4.4
    encoder h(z) = Q z, so M ≈ Q (orthogonal) and b ≈ 0."""
    Z = np.column_stack([z_gallery, np.ones(len(z_gallery))])
    Mb, *_ = np.linalg.lstsq(Z, h_gallery, rcond=None)
    return Mb[:-1].T, Mb[-1]          # (M, b)


def _solve_dare_lqr(A, B, W, R):
    """Infinite-horizon discrete-time LQR. Returns (P, K) where V*(z) = z^T P z
    and K is the optimal feedback gain."""
    P = solve_discrete_are(A, B, W, R)
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, gain


def make_lqr_equivalence_figure(ctx, save_path, n_samples=200, seed=0):
    """Two-panel figure: (left) scatter of V̂*(h(z)) vs V*(z) per initial state;
    (right) boxplot of relative value error |V̂* - V*| / |V*|.

    Exact Cor. 4.4 (Gaussian encoder, in the limit) would land every point on
    the diagonal on the left and give zero on the right. The approximate-
    identifiability residual of Thm. 4.3 determines how far off we land."""
    rng = np.random.default_rng(seed)
    gallery_angles = ctx["gallery_angles"]
    n = 2

    # Best-fit linear maps from true angles to each encoder's latent space.
    M_ou,   b_ou   = _linear_regress_encoder(gallery_angles,
                                             ctx["gallery_z_ou"])
    M_traj, b_traj = _linear_regress_encoder(gallery_angles,
                                             ctx["gallery_z_traj"])

    def _orth_err(M):
        return float(np.linalg.norm(M.T @ M - np.eye(n), "fro"))
    print(f"[LQR]  ||M_ou^T M_ou - I||_F   = {_orth_err(M_ou):.4f}  "
          f"(ideal: 0 for exact Cor. 4.4)")
    print(f"[LQR]  ||M_traj^T M_traj - I||_F = {_orth_err(M_traj):.4f}")

    # Synthetic linear dynamics in true θ-space: stable, slightly coupled.
    A_true = np.array([[0.95,  0.05],
                       [-0.03, 0.92]])
    B_true = 0.3 * np.eye(n)
    W, R   = np.eye(n), np.eye(n)

    # DARE in true space and in each encoder's pushforward.
    P_true, _ = _solve_dare_lqr(A_true, B_true, W, R)

    def _pushforward(M, A, B):
        M_inv = np.linalg.pinv(M)
        return M @ A @ M_inv, M @ B

    A_ou,   B_ou_p   = _pushforward(M_ou,   A_true, B_true)
    A_traj, B_traj_p = _pushforward(M_traj, A_true, B_true)

    # In ẑ-space, cost W_hat = M W M^T (covariant with rotation).
    W_ou_p   = M_ou   @ W @ M_ou.T
    W_traj_p = M_traj @ W @ M_traj.T

    P_ou,   _ = _solve_dare_lqr(A_ou,   B_ou_p,   W_ou_p,   R)
    P_traj, _ = _solve_dare_lqr(A_traj, B_traj_p, W_traj_p, R)

    # Compare V* pointwise for a random subset of the eval gallery.
    idx = rng.choice(len(gallery_angles), n_samples, replace=False)
    z0 = gallery_angles[idx]
    zhat_ou   = z0 @ M_ou.T   + b_ou
    zhat_traj = z0 @ M_traj.T + b_traj

    def _val(P, z):
        return np.einsum("ni,ij,nj->n", z, P, z)

    V_true = _val(P_true, z0)
    V_ou   = _val(P_ou,   zhat_ou)
    V_traj = _val(P_traj, zhat_traj)

    err_ou   = np.abs(V_ou   - V_true) / (np.abs(V_true) + 1e-9)
    err_traj = np.abs(V_traj - V_true) / (np.abs(V_true) + 1e-9)
    print(f"[LQR]  |V̂ - V*| / |V*|   OU   median={np.median(err_ou):.4f}  "
          f"mean={np.mean(err_ou):.4f}")
    print(f"[LQR]  |V̂ - V*| / |V*|   Traj median={np.median(err_traj):.4f}  "
          f"mean={np.mean(err_traj):.4f}")

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))

    lim = (min(V_true.min(), V_ou.min(), V_traj.min()),
           max(V_true.max(), V_ou.max(), V_traj.max()))
    ax_s.plot(lim, lim, "k--", lw=1, alpha=0.6, label="ideal ($\\hat V=V^*$)")
    ax_s.scatter(V_true, V_ou,   s=14, alpha=0.7, c=OU_COLOR,
                 edgecolors="none", label="Gaussian")
    ax_s.scatter(V_true, V_traj, s=14, alpha=0.7, c=TRAJ_COLOR,
                 edgecolors="none", label="Trajectory")
    ax_s.set_xlabel("True-latent LQR value $V^*(z_0)$")
    ax_s.set_ylabel("Learned-latent value $\\hat V^*(h(z_0))$")
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.grid(alpha=0.3); ax_s.legend(fontsize=8)

    bp = ax_b.boxplot([err_ou, err_traj], positions=[0, 1], widths=0.55,
                      patch_artist=True,
                      medianprops=dict(color="black", lw=1.8),
                      flierprops=dict(marker="o", markersize=3,
                                      markerfacecolor="#444",
                                      markeredgecolor="none", alpha=0.5))
    for patch, c in zip(bp["boxes"], [OU_COLOR, TRAJ_COLOR]):
        patch.set_facecolor(c); patch.set_edgecolor("black")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Gaussian", "Trajectory"], fontsize=9)
    ax_b.set_ylabel("$|\\hat V^* - V^*| / |V^*|$")
    ax_b.set_yscale("log")
    ax_b.axhline(1.0, ls=":", color="gray", lw=1, alpha=0.6)
    ax_b.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


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
    parser.add_argument("--planning_K",   type=int, default=100,
                        help="Number of (start, goal) pairs for planning figures.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotated frame: no models needed
    make_annotated_frame(out_dir / "reacher_annotated.png")

    # Shared resources
    ctx = load_planning_context(args.results_dir, args.data_root,
                                args.device, args.gallery_size)
    env = make_env()

    # Planning figures
    make_planning_figure(env, ctx, out_dir / "planning_demo.png")
    make_scatter_figure(env,  ctx, out_dir / "planning_scatter.png")

    # Cor. 4.4, Experiment A (main text)
    cache_path = os.path.join(
        args.results_dir, 
        # f"control_cost_cache_K{args.planning_K}.json"
        f"control_cost_cache.json"
    )
    make_control_cost_figure(
        env, ctx,
        save_path=out_dir / "control_cost.png",
        results_dir=args.results_dir,
        data_root=args.data_root,
        device=args.device,
        gallery_size=args.gallery_size,
        K=args.planning_K,
        cache_path=cache_path,
    )

    # Cor. 4.4, Experiment B (appendix, synthetic dynamics)
    make_lqr_equivalence_figure(ctx, out_dir / "lqr_equivalence.png")


if __name__ == "__main__":
    main()




# """
# Paper figures for Reacher experiment.

# Produces four figures:
#     1. reacher_annotated.png      — schematic of the two latent angles
#     2. planning_demo.png          — 3-row image grid: true / OU retrieval /
#                                     traj retrieval, with true-frame ghost
#                                     overlay on the two model rows
#     3. planning_scatter.png       — 3x3 scatter: embeddings, straight-in-true
#                                     trajectories, straight-in-model trajectories
#     4. planning_quantitative.png  — boxplots: path length (log y) and
#                                     control effort over K random (start, goal)
#                                     pairs, with kNN decoder θ̂ = f^{-1}(ẑ).

# Usage (needs GPU + MuJoCo + rendered gallery with z.npy):
#     python -m analysis.make_reacher_figures \
#         --results_dir results/reacher \
#         --data_root   data/reacher \
#         --out_dir     figures/reacher
# """

# import os
# os.environ.setdefault("MUJOCO_GL", "egl")

# import json
# import argparse
# import colorsys
# import numpy as np
# import torch
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as patches

# from sklearn.neighbors import KNeighborsRegressor

# from lejepa_id.reacher import make_env, render_at, solve_ik_grid
# from lejepa_id.models import make_cnn_encoder
# from run_reacher import normalize_uint8

# from sklearn.decomposition import PCA
# from scipy.linalg import solve_discrete_are
# from scipy.stats import pearsonr as _pearsonr_cc
# from tqdm import tqdm


# TARGET = np.array([0.1, 0.1])


# # ═════════════════════════════════════════════════════════════════════════════
# # Helpers
# # ═════════════════════════════════════════════════════════════════════════════

# def find_best_checkpoint(results_dir, condition):
#     """Highest-R² checkpointed run for 'ou' or 'traj'."""
#     best_r2, best_dir, best_res = -np.inf, None, None
#     for p in Path(results_dir).rglob("result.json"):
#         if not (p.parent / "checkpoint.pt").exists():
#             continue
#         with open(p) as f:
#             r = json.load(f)
#         is_traj = "delta" in r and r.get("rho") is None
#         if condition == "ou" and is_traj:
#             continue
#         if condition == "traj" and not is_traj:
#             continue
#         if r.get("r2_hz", -1) > best_r2:
#             best_r2, best_dir, best_res = r["r2_hz"], p.parent, r
#     if best_res is None:
#         raise RuntimeError(f"No {condition} results with checkpoint in {results_dir}")
#     print(f"Best {condition}: R²={best_r2:.4f}  run={best_res['run_name']}")
#     return best_dir, best_res


# def load_encoder(ckpt_path, device):
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
#     enc = make_cnn_encoder(d_latent=ckpt["d_latent"], device=device)
#     enc.load_state_dict(ckpt["encoder_state_dict"])
#     enc.eval()
#     return enc, ckpt["train_mean"], ckpt["train_std"], ckpt["d_latent"]


# @torch.no_grad()
# def encode_batched(enc, imgs_norm, device, bs=512):
#     outs = []
#     for i in range(0, len(imgs_norm), bs):
#         outs.append(enc(imgs_norm[i:i + bs].to(device)).cpu())
#     return torch.cat(outs).numpy()


# def encode_images(enc, imgs, mn, sd, device):
#     """Encode a small list of (3,H,W) float images."""
#     mn_b, sd_b = mn[:, None, None], sd[:, None, None]
#     arr = np.stack([(im - mn_b) / (sd_b + 1e-6) for im in imgs]).astype(np.float32)
#     with torch.no_grad():
#         return enc(torch.from_numpy(arr).to(device)).cpu().numpy()


# def project_to_2d(z_gallery, z_points):
#     """d=2: identity. Else PCA fit on gallery, applied to both."""
#     if z_gallery.shape[1] == 2:
#         return z_gallery, z_points, None
#     pca = PCA(n_components=2).fit(z_gallery)
#     return pca.transform(z_gallery), pca.transform(z_points), pca


# def try_load_true_angles(eval_dir, gallery_size):
#     for fname in ("z.npy", "angles.npy", "qpos.npy"):
#         p = os.path.join(eval_dir, fname)
#         if os.path.exists(p):
#             arr = np.load(p)[:gallery_size]
#             print(f"Loaded true angles from {fname}  shape={arr.shape}")
#             return arr
#     return None


# def make_colors(z):
#     """Polar color map: hue = angle, lightness = radius."""
#     if hasattr(z, "cpu"):
#         z = z.cpu().numpy()
#     x, y = z[:, 0], z[:, 1]
#     angles = np.arctan2(y, x)
#     radii = np.sqrt(x ** 2 + y ** 2)
#     hue = (angles + np.pi) / (2 * np.pi)
#     lightness = 0.3 + 0.4 * (radii / (radii.max() + 1e-8))
#     saturation = np.full_like(hue, 0.85)
#     return np.array([colorsys.hls_to_rgb(h, l, s)
#                      for h, l, s in zip(hue, lightness, saturation)])


# def square_extent(*arrays, pad=0.08):
#     """Shared square xlim/ylim covering all input (N, 2) arrays."""
#     pts = np.vstack([a for a in arrays if a is not None and len(a) > 0])
#     xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
#     ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
#     cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
#     half = 0.5 * max(xmax - xmin, ymax - ymin) * (1 + pad)
#     return (cx - half, cx + half), (cy - half, cy + half)


# def show_img(ax, img):
#     if img.ndim == 3 and img.shape[0] == 3:
#         ax.imshow(img.transpose(1, 2, 0))
#     else:
#         ax.imshow(img)
#     ax.set_xticks([]); ax.set_yticks([])
#     for sp in ax.spines.values():
#         sp.set_visible(False)


# def border(ax, color, width=4):
#     ax.set_xticks([]); ax.set_yticks([])
#     for sp in ax.spines.values():
#         sp.set_edgecolor(color); sp.set_linewidth(width); sp.set_visible(True)


# # ═════════════════════════════════════════════════════════════════════════════
# # Shared context (loaded once, passed to every figure function)
# # ═════════════════════════════════════════════════════════════════════════════

# def load_planning_context(results_dir, data_root, device, gallery_size):
#     """Load both encoders, the gallery, and per-encoder gallery embeddings."""
#     gallery_u8 = np.load(os.path.join(data_root, "eval", "img.npy"))[:gallery_size]
#     gallery_angles = try_load_true_angles(os.path.join(data_root, "eval"),
#                                           gallery_size)
#     if gallery_angles is None:
#         raise RuntimeError("Need eval angles (z.npy / angles.npy / qpos.npy)")

#     run_ou,   res_ou   = find_best_checkpoint(results_dir, "ou")
#     run_traj, res_traj = find_best_checkpoint(results_dir, "traj")
#     enc_ou,   mn_ou,   sd_ou,   _ = load_encoder(run_ou   / "checkpoint.pt", device)
#     enc_traj, mn_traj, sd_traj, _ = load_encoder(run_traj / "checkpoint.pt", device)

#     gnorm_ou   = torch.from_numpy(normalize_uint8(gallery_u8, mn_ou,   sd_ou))
#     gnorm_traj = torch.from_numpy(normalize_uint8(gallery_u8, mn_traj, sd_traj))
#     gallery_z_ou   = encode_batched(enc_ou,   gnorm_ou,   device)
#     gallery_z_traj = encode_batched(enc_traj, gnorm_traj, device)
#     gallery_2d_ou,   _, _ = project_to_2d(gallery_z_ou,   gallery_z_ou)
#     gallery_2d_traj, _, _ = project_to_2d(gallery_z_traj, gallery_z_traj)

#     return {
#         "device": device,
#         "gallery_u8": gallery_u8,
#         "gallery_display": [im.astype(np.float32) / 255.0 for im in gallery_u8],
#         "gallery_angles": gallery_angles,
#         "gallery_colors": make_colors(gallery_angles),
#         "enc_ou":   enc_ou,   "mn_ou":   mn_ou,   "sd_ou":   sd_ou,
#         "enc_traj": enc_traj, "mn_traj": mn_traj, "sd_traj": sd_traj,
#         "gallery_z_ou":   gallery_z_ou,
#         "gallery_z_traj": gallery_z_traj,
#         "gallery_2d_ou":   gallery_2d_ou,
#         "gallery_2d_traj": gallery_2d_traj,
#         "result_ou":   res_ou,
#         "result_traj": res_traj,
#     }


# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 1: annotated Reacher frame
# # ═════════════════════════════════════════════════════════════════════════════

# def make_annotated_frame(save_path, img_size=256):
#     env = make_env()
#     qpos = np.array([-np.pi / 2, -np.pi / 2])
#     img = render_at(env, qpos, TARGET, height=img_size, width=img_size)

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.imshow(img.transpose(1, 2, 0))

#     sh = (128, 128)
#     el = (128, 178)

#     arc1 = patches.Arc((sh[0] + 4, sh[1]), 46, 46, angle=0,
#                        theta1=0, theta2=90, color="#22cc22", linewidth=3)
#     ax.add_patch(arc1)
#     ax.annotate(r"$z_0$", xy=(sh[0] + 32, sh[1] + 32),
#                 fontsize=20, fontweight="bold", color="#22cc22")

#     arc2 = patches.Arc((el[0] - 4, el[1] - 4), 46, 46, angle=0,
#                        theta1=180, theta2=270, color="#ff8800", linewidth=3)
#     ax.add_patch(arc2)
#     ax.annotate(r"$z_1$", xy=(el[0] - 40, el[1] - 30),
#                 fontsize=20, fontweight="bold", color="#ff8800")

#     ax.plot(*sh, "o", color="#22cc22", markersize=8,
#             markeredgecolor="white", markeredgewidth=1.5)
#     ax.plot(*el, "o", color="#ff8800", markersize=8,
#             markeredgecolor="white", markeredgewidth=1.5)
#     ax.set_xticks([]); ax.set_yticks([])
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}")


# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 2: planning_demo.png — 3-row image grid with ghost overlay
# # ═════════════════════════════════════════════════════════════════════════════

# def _straight_latent_plan(z_endpoints, n_steps):
#     alphas = np.linspace(0, 1, n_steps)
#     return np.stack([(1 - a) * z_endpoints[0] + a * z_endpoints[1]
#                      for a in alphas])


# def _nn_retrieve(plan_z, gallery_z, gallery_display):
#     out = []
#     for pz in plan_z:
#         idx = int(np.linalg.norm(gallery_z - pz, axis=1).argmin())
#         out.append(gallery_display[idx])
#     return out


# def make_planning_figure(env, ctx, save_path, n_steps=8,
#                          qpos_start=None, ghost_alpha=0.3):
#     """True, OU retrieval, traj retrieval. Rows 2-3 blend each retrieved frame
#     with the corresponding true frame at weight `ghost_alpha`."""
#     qpos_goal, _ = solve_ik_grid(env, TARGET)
#     if qpos_start is None:
#         qpos_start = np.array([-3 / 4 * np.pi, 1 / 4 * np.pi])

#     alphas = np.linspace(0, 1, n_steps)
#     qpos_traj = np.array([(1 - a) * qpos_start + a * qpos_goal for a in alphas])
#     true_imgs = [render_at(env, q, TARGET) for q in qpos_traj]

#     def retrieval_row(enc, mn, sd, gallery_z):
#         z_ends = encode_images(enc, [true_imgs[0], true_imgs[-1]],
#                                mn, sd, ctx["device"])
#         plan_z = _straight_latent_plan(z_ends, n_steps)
#         retrieved = _nn_retrieve(plan_z, gallery_z, ctx["gallery_display"])
#         # Pin endpoints so Start/Goal columns are identical across rows.
#         retrieved[0]  = true_imgs[0]
#         retrieved[-1] = true_imgs[-1]
#         return retrieved

#     ou_imgs   = retrieval_row(ctx["enc_ou"],   ctx["mn_ou"],   ctx["sd_ou"],
#                               ctx["gallery_z_ou"])
#     traj_imgs = retrieval_row(ctx["enc_traj"], ctx["mn_traj"], ctx["sd_traj"],
#                               ctx["gallery_z_traj"])

#     def blend(retrieved, alpha=ghost_alpha):
#         return [np.clip(alpha * t + (1 - alpha) * r, 0.0, 1.0)
#                 for r, t in zip(retrieved, true_imgs)]

#     rows = [
#         ("True\ntrajectory",                                                  true_imgs),
#         (f"Gaussian\n(R²={ctx['result_ou']['r2_hz']:.2f})",                   blend(ou_imgs)),
#         (f"Trajectory\n(R²={ctx['result_traj']['r2_hz']:.2f})",               blend(traj_imgs)),
#     ]

#     fig, axes = plt.subplots(3, n_steps, figsize=(2.0 * n_steps, 6.0))
#     for r, (label, imgs) in enumerate(rows):
#         for c, im in enumerate(imgs):
#             show_img(axes[r, c], im)
#         axes[r, 0].text(-0.25, 0.5, label,
#                         transform=axes[r, 0].transAxes,
#                         fontsize=12, fontweight="bold",
#                         ha="right", va="center")
#         border(axes[r, 0],  "#22cc22")
#         border(axes[r, -1], "#dd2222")

#     axes[0, 0].set_title("Start", color="#22cc22", fontsize=13, fontweight="bold")
#     axes[0, -1].set_title("Goal", color="#dd2222", fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}")


# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 3: planning_scatter.png — 3x3 scatter grid
# # ═════════════════════════════════════════════════════════════════════════════

# TRAJ_COLORS = ["#1a1a1a", "#0072b2", "#cc79a7"]
# SPACES = ["true", "ou", "traj"]
# SPACE_TITLES = ["True (θ-space)", "Gaussian latent", "Trajectory latent"]


# def make_scatter_figure(env, ctx, save_path, n_steps=8):
#     """Three rows:
#        0. Gallery embedding in each space (polar-colored).
#        1. Three straight-in-θ trajectories, as they appear in each space.
#        2. Straight-in-OU and straight-in-traj plans (decoded via kNN for the
#           True panel), as they appear in each space.
#     """
#     qpos_goal, _ = solve_ik_grid(env, TARGET)
#     qpos_starts = [
#         np.array([-3 / 4 * np.pi,  1 / 4 * np.pi]),
#         np.array([ 1 / 4 * np.pi,  1 / 2 * np.pi]),
#         np.array([-1 / 2 * np.pi, -3 / 4 * np.pi]),
#     ]
#     alphas = np.linspace(0, 1, n_steps)

#     # Row 1 data: straight θ-line → encoded in each model.
#     multi_trajs = []
#     for qs in qpos_starts:
#         qpos_path = np.array([(1 - a) * qs + a * qpos_goal for a in alphas])
#         imgs = [render_at(env, q, TARGET) for q in qpos_path]
#         z_ou   = encode_images(ctx["enc_ou"],   imgs,
#                                 ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
#         z_traj = encode_images(ctx["enc_traj"], imgs,
#                                 ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
#         multi_trajs.append({"theta": qpos_path, "z_ou": z_ou, "z_traj": z_traj})

#     # Row 2 data: plan straight in each model, decode to θ via kNN.
#     dec_ou   = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
#         ctx["gallery_z_ou"],   ctx["gallery_angles"])
#     dec_traj = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
#         ctx["gallery_z_traj"], ctx["gallery_angles"])

#     multi_modelplan = []
#     for traj in multi_trajs:
#         plan_ou_z   = _straight_latent_plan(
#             np.array([traj["z_ou"][0],   traj["z_ou"][-1]]),   n_steps)
#         plan_traj_z = _straight_latent_plan(
#             np.array([traj["z_traj"][0], traj["z_traj"][-1]]), n_steps)

#         theta_from_ou   = dec_ou.predict(plan_ou_z)
#         theta_from_traj = dec_traj.predict(plan_traj_z)
#         theta_from_ou[0],   theta_from_ou[-1]   = traj["theta"][0], traj["theta"][-1]
#         theta_from_traj[0], theta_from_traj[-1] = traj["theta"][0], traj["theta"][-1]

#         # To display the OU plan in the traj panel (and vice versa), re-render
#         # the decoded θ and re-encode.
#         imgs_from_ou   = [render_at(env, q, TARGET) for q in theta_from_ou]
#         imgs_from_traj = [render_at(env, q, TARGET) for q in theta_from_traj]
#         z_traj_from_ou   = encode_images(ctx["enc_traj"], imgs_from_ou,
#                                           ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
#         z_ou_from_traj   = encode_images(ctx["enc_ou"],   imgs_from_traj,
#                                           ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])

#         multi_modelplan.append({
#             "true_from_ou":   theta_from_ou,
#             "true_from_traj": theta_from_traj,
#             "ou_from_ou":     plan_ou_z,         # literally straight in OU
#             "traj_from_traj": plan_traj_z,       # literally straight in traj
#             "traj_from_ou":   z_traj_from_ou,
#             "ou_from_traj":   z_ou_from_traj,
#         })

#     # Accessors
#     def gallery(space):
#         return {"true": ctx["gallery_angles"],
#                 "ou":   ctx["gallery_2d_ou"],
#                 "traj": ctx["gallery_2d_traj"]}[space]

#     def row1_coords(space, traj):
#         return {"true": traj["theta"],
#                 "ou":   traj["z_ou"],
#                 "traj": traj["z_traj"]}[space]

#     def row2_coords(space, mp):
#         """Return (solid, dashed) = (planned-in-OU, planned-in-traj), in `space`."""
#         if space == "true":
#             return mp["true_from_ou"], mp["true_from_traj"]
#         if space == "ou":
#             return mp["ou_from_ou"],   mp["ou_from_traj"]
#         if space == "traj":
#             return mp["traj_from_ou"], mp["traj_from_traj"]

#     # Per-column extent (shared across all 3 rows of that column)
#     col_extents = []
#     for space in SPACES:
#         g = gallery(space)
#         row1 = [row1_coords(space, t) for t in multi_trajs]
#         row2_flat = [p for mp in multi_modelplan
#                        for p in row2_coords(space, mp)]
#         col_extents.append(square_extent(g, *row1, *row2_flat))

#     # Compose
#     fig = plt.figure(figsize=(14, 14))
#     gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.08,
#                           top=0.95, bottom=0.03, left=0.07, right=0.99)
#     faint = 0.35 * ctx["gallery_colors"] + 0.65
#     row_titles = ["Embedding", "Straight in true", "Straight in model"]

#     for col_idx, space in enumerate(SPACES):
#         g = gallery(space)
#         xlim, ylim = col_extents[col_idx]

#         # Row 0
#         ax = fig.add_subplot(gs[0, col_idx])
#         ax.scatter(g[:, 0], g[:, 1], c=ctx["gallery_colors"],
#                    s=5, alpha=0.6, linewidths=0)
#         ax.set_xlim(xlim); ax.set_ylim(ylim)
#         ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
#         ax.set_title(SPACE_TITLES[col_idx], fontsize=13, fontweight="bold")
#         if col_idx == 0:
#             ax.set_ylabel(row_titles[0], fontsize=13, fontweight="bold")

#         # Row 1
#         ax = fig.add_subplot(gs[1, col_idx])
#         ax.scatter(g[:, 0], g[:, 1], c=faint, s=4, alpha=0.5,
#                    linewidths=0, zorder=1)
#         for t_idx, traj in enumerate(multi_trajs):
#             c = row1_coords(space, traj)
#             color = TRAJ_COLORS[t_idx]
#             ax.plot(c[:, 0], c[:, 1], "-", color=color, lw=2.2, zorder=3)
#             ax.scatter(c[:, 0], c[:, 1], c=color, s=22,
#                        ec="white", lw=0.7, zorder=4)
#             ax.scatter(c[0, 0], c[0, 1], c=color, s=110, marker="o",
#                        ec="white", lw=1.5, zorder=5)
#         g_goal = row1_coords(space, multi_trajs[0])[-1]
#         ax.scatter(g_goal[0], g_goal[1], c="#dd2222", s=180, marker="*",
#                    ec="white", lw=1.5, zorder=6)
#         ax.set_xlim(xlim); ax.set_ylim(ylim)
#         ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
#         if col_idx == 0:
#             ax.set_ylabel(row_titles[1], fontsize=13, fontweight="bold")

#         # Row 2: two lines per start (solid = from OU, dashed = from traj)
#         ax = fig.add_subplot(gs[2, col_idx])
#         ax.scatter(g[:, 0], g[:, 1], c=faint, s=4, alpha=0.5,
#                    linewidths=0, zorder=1)
#         for t_idx, mp in enumerate(multi_modelplan):
#             c_ou, c_traj = row2_coords(space, mp)
#             color = TRAJ_COLORS[t_idx]
#             for c_path, ls, alpha in [(c_ou,   "-",  1.0),
#                                        (c_traj, "--", 0.85)]:
#                 ax.plot(c_path[:, 0], c_path[:, 1], ls, color=color,
#                         lw=2.0, alpha=alpha, zorder=3)
#                 ax.scatter(c_path[:, 0], c_path[:, 1], c=color, s=18,
#                            ec="white", lw=0.6, alpha=alpha, zorder=4)
#             ax.scatter(c_ou[0, 0], c_ou[0, 1], c=color, s=110,
#                        marker="o", ec="white", lw=1.5, zorder=5)
#         g_goal = row2_coords(space, multi_modelplan[0])[0][-1]
#         ax.scatter(g_goal[0], g_goal[1], c="#dd2222", s=180, marker="*",
#                    ec="white", lw=1.5, zorder=6)
#         ax.set_xlim(xlim); ax.set_ylim(ylim)
#         ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
#         if col_idx == 0:
#             ax.set_ylabel(row_titles[2], fontsize=13, fontweight="bold")

#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}")


# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 4: planning_quantitative.png — box plots
# # ═════════════════════════════════════════════════════════════════════════════

# def _action_ratio(theta):
#     """(N-1) · Σ‖Δθ‖² / ‖θ_N − θ_0‖²  ≥ 1  (Cauchy–Schwarz)."""
#     chord_sq = float(np.sum((theta[-1] - theta[0]) ** 2))
#     step_sq  = float(np.sum(np.diff(theta, axis=0) ** 2))
#     return (len(theta) - 1) * step_sq / max(chord_sq, 1e-12)


# def _tracking_error(theta, theta_opt):
#     chord = float(np.linalg.norm(theta_opt[-1] - theta_opt[0]))
#     return float(np.linalg.norm(theta - theta_opt, axis=1).mean()
#                  / max(chord, 1e-12))


# def make_quantitative_figure(env, ctx, save_path, n_steps=8,
#                              K=30, k_nn=5, margin=0.25, seed=0):
#     """For K random (start, goal) pairs well inside [−π, π]:
#           - plan straight in each latent between encoded endpoints
#           - decode to θ̂ via kNN on (gallery_z, gallery_angles)
#           - compare θ̂ to θ_opt = straight θ-line.
#     """
#     gallery_angles = ctx["gallery_angles"]
#     alphas = np.linspace(0, 1, n_steps)
#     rng = np.random.default_rng(seed)

#     inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
#     inside_idx = np.where(inside)[0]
#     print(f"Endpoint pool: {len(inside_idx)} / {len(gallery_angles)} "
#           f"within ±{np.pi - margin:.2f}")
#     pair_idx = np.stack([
#         rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
#     ])

#     dec_ou   = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_ou"],   gallery_angles)
#     dec_traj = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_traj"], gallery_angles)

#     action_data   = {k: [] for k in SPACES}
#     tracking_data = {k: [] for k in SPACES}

#     for i, j in pair_idx:
#         theta_0, theta_N = gallery_angles[i], gallery_angles[j]
#         if np.linalg.norm(theta_N - theta_0) < 1e-4:
#             continue

#         theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])
#         img_0 = render_at(env, theta_0, TARGET)
#         img_N = render_at(env, theta_N, TARGET)

#         z_ou   = encode_images(ctx["enc_ou"],   [img_0, img_N],
#                                 ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
#         z_traj = encode_images(ctx["enc_traj"], [img_0, img_N],
#                                 ctx["mn_traj"], ctx["sd_traj"], ctx["device"])

#         plan_ou   = _straight_latent_plan(z_ou,   n_steps)
#         plan_traj = _straight_latent_plan(z_traj, n_steps)
#         theta_hat_ou   = dec_ou.predict(plan_ou)
#         theta_hat_traj = dec_traj.predict(plan_traj)
#         theta_hat_ou[0],   theta_hat_ou[-1]   = theta_0, theta_N
#         theta_hat_traj[0], theta_hat_traj[-1] = theta_0, theta_N

#         for name, theta_hat in [("true", theta_opt),
#                                  ("ou",   theta_hat_ou),
#                                  ("traj", theta_hat_traj)]:
#             action_data[name].append(_action_ratio(theta_hat))
#             tracking_data[name].append(_tracking_error(theta_hat, theta_opt))

#     # Plot
#     fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=0.4 * np.array((10, 5)))
#     labels = ["Optimum", "Gaussian", "Trajectory"]
#     colors = ["#888888", "#0072b2", "#cc79a7"]

#     for ax, data, title, ideal in [
#         (ax_a, action_data,   "Path length",   1.0),
#         (ax_t, tracking_data, "Control effort", 0.0),
#     ]:
#         values = [data[k] for k in SPACES]
#         bp = ax.boxplot(values, positions=np.arange(3), widths=0.55,
#                         patch_artist=True, showfliers=True,
#                         medianprops=dict(color="black", lw=1.8),
#                         flierprops=dict(marker="o", markersize=3,
#                                         markerfacecolor="#444",
#                                         markeredgecolor="none", alpha=0.6))
#         for patch, c in zip(bp["boxes"], colors):
#             patch.set_facecolor(c); patch.set_edgecolor("black")
#             patch.set_linewidth(0.8)
#         ax.axhline(ideal, ls="--", color="gray", lw=1, alpha=0.7)
#         ax.set_xticks(np.arange(3))
#         ax.set_xticklabels(labels, rotation=45, fontsize=8)
#         ax.set_ylabel(title)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.grid()
#         if title == "Path length":
#             ax.set_yscale("log")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}  (K={K} pairs, kNN decoder with k={k_nn})")






# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 6 (Experiment A): O(n)-invariant quadratic cost, computed over the
# # SAME K start-goal pairs as the existing quantitative figure. Direct numerical
# # instance of Cor. 4.4: for an O(n)-invariant cost, value in ẑ-space equals
# # value in z-space up to the approx-identifiability residual.
# # ═════════════════════════════════════════════════════════════════════════════

# def _quadratic_cost(theta_path, theta_goal, w_state=1.0, w_action=1.0):
#     """
#     J(path) = Σ w_state · ||θ_t - θ_goal||²  +  Σ w_action · ||θ_{t+1} - θ_t||².
#     Both terms are O(n)-invariant: ||Rθ - Rθ*|| = ||θ - θ*|| for R ∈ O(n).
#     """
#     theta_path = np.asarray(theta_path)
#     theta_goal = np.asarray(theta_goal)
#     state_cost  = float(np.sum(np.sum((theta_path - theta_goal) ** 2, axis=1)))
#     action_cost = float(np.sum(np.sum(np.diff(theta_path, axis=0) ** 2, axis=1)))
#     return w_state * state_cost + w_action * action_cost


# def make_invariant_cost_figure(env, ctx, save_path, n_steps=8,
#                                K=30, k_nn=5, margin=0.25, seed=0,
#                                w_state=1.0, w_action=1.0):
#     """
#     For K random (start, goal) pairs:
#         - oracle (straight in θ),
#         - OU latent plan decoded via kNN,
#         - Traj latent plan decoded via kNN,
#     compute quadratic cost (O(n)-invariant), plot cost / oracle cost as ratios.
#     Ideal ratio = 1 for any encoder satisfying Cor. 4.4; larger ratios measure
#     the approx-identifiability residual quantitatively.
#     """
#     gallery_angles = ctx["gallery_angles"]
#     alphas = np.linspace(0, 1, n_steps)
#     rng = np.random.default_rng(seed)

#     inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
#     inside_idx = np.where(inside)[0]
#     pair_idx = np.stack([
#         rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
#     ])

#     dec_ou   = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_ou"],   gallery_angles)
#     dec_traj = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_traj"], gallery_angles)

#     cost_oracle, cost_ou, cost_traj = [], [], []

#     for i, j in pair_idx:
#         theta_0, theta_N = gallery_angles[i], gallery_angles[j]
#         if np.linalg.norm(theta_N - theta_0) < 1e-4:
#             continue
#         theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])

#         img_0 = render_at(env, theta_0, TARGET)
#         img_N = render_at(env, theta_N, TARGET)
#         z_ou   = encode_images(ctx["enc_ou"],   [img_0, img_N],
#                                 ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
#         z_traj = encode_images(ctx["enc_traj"], [img_0, img_N],
#                                 ctx["mn_traj"], ctx["sd_traj"], ctx["device"])

#         plan_ou   = _straight_latent_plan(z_ou,   n_steps)
#         plan_traj = _straight_latent_plan(z_traj, n_steps)
#         theta_hat_ou   = dec_ou.predict(plan_ou)
#         theta_hat_traj = dec_traj.predict(plan_traj)
#         theta_hat_ou[0],   theta_hat_ou[-1]   = theta_0, theta_N
#         theta_hat_traj[0], theta_hat_traj[-1] = theta_0, theta_N

#         cost_oracle.append(_quadratic_cost(theta_opt,      theta_N,
#                                            w_state, w_action))
#         cost_ou    .append(_quadratic_cost(theta_hat_ou,   theta_N,
#                                            w_state, w_action))
#         cost_traj  .append(_quadratic_cost(theta_hat_traj, theta_N,
#                                            w_state, w_action))

#     cost_oracle = np.array(cost_oracle)
#     cost_ou     = np.array(cost_ou)
#     cost_traj   = np.array(cost_traj)
#     ratio_ou   = cost_ou   / cost_oracle
#     ratio_traj = cost_traj / cost_oracle

#     # Plot ratios (log y). Ideal ratio = 1 corresponds to exact corollary.
#     fig, ax = plt.subplots(figsize=(4.5, 3.5))
#     labels = ["Optimum", "Gaussian", "Trajectory"]
#     colors = ["#888888", "#0072b2", "#cc79a7"]
#     values = [np.ones_like(ratio_ou), ratio_ou, ratio_traj]

#     bp = ax.boxplot(values, positions=np.arange(3), widths=0.55,
#                     patch_artist=True, showfliers=True,
#                     medianprops=dict(color="black", lw=1.8),
#                     flierprops=dict(marker="o", markersize=3,
#                                     markerfacecolor="#444",
#                                     markeredgecolor="none", alpha=0.6))
#     for patch, c in zip(bp["boxes"], colors):
#         patch.set_facecolor(c); patch.set_edgecolor("black")
#         patch.set_linewidth(0.8)
#     ax.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.7)
#     ax.set_xticks(np.arange(3))
#     ax.set_xticklabels(labels, rotation=45, fontsize=8)
#     ax.set_ylabel("Quadratic cost / oracle")
#     ax.set_yscale("log")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid()

#     # Also print summary numbers
#     print(f"\n[Invariant-cost]  ratio_ou   median={np.median(ratio_ou):.3f}  "
#           f"mean={np.mean(ratio_ou):.3f}")
#     print(f"[Invariant-cost]  ratio_traj median={np.median(ratio_traj):.3f}  "
#           f"mean={np.mean(ratio_traj):.3f}")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}  (K={len(ratio_ou)} pairs, quadratic "
#           f"O(n)-invariant cost)")


# # ═════════════════════════════════════════════════════════════════════════════
# # Figure 7 (Experiment B): LQR covariance test. Cor. 4.4 predicts that for an
# # O(n)-invariant quadratic cost, the Riccati equation transforms covariantly
# # under Q, and the optimal value V*(z_0) equals V̂*(h(z_0)). This test uses
# # synthetic linear dynamics so that we can solve DARE analytically in both
# # coordinate systems and compare V*.
# #
# # We stress: the LINEAR DYNAMICS here are synthetic; we are testing whether
# # the ENCODER's residual rotation preserves LQR value, not whether the real
# # reacher is linear. This isolates Cor. 4.4's covariance claim cleanly.
# # ═════════════════════════════════════════════════════════════════════════════


# def _linear_regress_encoder(z_gallery, h_gallery):
#     """
#     Fit ẑ = M z + b via OLS. Returns (M, b).  For an ideal Cor. 4.4 encoder
#     this is h(z) = Q z, so M ≈ Q (orthogonal) and b ≈ 0.
#     """
#     # augment with bias, solve via lstsq
#     Z = np.column_stack([z_gallery, np.ones(len(z_gallery))])
#     Mb, *_ = np.linalg.lstsq(Z, h_gallery, rcond=None)
#     M = Mb[:-1].T          # (n_out, n_in)
#     b = Mb[-1]             # (n_out,)
#     return M, b


# def _solve_dare_lqr(A, B, W, R):
#     """
#     Infinite-horizon discrete-time LQR. Returns (P, K) where V*(z) = z^T P z.
#     """
#     P = solve_discrete_are(A, B, W, R)
#     gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
#     return P, gain


# def make_lqr_equivalence_figure(ctx, save_path, n_samples=200,
#                                 noise_level=0.05, seed=0):
#     """
#     Synthetic LQR test of Cor. 4.4 covariance claim.
#       - pick linear dynamics A, B in θ-space (small random rotation-like A)
#       - pick O(n)-invariant quadratic cost W=I, W_T=I, R=I
#       - solve DARE in θ-space:  P, V*(z)  = z^T P z
#       - solve DARE in ẑ-space:  A_hat = M A M^{-1}, B_hat = M B
#         (pushforward under the fitted linear map M ≈ Q)
#       - compare V̂*(ẑ) vs V*(z) for n_samples random initial states.
#     If Cor. 4.4 holds: V̂*(h(z)) = V*(z) exactly (up to approx-identifiability).
#     """
#     rng = np.random.default_rng(seed)
#     gallery_angles = ctx["gallery_angles"]
#     n = 2

#     # Fit the effective linear map from true latent to each encoder's output
#     M_ou,   b_ou   = _linear_regress_encoder(
#         gallery_angles, ctx["gallery_z_ou"])
#     M_traj, b_traj = _linear_regress_encoder(
#         gallery_angles, ctx["gallery_z_traj"])

#     # Orthogonality diagnostic
#     def _orth_err(M):
#         MtM = M.T @ M
#         return float(np.linalg.norm(MtM - np.eye(n), 'fro'))
#     print(f"[LQR]  ||M_ou^T M_ou - I||_F   = {_orth_err(M_ou):.4f}  "
#           f"(ideal: 0 for exact Cor. 4.4)")
#     print(f"[LQR]  ||M_traj^T M_traj - I||_F = {_orth_err(M_traj):.4f}")

#     # Synthetic linear dynamics in true θ-space. A close to identity with
#     # slight coupling — representative of linearized-Reacher near a fixed point.
#     A_true = np.array([[0.95,  0.05],
#                        [-0.03, 0.92]])
#     B_true = np.array([[1.0, 0.0],
#                        [0.0, 1.0]]) * 0.3
#     # Costs: unit penalties, rotation-invariant.
#     W, W_T, R = np.eye(n), np.eye(n), np.eye(n)

#     # DARE in true space
#     P_true, _ = _solve_dare_lqr(A_true, B_true, W, R)

#     # Pushforward dynamics in each encoder's space: A_hat = M A M^{-1}, etc.
#     def _pushforward(M, A, B):
#         M_inv = np.linalg.pinv(M)
#         return M @ A @ M_inv, M @ B

#     A_ou,   B_ou   = _pushforward(M_ou,   A_true, B_true)
#     A_traj, B_traj = _pushforward(M_traj, A_true, B_true)

#     # In ẑ-space, cost W_hat = M W M^T (covariant with rotation).
#     # For W = I and exact orthogonal M: W_hat = I, identical problem.
#     W_ou      = M_ou   @ W   @ M_ou.T
#     W_T_ou    = M_ou   @ W_T @ M_ou.T
#     W_traj    = M_traj @ W   @ M_traj.T
#     W_T_traj  = M_traj @ W_T @ M_traj.T

#     P_ou, _   = _solve_dare_lqr(A_ou,   B_ou,   W_ou,   R)
#     P_traj, _ = _solve_dare_lqr(A_traj, B_traj, W_traj, R)

#     # Sample initial θ states; compare V*(θ) to V̂*(M θ + b) for each encoder.
#     idx = rng.choice(len(gallery_angles), n_samples, replace=False)
#     z0 = gallery_angles[idx]                                    # (N, 2)
#     zhat_ou   = z0 @ M_ou.T   + b_ou                            # (N, 2)
#     zhat_traj = z0 @ M_traj.T + b_traj

#     def _val(P, z):   # z^T P z per row
#         return np.einsum("ni,ij,nj->n", z, P, z)

#     V_true   = _val(P_true,   z0)
#     V_ou     = _val(P_ou,     zhat_ou)
#     V_traj   = _val(P_traj,   zhat_traj)

#     # The corollary predicts V_ou ≈ V_true, V_traj ≠ V_true.
#     err_ou   = np.abs(V_ou   - V_true) / (np.abs(V_true) + 1e-9)
#     err_traj = np.abs(V_traj - V_true) / (np.abs(V_true) + 1e-9)
#     print(f"[LQR]  |V̂ - V*| / |V*|   OU   median={np.median(err_ou):.4f}  "
#           f"mean={np.mean(err_ou):.4f}")
#     print(f"[LQR]  |V̂ - V*| / |V*|   Traj median={np.median(err_traj):.4f}  "
#           f"mean={np.mean(err_traj):.4f}")

#     # Two-panel figure: scatter V̂ vs V*, relative-error boxplot.
#     fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))

#     lim = (min(V_true.min(), V_ou.min(), V_traj.min()),
#            max(V_true.max(), V_ou.max(), V_traj.max()))
#     ax_s.plot(lim, lim, "k--", lw=1, alpha=0.6, label="ideal ($\\hat V=V^*$)")
#     ax_s.scatter(V_true, V_ou,   s=14, alpha=0.7, c="#0072b2",
#                  edgecolors="none", label="Gaussian")
#     ax_s.scatter(V_true, V_traj, s=14, alpha=0.7, c="#cc79a7",
#                  edgecolors="none", label="Trajectory")
#     ax_s.set_xlabel("True-latent LQR value $V^*(z_0)$")
#     ax_s.set_ylabel("Learned-latent value $\\hat V^*(h(z_0))$")
#     ax_s.set_aspect("equal", adjustable="box")
#     ax_s.grid(alpha=0.3); ax_s.legend(fontsize=8)

#     ax_b.boxplot([err_ou, err_traj], positions=[0, 1], widths=0.55,
#                  patch_artist=True,
#                  medianprops=dict(color="black", lw=1.8),
#                  flierprops=dict(marker="o", markersize=3,
#                                  markerfacecolor="#444",
#                                  markeredgecolor="none", alpha=0.5))
#     for patch, c in zip(ax_b.patches, ["#0072b2", "#cc79a7"]):
#         patch.set_facecolor(c); patch.set_edgecolor("black")
#     ax_b.set_xticks([0, 1])
#     ax_b.set_xticklabels(["Gaussian", "Trajectory"], fontsize=9)
#     ax_b.set_ylabel("$|\\hat V^* - V^*| / |V^*|$")
#     ax_b.set_yscale("log")
#     ax_b.axhline(1.0, ls=":", color="gray", lw=1, alpha=0.6)
#     ax_b.grid(alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}")



# # ═════════════════════════════════════════════════════════════════════════════
# # Figure: control cost boxplot (best-run summary)  +
# #         control cost vs R² scatter (all runs)
# #
# # Single two-panel figure for the main text. Left: reproduces the current
# # invariant_cost boxplot with updated naming. Right: across all reacher runs,
# # control cost (normalized by oracle) vs. R²(h→z), colored by OU vs Traj.
# #
# # Replaces make_invariant_cost_figure. The boxplot part is identical except
# # for label strings.
# # ═════════════════════════════════════════════════════════════════════════════


# def _compute_control_cost_for_encoder(enc, mean, std, gallery_u8,
#                                       gallery_angles, env, device,
#                                       K=30, n_steps=8, k_nn=5, margin=0.25,
#                                       seed=0, w_state=1.0, w_action=1.0):
#     """
#     Mean control-cost ratio (vs oracle) over K random start-goal pairs.
#     Returns (mean_ratio, raw_ratios_array).
#     """
#     gnorm = torch.from_numpy(normalize_uint8(gallery_u8, mean, std))
#     gallery_z = encode_batched(enc, gnorm, device)
#     alphas = np.linspace(0, 1, n_steps)
#     rng = np.random.default_rng(seed)

#     inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
#     inside_idx = np.where(inside)[0]
#     pair_idx = np.stack([
#         rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
#     ])
#     decoder = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         gallery_z, gallery_angles)

#     ratios = []
#     for i, j in pair_idx:
#         theta_0, theta_N = gallery_angles[i], gallery_angles[j]
#         if np.linalg.norm(theta_N - theta_0) < 1e-4:
#             continue
#         theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])
#         img_0 = render_at(env, theta_0, TARGET)
#         img_N = render_at(env, theta_N, TARGET)
#         z_ends = encode_images(enc, [img_0, img_N], mean, std, device)
#         plan_z = _straight_latent_plan(z_ends, n_steps)
#         theta_hat = decoder.predict(plan_z)
#         theta_hat[0], theta_hat[-1] = theta_0, theta_N

#         cost_oracle = _quadratic_cost(theta_opt,  theta_N, w_state, w_action)
#         cost_enc    = _quadratic_cost(theta_hat,  theta_N, w_state, w_action)
#         if cost_oracle > 1e-9:
#             ratios.append(cost_enc / cost_oracle)
#     ratios = np.array(ratios)
#     return float(np.mean(ratios)), ratios


# def _collect_control_cost_across_runs(results_dir, data_root, device,
#                                       gallery_size=10000, K=30, n_steps=8,
#                                       k_nn=5, seed=0, cache_path=None):
#     """
#     For every reacher run with a checkpoint, compute mean control-cost ratio
#     and pair with R² values. Returns list of dicts, one per run. Cached to
#     JSON so reruns are instant.
#     """
#     if cache_path is not None and Path(cache_path).exists():
#         with open(cache_path) as f:
#             out = json.load(f)
#         print(f"Loaded {len(out)} cached control-cost entries from {cache_path}")
#         return out

#     gallery_u8 = np.load(os.path.join(data_root, "eval", "img.npy"))[:gallery_size]
#     gallery_angles = try_load_true_angles(os.path.join(data_root, "eval"),
#                                           gallery_size)
#     env = make_env()

#     out = []
#     result_paths = sorted(Path(results_dir).rglob("result.json"))
#     for idx, rp in enumerate(tqdm(result_paths, desc="control-cost across runs")):
#         with open(rp) as f:
#             r = json.load(f)
#         ckpt = rp.parent / "checkpoint.pt"
#         if not ckpt.exists():
#             continue
#         enc, mean, std, _ = load_encoder(ckpt, device)
#         cost_mean, _ = _compute_control_cost_for_encoder(
#             enc, mean, std, gallery_u8, gallery_angles, env, device,
#             K=K, n_steps=n_steps, k_nn=k_nn, seed=seed)
#         out.append({
#             "run_name":  r["run_name"],
#             "type":      r.get("type", "ou" if r.get("rho") is not None
#                                        else "traj"),
#             "rho":       r.get("rho"),
#             "delta":     r.get("delta"),
#             "lamb":      r.get("lamb"),
#             "seed":      r.get("seed"),
#             "r2_zh":     r.get("r2_zh"),
#             "r2_hz":     r.get("r2_hz"),
#             "r2_hz_dim0": r.get("r2_hz_per_dim", [None, None])[0],
#             "r2_hz_dim1": (r.get("r2_hz_per_dim", [None, None])[1]
#                           if len(r.get("r2_hz_per_dim", [])) > 1 else None),
#             "control_cost_ratio_mean": cost_mean,
#         })
#         # print(f"[{idx+1}/{len(result_paths)}] {r['run_name']}  "
#         #       f"R²(h→z)={r.get('r2_hz', float('nan')):.3f}  "
#         #       f"cost/oracle={cost_mean:.3f}")
#     if cache_path is not None:
#         with open(cache_path, "w") as f:
#             json.dump(out, f, indent=2)
#         print(f"Cached {len(out)} entries to {cache_path}")
#     return out


# def make_control_cost_figure(env, ctx, save_path,
#                              results_dir, data_root, device,
#                              n_steps=8, K=30, k_nn=5, margin=0.25, seed=0,
#                              w_state=1.0, w_action=1.0,
#                              cache_path=None, gallery_size=10000):
#     """
#     Two-panel figure for main text.
#     Left: control-cost boxplot (Optimum / Gaussian / Trajectory) for the best
#           OU and best Traj encoders, over K random start-goal pairs.
#     Right: scatter of mean control-cost ratio vs R²(h→z) across ALL reacher
#            runs, colored by OU vs Traj, with Pearson r in legend.
#     """
#     gallery_angles = ctx["gallery_angles"]
#     alphas = np.linspace(0, 1, n_steps)
#     rng = np.random.default_rng(seed)

#     # ── LEFT PANEL: boxplot of best OU / best Traj vs oracle ────────────
#     inside = np.all(np.abs(gallery_angles) < (np.pi - margin), axis=1)
#     inside_idx = np.where(inside)[0]
#     pair_idx = np.stack([
#         rng.choice(inside_idx, size=2, replace=False) for _ in range(K)
#     ])
#     dec_ou   = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_ou"],   gallery_angles)
#     dec_traj = KNeighborsRegressor(n_neighbors=k_nn, weights="distance").fit(
#         ctx["gallery_z_traj"], gallery_angles)

#     cost_oracle, cost_ou, cost_traj = [], [], []
#     for i, j in pair_idx:
#         theta_0, theta_N = gallery_angles[i], gallery_angles[j]
#         if np.linalg.norm(theta_N - theta_0) < 1e-4:
#             continue
#         theta_opt = np.stack([(1 - a) * theta_0 + a * theta_N for a in alphas])
#         img_0 = render_at(env, theta_0, TARGET)
#         img_N = render_at(env, theta_N, TARGET)
#         z_ou   = encode_images(ctx["enc_ou"],   [img_0, img_N],
#                                 ctx["mn_ou"],   ctx["sd_ou"],   ctx["device"])
#         z_traj = encode_images(ctx["enc_traj"], [img_0, img_N],
#                                 ctx["mn_traj"], ctx["sd_traj"], ctx["device"])
#         plan_ou   = _straight_latent_plan(z_ou,   n_steps)
#         plan_traj = _straight_latent_plan(z_traj, n_steps)
#         theta_hat_ou   = dec_ou.predict(plan_ou)
#         theta_hat_traj = dec_traj.predict(plan_traj)
#         theta_hat_ou[0],   theta_hat_ou[-1]   = theta_0, theta_N
#         theta_hat_traj[0], theta_hat_traj[-1] = theta_0, theta_N
#         cost_oracle.append(_quadratic_cost(theta_opt,      theta_N, w_state, w_action))
#         cost_ou    .append(_quadratic_cost(theta_hat_ou,   theta_N, w_state, w_action))
#         cost_traj  .append(_quadratic_cost(theta_hat_traj, theta_N, w_state, w_action))

#     cost_oracle = np.array(cost_oracle)
#     cost_ou     = np.array(cost_ou)
#     cost_traj   = np.array(cost_traj)
#     ratio_ou   = cost_ou   / cost_oracle
#     ratio_traj = cost_traj / cost_oracle

#     print(f"\n[Control-cost]  ratio_ou   median={np.median(ratio_ou):.3f}  "
#           f"mean={np.mean(ratio_ou):.3f}")
#     print(f"[Control-cost]  ratio_traj median={np.median(ratio_traj):.3f}  "
#           f"mean={np.mean(ratio_traj):.3f}")

#     # ── RIGHT PANEL: scatter across all runs ────────────────────────────
#     all_runs = _collect_control_cost_across_runs(
#         results_dir, data_root, device,
#         gallery_size=gallery_size, K=K, n_steps=n_steps, k_nn=k_nn,
#         seed=seed, cache_path=cache_path)

#     # ── Figure ──────────────────────────────────────────────────────────
#     fig, (ax_box, ax_sc) = plt.subplots(1, 2, figsize=0.7 * np.array((9, 3.8)))

#     # Left: boxplot
#     labels = ["Optimum", "Gaussian", "Trajectory"]
#     colors = ["#888888", "#0072b2", "#cc79a7"]
#     values = [np.ones_like(ratio_ou), ratio_ou, ratio_traj]
#     bp = ax_box.boxplot(values, positions=np.arange(3), widths=0.55,
#                         patch_artist=True, showfliers=True,
#                         medianprops=dict(color="black", lw=1.8),
#                         flierprops=dict(marker="o", markersize=3,
#                                         markerfacecolor="#444",
#                                         markeredgecolor="none", alpha=0.6))
#     for patch, c in zip(bp["boxes"], colors):
#         patch.set_facecolor(c); patch.set_edgecolor("black"); patch.set_linewidth(0.8)
#     ax_box.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.7)
#     ax_box.set_xticks(np.arange(3))
#     ax_box.set_xticklabels(labels, rotation=0, fontsize=9)
#     ax_box.set_ylabel("Control Cost")
#     ax_box.set_yscale("log")
#     ax_box.spines["top"].set_visible(False)
#     ax_box.spines["right"].set_visible(False)
#     ax_box.grid(alpha=0.3)
#     # ax_box.set_title("Best encoders, K={} pairs".format(K), fontsize=10)

#     # Right: scatter
#     ou_runs   = [r for r in all_runs if r["type"] == "ou"
#                  and r["r2_hz"] is not None
#                  and r["control_cost_ratio_mean"] is not None]
#     traj_runs = [r for r in all_runs if r["type"] == "traj"
#                  and r["r2_hz"] is not None
#                  and r["control_cost_ratio_mean"] is not None]

#     def _plot_group(runs, color, marker, label):
#         xs = np.array([r["r2_hz"] for r in runs])
#         ys = np.array([r["control_cost_ratio_mean"] for r in runs])
#         # clip negative R² to 0 for display, but keep for correlation
#         if len(xs) >= 3:
#             r, p = _pearsonr_cc(xs, ys)
#             # lbl = f"{label}  (n={len(xs)},  r={r:+.2f})"
#             lbl = f"{label}"
#         else:
#             # lbl = f"{label}  (n={len(xs)})"
#             lbl = f"{label}"
#         ax_sc.scatter(xs, ys, s=32, alpha=0.75, c=color, marker=marker,
#                       edgecolors="black", linewidths=0.3, label=lbl)

#     _plot_group(ou_runs,   "#0072b2", "o", "OU")
#     _plot_group(traj_runs, "#cc79a7", "s", "Trajectory")
#     ax_sc.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.6)
#     # ax_sc.set_xlabel(r"$R^2(h \to z)$")
#     ax_sc.set_xlabel(r"Linear Identifiability [$R^2$]")
#     ax_sc.set_ylabel("Control Cost")
#     ax_sc.set_yscale("log")
#     ax_sc.grid(alpha=0.3)
#     # ax_sc.legend(fontsize=8, loc="best", framealpha=0.9)
#     # ax_sc.set_title("All runs", fontsize=10)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"Saved {save_path}")




# # ═════════════════════════════════════════════════════════════════════════════
# # Main
# # ═════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--results_dir",  type=str, default="results/reacher")
#     parser.add_argument("--data_root",    type=str, default="data/reacher")
#     parser.add_argument("--out_dir",      type=str, default="figures/reacher")
#     parser.add_argument("--device",       type=str, default="cuda")
#     parser.add_argument("--gallery_size", type=int, default=10000)
#     parser.add_argument("--planning_K", type=int, default=32)
#     args = parser.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Annotated frame: no models needed
#     make_annotated_frame(out_dir / "reacher_annotated.png")

#     # Shared resources for the other three figures
#     ctx = load_planning_context(args.results_dir, args.data_root,
#                                 args.device, args.gallery_size)
#     env = make_env()

#     make_planning_figure(env, ctx, out_dir / "planning_demo.png")
#     make_scatter_figure(env, ctx, out_dir / "planning_scatter.png")
#     # make_quantitative_figure(env, ctx, out_dir / "planning_quantitative.png")


#     # Cor. 4.4: O(n)-invariant cost on K random pairs (Experiment A)
#     make_control_cost_figure(
#         env, ctx,
#         save_path=out_dir / "control_cost.png",
#         results_dir=args.results_dir,
#         data_root=args.data_root,
#         device=args.device,
#         gallery_size=args.gallery_size,
#         K=args.planning_K,
#         cache_path=os.path.join(args.results_dir, "control_cost_cache.json"),
#     )

#     # Cor. 4.4: LQR value equivalence (Experiment B, synthetic dynamics)
#     make_lqr_equivalence_figure(ctx, out_dir / "lqr_equivalence.png")


# if __name__ == "__main__":
#     main()