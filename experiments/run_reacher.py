"""
Train LeJEPA on prerendered Reacher data.

Loads images + latents from disk (output of prerender.py),
sweeps lambda × seed. Final model is always used (no K inits, no loss selection).

Works identically for OU and trajectory data — just point --data_dir
at the right directory.

Usage:
    python run_reacher.py --config configs/reacher.yaml \
        --data_dir data/reacher/ou/rho=0.95
    python run_reacher.py --config configs/reacher.yaml \
        --data_dir data/reacher/traj/delta=16
"""

import argparse, os, json, yaml
import numpy as np
import torch
import torch.nn.functional as F

from lejepa_id.losses import SIGReg, alignment_loss
from lejepa_id.models import make_cnn_encoder
from lejepa_id.metrics import bidirectional_r2

from sklearn.linear_model import LinearRegression
from scipy.linalg import orthogonal_procrustes


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(data_dir):
    """Load prerendered (img, z) pairs. Images stored as uint8."""
    data_dir = str(data_dir)
    z_t = np.load(os.path.join(data_dir, "z_t.npy"))
    z_tp1 = np.load(os.path.join(data_dir, "z_tp1.npy"))
    img_t = np.load(os.path.join(data_dir, "img_t.npy"))
    img_tp1 = np.load(os.path.join(data_dir, "img_tp1.npy"))
    mean = np.load(os.path.join(data_dir, "img_mean.npy"))
    std = np.load(os.path.join(data_dir, "img_std.npy"))
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    print(f"Loaded {len(z_t)} pairs from {data_dir}")
    return z_t, z_tp1, img_t, img_tp1, mean, std, meta


def load_eval(eval_dir):
    """Load prerendered eval set."""
    eval_dir = str(eval_dir)
    z = np.load(os.path.join(eval_dir, "z.npy"))
    img = np.load(os.path.join(eval_dir, "img.npy"))
    mean = np.load(os.path.join(eval_dir, "img_mean.npy"))
    std = np.load(os.path.join(eval_dir, "img_std.npy"))
    print(f"Loaded {len(z)} eval samples")
    return z, img, mean, std


def normalize_uint8(img_uint8, mean, std):
    """Convert uint8 → float32 normalized. mean/std are (3,) arrays."""
    img = img_uint8.astype(np.float32) / 255.0
    img = (img - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-6)
    return img


class ImageDataset(torch.utils.data.Dataset):
    """Normalized float32 image pairs + latents."""
    def __init__(self, img_t, img_tp1, z_t, z_tp1, mean, std):
        self.img_t = torch.from_numpy(normalize_uint8(img_t, mean, std))
        self.img_tp1 = torch.from_numpy(normalize_uint8(img_tp1, mean, std))
        self.z_t = torch.from_numpy(z_t)
        self.z_tp1 = torch.from_numpy(z_tp1)

    def __len__(self):
        return len(self.img_t)

    def __getitem__(self, i):
        return self.img_t[i], self.img_tp1[i], self.z_t[i], self.z_tp1[i]


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(encoder, images, device, batch_size=512):
    encoder.eval()
    embeds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        embeds.append(encoder(batch).cpu())
    return torch.cat(embeds)


def train_one(encoder, loader, eval_data, lamb, cfg, device):
    """Train one encoder. Returns final model (no selection)."""
    sigreg = SIGReg(n_slices=cfg["n_slices"]).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    eval_imgs, eval_z = eval_data

    log = {"align": [], "sigreg": [], "total": [], "z_std": [], "r2_hz": []}

    for epoch in range(cfg["epochs"]):
        encoder.train()
        ep = {k: [] for k in ["align", "sigreg", "total", "z_std"]}

        for img_t, img_tp1, _, _ in loader:
            img_t, img_tp1 = img_t.to(device), img_tp1.to(device)
            z_t = encoder(img_t)
            z_tp1 = encoder(img_tp1)

            h = torch.stack([z_t, z_tp1], dim=0)
            L_align = alignment_loss(h)
            L_sig = sigreg(h)
            loss = lamb * L_sig + (1 - lamb) * L_align

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            ep["total"].append(loss.item())
            ep["align"].append(L_align.item())
            ep["sigreg"].append(L_sig.item())
            with torch.no_grad():
                ep["z_std"].append(z_t.std(0).mean().item())

        scheduler.step()

        # Quick eval (on eval subset, for logging only)
        encoder.eval()
        h_eval = extract_embeddings(encoder, eval_imgs, device)
        _, r2_hz = bidirectional_r2(eval_z, h_eval)

        for k in ep:
            log[k].append(float(np.mean(ep[k])))
        log["r2_hz"].append(r2_hz)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"align={log['align'][-1]:.5f}  "
                  f"sig={log['sigreg'][-1]:.1f}  "
                  f"z_std={log['z_std'][-1]:.3f}  "
                  f"R²={r2_hz:.4f}")

    return log


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def final_eval(encoder, train_imgs, train_z, eval_imgs, eval_z, device):
    """
    Full eval with proper train/test split.
    Fit linear regression on train embeddings, score on eval embeddings.
    """
    h_train = extract_embeddings(encoder, train_imgs, device).numpy()
    h_eval = extract_embeddings(encoder, eval_imgs, device).numpy()
    z_train = train_z.numpy() if isinstance(train_z, torch.Tensor) else train_z
    z_eval = eval_z.numpy() if isinstance(eval_z, torch.Tensor) else eval_z

    # Overall R² (fit on train, score on test)
    reg_hz = LinearRegression().fit(h_train, z_train)
    r2_hz = reg_hz.score(h_eval, z_eval)

    reg_zh = LinearRegression().fit(z_train, h_train)
    r2_zh = reg_zh.score(z_eval, h_eval)

    # Per-dimension R² (fit on train, score on test)
    r2_hz_per = []
    for i in range(z_train.shape[1]):
        reg_i = LinearRegression().fit(h_train, z_train[:, i])
        r2_hz_per.append(reg_i.score(h_eval, z_eval[:, i]))

    # Sin/cos diagnostic (fit on train, score on test)
    z_train_sc = np.column_stack([np.sin(z_train), np.cos(z_train)])
    z_eval_sc = np.column_stack([np.sin(z_eval), np.cos(z_eval)])
    reg_sc = LinearRegression().fit(h_train, z_train_sc)
    r2_sincos = reg_sc.score(h_eval, z_eval_sc)

    # Per-component sin/cos R²
    sincos_names = ["sin_shoulder", "cos_shoulder", "sin_wrist", "cos_wrist"]
    r2_sincos_per = {}
    for i, name in enumerate(sincos_names):
        reg_i = LinearRegression().fit(h_train, z_train_sc[:, i])
        r2_sincos_per[name] = reg_i.score(h_eval, z_eval_sc[:, i])

    # Orthogonality error
    d = min(z_eval.shape[1], h_eval.shape[1])
    Zt = (z_eval[:, :d] - z_eval[:, :d].mean(0)).copy()
    Zl = (h_eval[:, :d] - h_eval[:, :d].mean(0)).copy()
    for Z in [Zt, Zl]:
        cov = np.cov(Z, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-8)
        W = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T
        Z[:] = Z @ W
    R, _ = orthogonal_procrustes(Zl, Zt)
    orth_err = float(np.linalg.norm(Zl @ R - Zt) / np.linalg.norm(Zt))

    return {
        "r2_zh": r2_zh,
        "r2_hz": r2_hz,
        "r2_hz_per_dim": r2_hz_per,
        "r2_sincos": r2_sincos,
        "r2_sincos_per": r2_sincos_per,
        "orth_error": orth_err,
        "linear_map_W": reg_hz.coef_.T,
        "linear_map_b": reg_hz.intercept_,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to prerendered dataset (ou/rho=X or traj/delta=X)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    z_t, z_tp1, img_t, img_tp1, train_mean, train_std, data_meta = \
        load_dataset(args.data_dir)

    eval_dir = os.path.join(cfg["data_root"], "eval")
    z_eval, img_eval_u8, eval_mean, eval_std = load_eval(eval_dir)

    # Normalize with training stats
    dataset = ImageDataset(img_t, img_tp1, z_t, z_tp1, train_mean, train_std)
    eval_imgs = torch.from_numpy(
        normalize_uint8(img_eval_u8, train_mean, train_std))
    eval_z = torch.from_numpy(z_eval)

    # Train embeddings for linreg fitting (subsample for speed)
    n_fit = min(10000, len(dataset))
    fit_imgs = dataset.img_t[:n_fit]
    fit_z = dataset.z_t[:n_fit]

    # Fast eval subset for in-training monitoring
    n_fast = cfg.get("n_eval_fast", 2000)
    eval_data_fast = (eval_imgs[:n_fast], eval_z[:n_fast])

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    # Dataset label for output paths
    data_label = os.path.basename(args.data_dir)

    # ── Sweep lambda × seed ──────────────────────────────────────────────
    all_results = []

    for lamb in cfg["lambs"]:
        for seed in cfg["seeds"]:
            run_name = f"{data_label}_lamb={lamb:.0e}_seed={seed}"
            print(f"\n{'='*60}")
            print(f"  {run_name}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            encoder = make_cnn_encoder(
                d_latent=cfg["d_latent"], device=device)

            log = train_one(
                encoder, loader, eval_data_fast,
                lamb=lamb, cfg=cfg, device=device)

            # Full eval with train/test split
            metrics = final_eval(encoder, fit_imgs, fit_z,
                                 eval_imgs, eval_z, device)

            result = {
                "experiment": "reacher",
                "run_name": run_name,
                "data_dir": args.data_dir,
                "lamb": lamb,
                "seed": seed,
                "d_latent": cfg["d_latent"],
                # Data meta (exclude 'seed' key to avoid overwriting training seed)
                **{k: v for k, v in data_meta.items() if k != "seed"},
                "render_seed": data_meta.get("seed", None),
                # Metrics
                **{k: v for k, v in metrics.items()
                   if not isinstance(v, np.ndarray)},
                "best_r2_during_training": max(log["r2_hz"]),
                "final_r2_during_training": log["r2_hz"][-1],
                "final_loss": log["total"][-1],
                "final_align": log["align"][-1],
                "final_sigreg": log["sigreg"][-1],
                "log": log,
            }
            all_results.append(result)

            print(f"  → R²(h→z)={metrics['r2_hz']:.4f}  "
                  f"orth_err={metrics['orth_error']:.4f}  "
                  f"R²(sincos)={metrics['r2_sincos']:.4f}")
            print(f"    per-dim R²: {['%.4f' % r for r in metrics['r2_hz_per_dim']]}")
            print(f"    sincos:  {metrics['r2_sincos_per']}")

            # Save checkpoint + result
            run_dir = os.path.join(out_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "train_mean": train_mean,
                "train_std": train_std,
                "d_latent": cfg["d_latent"],
            }, os.path.join(run_dir, "checkpoint.pt"))

            with open(os.path.join(run_dir, "result.json"), "w") as f:
                json.dump(_jsonify(result), f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {r["run_name"]: {k: v for k, v in r.items() if k != "log"}
               for r in all_results}
    with open(os.path.join(out_dir, f"summary_{data_label}.json"), "w") as f:
        json.dump(_jsonify(summary), f, indent=2)

    print(f"\n{'data':>12s} {'lamb':>8s} {'seed':>4s} "
          f"{'R²(h→z)':>8s} {'R²(sc)':>8s} {'orth_err':>8s}")
    print("-" * 56)
    for r in all_results:
        print(f"{data_label:>12s} {r['lamb']:8.1e} {r['seed']:4d} "
              f"{r['r2_hz']:8.4f} {r['r2_sincos']:8.4f} "
              f"{r['orth_error']:8.4f}")


if __name__ == "__main__":
    main()