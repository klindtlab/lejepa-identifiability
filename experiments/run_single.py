"""
LeJEPA identifiability experiment — single run.

Usage:
    python run_single.py --lamb 0.01 --rho 0.9 --seed 42 \
        --steps 10000 --D 10000 --out results/

Saves a .pt file with training curves and final metrics.
"""

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


# ═══════════════════════════════════════════════════════════════════════════════
# SIGReg loss (Balestriero & LeCun 2025)
# ═══════════════════════════════════════════════════════════════════════════════
class SIGReg(nn.Module):
    def __init__(self, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt); w[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-t**2 / 2))
        self.register_buffer("weights", w * torch.exp(-t**2 / 2))

    def forward(self, proj):
        # proj: (V, B, d)  -> flatten views
        proj = proj.flatten(0, 1)
        A = F.normalize(torch.randn(proj.size(-1), self.n_slices, device=proj.device), dim=0)
        xt = (proj @ A).unsqueeze(-1) * self.t          # (N, S, K)
        err = (xt.cos().mean(0) - self.phi)**2 + xt.sin().mean(0)**2
        return (err @ self.weights).mean() * proj.size(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Nonlinear mixing: norm-dependent rotation
# ═══════════════════════════════════════════════════════════════════════════════
def g_torch(z):
    """g(z) = R(pi * ||z||) z — measure-preserving spiral diffeomorphism."""
    norms = z.norm(dim=-1) * torch.pi
    c, s = norms.cos(), norms.sin()
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)
    return (R @ z.unsqueeze(-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def compute_mcc(z_true, z_learned):
    """
    Mean Correlation Coefficient (MCC):
    Absolute Pearson correlation between each pair of (true, learned) latent
    dimensions, maximized over permutations via the Hungarian algorithm.

    Returns MCC (scalar) and the best permutation.
    """
    N_dim = z_true.shape[1]
    # Build absolute correlation matrix
    C = np.zeros((N_dim, N_dim))
    for i in range(N_dim):
        for j in range(N_dim):
            C[i, j] = abs(pearsonr(z_true[:, i], z_learned[:, j])[0])
    # Hungarian algorithm (maximize = minimize -C)
    row_ind, col_ind = linear_sum_assignment(-C)
    mcc = C[row_ind, col_ind].mean()
    return mcc, col_ind


def compute_linear_metrics(z_true, z_learned):
    """
    Fit z_learned = A @ z_true + b via OLS.
    Returns:
        r2       : R² of the linear fit (multivariate)
        A        : the fitted linear map (N_dim x N_dim)
        orth_err : ||A^T A - I||_F  (0 if perfectly orthogonal)
        orth_err_normalized : orth_err / sqrt(N_dim) for scale-free comparison
        residual_mse : (1/D) sum_i ||h(z_i) - A z_i - b||²
    """
    N_dim = z_true.shape[1]
    D = len(z_true)
    # Add intercept column
    X = np.column_stack([z_true, np.ones(D)])  # (D, N+1)
    # OLS: z_learned = X @ W,  W = (X^T X)^{-1} X^T z_learned
    W, residuals, rank, sv = np.linalg.lstsq(X, z_learned, rcond=None)
    A = W[:N_dim, :].T   # (N_dim, N_dim)  — the linear map
    b = W[N_dim, :]       # intercept

    # R² per output dimension, then average
    z_pred = z_true @ A.T + b
    ss_res = ((z_learned - z_pred) ** 2).sum(axis=0)
    ss_tot = ((z_learned - z_learned.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = 1 - ss_res / ss_tot
    r2 = r2_per_dim.mean()

    # Orthogonality: ||A^T A - I||_F
    orth_err = np.linalg.norm(A.T @ A - np.eye(N_dim), 'fro')
    orth_err_normalized = orth_err / np.sqrt(N_dim)

    # Raw residual MSE: E[||h(z) - Az - b||²]
    residual_mse = ss_res.sum() / D

    return r2, A, orth_err, orth_err_normalized, r2_per_dim, residual_mse


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Fixed dataset ────────────────────────────────────────────────────────
    z_np = np.random.normal(0, 1, (args.D, args.N))
    z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

    # ── Model ────────────────────────────────────────────────────────────────
    f = nn.Sequential(
        nn.Linear(args.N, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.hidden), nn.GELU(),
        nn.Linear(args.hidden, args.N),
    ).to(device)

    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(f.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── Training loop ────────────────────────────────────────────────────────
    log_steps, log_inv, log_sig, log_total = [], [], [], []
    LAMB = args.lamb
    RHO = args.rho
    N_VIEWS = 2
    BS = args.batch_size

    for step in range(args.steps):
        idx = torch.randperm(len(z_t), device=device)[:BS]
        z_b = z_t[idx]                                           # (B, N)

        # OU channel: z' = rho*z + sqrt(1-rho^2)*eta
        eta = torch.randn(N_VIEWS, BS, args.N, device=device)
        z_aug = RHO * z_b.unsqueeze(0) + np.sqrt(1 - RHO**2) * eta  # (V, B, N)

        # Observations through g, then encode
        ys = g_torch(z_aug)                                      # (V, B, N)
        proj = f(ys.flatten(0, 1)).reshape(N_VIEWS, BS, args.N)  # (V, B, N)

        # Losses
        inv_loss = (proj.mean(0) - proj).square().mean()
        sig_loss = sigreg(proj)
        loss = sig_loss * LAMB + inv_loss * (1 - LAMB)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            log_steps.append(step)
            log_inv.append(inv_loss.item())
            log_sig.append(sig_loss.item())
            log_total.append(loss.item())

    # ── Final evaluation ─────────────────────────────────────────────────────
    f.eval()
    with torch.no_grad():
        fgz_t = f(g_torch(z_t))               # (D, N) tensor
        fgz = fgz_t.cpu().numpy()
    z_eval = z_np

    mcc, best_perm = compute_mcc(z_eval, fgz)
    r2, A, orth_err, orth_err_norm, r2_per_dim, residual_mse = compute_linear_metrics(z_eval, fgz)

    # ── Approximate identifiability: ε, δ, and Procrustes error ──────────
    # ε = ||Cov(h(z)) - I||_F
    cov_hz = np.cov(fgz, rowvar=False)                          # (N, N)
    epsilon = np.linalg.norm(cov_hz - np.eye(args.N), 'fro')
    trace_cov = np.trace(cov_hz)

    # L(h) = E[||h(z') - h(z)||²]  computed on fresh OU pairs
    with torch.no_grad():
        eta_eval = torch.randn(len(z_t), args.N, device=device)
        z_prime = RHO * z_t + np.sqrt(1 - RHO**2) * eta_eval
        hz_prime = f(g_torch(z_prime))
        L_h = (hz_prime - fgz_t).square().sum(dim=-1).mean().item()

    # δ = L(h) - 2(1-ρ) tr(Cov(h(z)))
    theoretical_baseline = 2 * (1 - RHO) * trace_cov
    delta = max(L_h - theoretical_baseline, 0.0)

    # D = δ / (2ρ(1-ρ)),  bound = D + (ε + D)²
    spectral_gap = 2 * RHO * (1 - RHO)
    D_bound = delta / spectral_gap if spectral_gap > 0 else float('inf')
    approx_bound = D_bound + (epsilon + D_bound) ** 2

    # Actual error: min_{Q in O(n)} E[||h(z) - Qz||²] via orthogonal Procrustes
    # Solution: Q = UV^T where h(z)^T z = UΣV^T
    M = fgz.T @ z_eval / len(z_eval)              # (N, N)
    U, S, Vt = np.linalg.svd(M)
    Q = U @ Vt
    procrustes_mse = np.mean(np.sum((fgz - z_eval @ Q.T)**2, axis=1))

    # ── Package results ──────────────────────────────────────────────────────
    results = {
        # Config
        "lamb": args.lamb,
        "rho": args.rho,
        "seed": args.seed,
        "D": args.D,
        "N": args.N,
        "steps": args.steps,
        "hidden": args.hidden,
        "lr": args.lr,
        "wd": args.wd,
        "batch_size": args.batch_size,
        # Training curves
        "log_steps": log_steps,
        "log_inv_loss": log_inv,
        "log_sig_loss": log_sig,
        "log_total_loss": log_total,
        # Final metrics
        "mcc": mcc,
        "linear_r2": r2,
        "linear_r2_per_dim": r2_per_dim.tolist(),
        "orth_err": orth_err,
        "orth_err_normalized": orth_err_norm,
        "linear_map_A": A.tolist(),
        "best_permutation": best_perm.tolist(),
        # Approximate identifiability
        "epsilon": epsilon,
        "delta": delta,
        "D_bound": D_bound,
        "approx_bound": approx_bound,
        "residual_mse": residual_mse,
        "procrustes_mse": procrustes_mse,
        "L_h": L_h,
        "trace_cov": trace_cov,
        "theoretical_baseline": theoretical_baseline,
        "cov_hz": cov_hz.tolist(),
        # Model (for post-hoc analysis)
        "model_state_dict": f.state_dict(),
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    fname = f"lamb={args.lamb:.1e}_rho={args.rho:.2f}_seed={args.seed}.pt"
    path = os.path.join(args.out, fname)
    torch.save(results, path)
    print(f"Saved → {path}")
    print(f"  MCC={mcc:.4f}  R²={r2:.4f}  orth_err={orth_err:.4f}")
    print(f"  ε={epsilon:.4f}  δ={delta:.4f}  D={D_bound:.4f}  "
          f"bound={approx_bound:.4f}  procrustes={procrustes_mse:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lamb",       type=float, required=True,  help="Balance coefficient (0=pure invariance, 1=pure SIGReg)")
    p.add_argument("--rho",        type=float, required=True,  help="OU correlation (0,1)")
    p.add_argument("--seed",       type=int,   required=True)
    p.add_argument("--D",          type=int,   default=10000,  help="Fixed number of datapoints")
    p.add_argument("--N",          type=int,   default=2,      help="Latent dimension")
    p.add_argument("--steps",      type=int,   default=10000)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-3)
    p.add_argument("--wd",         type=float, default=5e-2)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--log_every",  type=int,   default=100)
    p.add_argument("--out",        type=str,   default="results/")
    args = p.parse_args()
    run_experiment(args)