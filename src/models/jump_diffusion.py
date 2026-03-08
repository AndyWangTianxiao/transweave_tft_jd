"""
Jump-diffusion process: simulator (formula 1), density (formula 8), static MLE.
All parameters are annualized: μ, σ, λ per year; μ_J, σ_J are per-jump (return units).
Δt = 1 / bars_per_year (e.g. 1/35040 for 15min).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from scipy import optimize


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def simulate_path(
    mu: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    T: int,
    dt: float,
    seed: int,
    batch: int = 1,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Simulate jump-diffusion path per formula (1).
    r_{t+1} = μ·Δt + σ·√Δt·ε_t + Σ_{k=1}^{N_t} Y_{t,k}.

    Args:
        mu, sigma, lam, mu_J, sigma_J: annualized parameters (μ_J, σ_J in return units).
        T: number of steps.
        dt: 1 / bars_per_year.
        seed: random seed.
        batch: batch size (optional).
    Returns:
        (batch, T) or (T,) tensor of log returns.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    single = batch == 1
    if single:
        batch = 1

    # (batch, T)
    eps = torch.randn(batch, T, device=device, generator=gen)
    # Poisson counts per step: N_t ~ Poisson(λ Δt)
    N_t = torch.poisson(torch.full((batch, T), lam * dt, device=device), generator=gen)
    # Jump sum: for each (b,t), sum of N_t[b,t] iid N(μ_J, σ_J²) = N(N_t*μ_J, N_t*σ_J²)
    # Implement: for each (b,t), draw N_t[b,t] jumps. Vectorized: draw max(N_t) jumps and mask.
    n_max = int(N_t.max().item()) + 1
    # (batch, T, n_max) - each row has n_max iid N(μ_J, σ_J²)
    jumps_iid = mu_J + sigma_J * torch.randn(batch, T, n_max, device=device, generator=gen)
    # For each (b,t), take sum of first N_t[b,t] jumps
    N_t_exp = N_t.unsqueeze(-1).clamp(max=n_max - 1)  # (batch, T, 1)
    indices = torch.arange(n_max, device=device).view(1, 1, -1)
    mask = (indices < N_t_exp).float()  # (batch, T, n_max)
    jump_sum = (jumps_iid * mask).sum(dim=-1)  # (batch, T)

    drift = mu * dt
    vol = sigma * (dt ** 0.5)
    r = drift + vol * eps + jump_sum
    if single:
        return r.squeeze(0)
    return r


def log_density(
    r: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    n_max: int,
) -> torch.Tensor:
    """
    Log of jump-diffusion density formula (8) in log-space for numerical stability.
    p(r|θ) = e^{-λΔt} φ_N(r; μΔt, σ²Δt) + Σ_{n=1}^{n_max} [(λΔt)^n e^{-λΔt}/n!] p_n(r).
    With normal jumps: p_n(r) = φ_N(r; μΔt + n·μ_J, σ²Δt + n·σ_J²).
    Uses log_poisson_weight_n = n*log(λΔt) - λΔt - log(n!), then logsumexp.

    Args:
        r: (...,) observed return(s).
        mu, sigma, lam, mu_J, sigma_J: scalars or broadcastable to r.shape.
        dt: 1 / bars_per_year.
        n_max: truncation (config jd_truncation_n).
    Returns:
        log p(r|θ), same shape as r.
    """
    device = r.device
    r = r.double()
    if isinstance(mu, (int, float)):
        mu = torch.tensor(mu, dtype=torch.float64, device=device)
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=torch.float64, device=device)
    if isinstance(mu_J, (int, float)):
        mu_J = torch.tensor(mu_J, dtype=torch.float64, device=device)
    if isinstance(sigma_J, (int, float)):
        sigma_J = torch.tensor(sigma_J, dtype=torch.float64, device=device)
    mu, sigma, lam, mu_J, sigma_J = mu.to(device).double(), sigma.to(device).double(), lam.to(device).double(), mu_J.to(device).double(), sigma_J.to(device).double()

    lam_dt = lam * dt
    drift_dt = mu * dt
    var_dt = (sigma ** 2) * dt

    # Log weights and log normal densities in list, then stack and logsumexp
    # n = 0: weight = e^{-λΔt}, mean = μΔt, var = σ²Δt
    log_w0 = -lam_dt
    mean0 = drift_dt
    var0 = var_dt.clamp(min=1e-12)
    log_p0 = log_w0 + _log_normal(r, mean0, var0)

    terms = [log_p0]
    for n in range(1, n_max + 1):
        # log_poisson_weight_n = n*log(λΔt) - λΔt - log(n!)
        log_w_n = n * torch.log(lam_dt.clamp(min=1e-12)) - lam_dt - torch.lgamma(torch.tensor(n + 1, device=device, dtype=torch.float64))
        mean_n = drift_dt + n * mu_J
        var_n = var_dt + n * (sigma_J ** 2)
        var_n = var_n.clamp(min=1e-12)
        log_p_n = log_w_n + _log_normal(r, mean_n, var_n)
        terms.append(log_p_n)

    # (..., n_max+1)
    stacked = torch.stack(terms, dim=-1)
    out = torch.logsumexp(stacked, dim=-1)
    return out.to(r.dtype)


def _log_normal(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Log of normal pdf; broadcast over x, mean, var."""
    return -0.5 * (np.log(2 * np.pi) + torch.log(var) + (x - mean) ** 2 / var)


def density(
    r: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    n_max: int,
) -> torch.Tensor:
    """
    Jump-diffusion density p(r|θ), formula (8). Implemented via log_density + exp.
    """
    return torch.exp(log_density(r, mu, sigma, lam, mu_J, sigma_J, dt, n_max))


def sample_one_step(
    mu: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    dt: float,
    n_samples: int,
    seed: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Sample n_samples independent draws from the one-step JD marginal p(r|θ).
    Used for CRPS Monte Carlo: r = μΔt + σ√Δt·ε + N·μ_J + √N·σ_J·ε_J with N~Poisson(λΔt).
    Returns shape (n_samples,).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    eps = torch.randn(n_samples, device=device, generator=gen)
    lam_val = float(lam) * dt
    # Defensive: Poisson requires finite non-negative rate; NaN/Inf/neg -> fallback
    lam_dt = max(lam_val, 1e-12) if (np.isfinite(lam_val) and lam_val >= 0) else 1e-12
    N = torch.poisson(torch.full((n_samples,), lam_dt, device=device), generator=gen)
    jump_mean = N.double() * mu_J
    # When N=0, jump = 0; when N>0, jump ~ N(N*μ_J, N*σ_J²)
    jump_std = torch.where(N > 0, (N.double() ** 0.5) * sigma_J, torch.zeros(n_samples, device=device, dtype=torch.float64))
    jump = jump_mean + jump_std * torch.randn(n_samples, device=device, generator=gen)
    r = mu * dt + sigma * (dt ** 0.5) * eps + jump
    return r


def fit_static_mle(
    r_train: np.ndarray,
    config_path: str = "config.yaml",
    out_filename: Optional[str] = "eth_static_jd_params.json",
    prior_weight_override: Optional[float] = None,
    asset: Optional[str] = None,
    lambda_max_override: Optional[float] = None,
    sigma_j_min_override: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Static MAP for annualized (μ, σ, λ, μ_J, σ_J). Scheme 1 + 2: robust σ anchor,
    per-bar priors on p=λΔt and κ=σ_J/(σ√Δt) to break λ–σ_J identifiability ridge.
    Minimizes -Σ_t log p(r_t|θ) + prior_penalty.
    Saves result to experiments/checkpoints/{out_filename}.

    Args:
        r_train: 1d array of train log returns.
        out_filename: Output JSON filename (e.g. "ETH_static_jd_params.json").
    Returns:
        (mu, sigma, lam, mu_J, sigma_J) annualized.
    """
    config = _load_config(config_path)
    dt = 1.0 / config["training"]["bars_per_year"]
    n_max = config["training"]["jd_truncation_n"]
    seed = config["training"]["seed"]
    paths = config["paths"]
    root = Path(__file__).resolve().parents[2]
    out_dir = root / paths["checkpoints"]
    out_dir.mkdir(parents=True, exist_ok=True)

    r_tensor = torch.from_numpy(r_train.astype(np.float64))

    # --- Scheme 1: robust sigma estimate (per-bar) to anchor diffusion volatility ---
    # Use MAD-based robust estimator on per-bar returns, then annualize via bars_per_year.
    bars_per_year = config["training"]["bars_per_year"]
    mad = float(np.median(np.abs(r_train - np.median(r_train))))
    # For Normal, sigma ≈ 1.4826 * MAD
    sigma_bar_robust = 1.4826 * mad
    sigma_annual_robust = sigma_bar_robust * (bars_per_year ** 0.5)
    if out_filename is not None:
        print(
            f"Robust sigma_annual ≈ {sigma_annual_robust:.4f} "
            f"({sigma_annual_robust * 100:.2f}%), per-bar sigma ≈ {sigma_bar_robust:.6f}"
        )

    # Scheme 2: per-bar MAP priors (stage2_jd.md)
    map_cfg = config.get("static_jd_map", {})
    p_center_source = map_cfg.get("p_center_source", "config")
    if p_center_source == "empirical":
        emp_asset = (asset or "eth").lower()
        emp_path = out_dir / f"{emp_asset}_empirical_jump_params.json"
        if emp_path.exists():
            import json
            with open(emp_path) as f:
                emp = json.load(f)
            p_center = emp.get("p_center_recommended", map_cfg.get("p_center", 0.002))
        else:
            p_center = map_cfg.get("p_center", 0.002)
    else:
        p_center = map_cfg.get("p_center", 0.002)
    scale_p = map_cfg.get("scale_p", 1.0)
    kappa_center = map_cfg.get("kappa_center", 10.0)
    scale_kappa = map_cfg.get("scale_kappa", 0.5)
    prior_weight = prior_weight_override if prior_weight_override is not None else map_cfg.get("prior_weight", 5000)

    def nll(x: np.ndarray) -> float:
        """
        MAP objective for static JD: NLL + prior penalties (Scheme 2).
        x = [mu, log_sigma, log_lam, mu_J, log_sigma_J]

        Prior on p = λΔt (logit-normal): penalizes high λ (many small jumps).
        Prior on κ = σ_J/(σ√Δt) (log-normal): penalizes small σ_J (jumps too small).
        """
        mu, log_sigma, log_lam, mu_J, log_sigma_J = x
        sigma = float(np.exp(log_sigma))
        lam = float(np.exp(log_lam))
        sigma_J = float(np.exp(log_sigma_J))

        log_p = log_density(r_tensor, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
        base_nll = -float(log_p.sum().item())
        if not np.isfinite(base_nll):
            return 1e12

        # Prior 1: p = λΔt, logit(p) ~ N(logit(p_center), scale_p^2)
        p = lam * dt
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1 - p))
        logit_p_center = np.log(p_center / (1 - p_center))
        pen_p = 0.5 * ((logit_p - logit_p_center) / scale_p) ** 2

        # Prior 2: κ = σ_J / (σ√Δt), log(κ) ~ N(log(κ_center), scale_κ^2)
        sigma_bar = sigma * (dt ** 0.5)
        kappa = sigma_J / sigma_bar if sigma_bar > 1e-12 else 1.0
        kappa = max(kappa, 1e-6)
        log_kappa = np.log(kappa)
        log_kappa_center = np.log(kappa_center)
        pen_kappa = 0.5 * ((log_kappa - log_kappa_center) / scale_kappa) ** 2

        prior_penalty = prior_weight * (pen_p + pen_kappa)
        return base_nll + prior_penalty

    # Initial: anchor sigma at robust estimate (scheme 1), then optimize jump parameters around it.
    sigma_init = sigma_annual_robust
    lam_init = 50.0  # ~50 jumps per year
    mu_init = float(np.mean(r_train)) * bars_per_year
    mu_J_init = -0.005
    sigma_J_init = 0.02
    x0 = np.array([
        mu_init,
        np.log(sigma_init),
        np.log(lam_init),
        mu_J_init,
        np.log(sigma_J_init),
    ])

    # Bounds to keep parameters in economically reasonable ranges:
    # mu: free (annual drift)
    # sigma_annual tightly around robust estimate (e.g. ±20%)
    # lambda_annual in [1, 500]
    # mu_J in [-0.5, 0] (up to -50% average jump, non-positive)
    # sigma_J in [0.02, 0.5] (~2%–50%)
    log_sigma_center = np.log(sigma_annual_robust)
    log_sigma_lo = np.log(sigma_annual_robust * 0.8)
    log_sigma_hi = np.log(sigma_annual_robust * 1.2)
    bounds = [
        (-np.inf, np.inf),               # mu
        (log_sigma_lo, log_sigma_hi),    # log_sigma (scheme 1: narrow band around robust sigma)
        (np.log(1.0), np.log(lambda_max_override if lambda_max_override is not None else 500.0)),    # log_lam
        (-0.5, 0.0),                     # mu_J
        (np.log(sigma_j_min_override if sigma_j_min_override is not None else 0.02), np.log(0.5)),     # log_sigma_J
    ]

    res = optimize.minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    mu_hat, log_sigma, log_lam, mu_J_hat, log_sigma_J = res.x
    sigma_hat = np.exp(log_sigma)
    lam_hat = np.exp(log_lam)
    sigma_J_hat = np.exp(log_sigma_J)

    # Save (skip when out_filename is None, e.g. for rolling MLE)
    if out_filename is not None:
        import json
        out_path = out_dir / out_filename
        with open(out_path, "w") as f:
            json.dump({
                "mu": float(mu_hat),
                "sigma": float(sigma_hat),
                "lambda": float(lam_hat),
                "mu_J": float(mu_J_hat),
                "sigma_J": float(sigma_J_hat),
            }, f, indent=2)
        print(f"Static MAP (Scheme 2, annualized) saved to {out_path}")
    if out_filename is not None:
        p_hat = lam_hat * dt
        sigma_bar_hat = sigma_hat * (dt ** 0.5)
        kappa_hat = sigma_J_hat / sigma_bar_hat if sigma_bar_hat > 1e-12 else 0.0
        print(f"  μ = {mu_hat:.6f}, σ = {sigma_hat:.4f} ({sigma_hat*100:.2f}%), λ = {lam_hat:.2f}, μ_J = {mu_J_hat:.6f}, σ_J = {sigma_J_hat:.4f} ({sigma_J_hat*100:.2f}%)")
        print(f"  p = λΔt = {p_hat:.6f}, κ = σ_J/(σ√Δt) = {kappa_hat:.2f}")
    return (float(mu_hat), float(sigma_hat), float(lam_hat), float(mu_J_hat), float(sigma_J_hat))


def fit_rolling_mle(
    r_train: np.ndarray,
    window_bars: int,
    stride_bars: int,
    config_path: str = "config.yaml",
) -> List[Dict[str, float]]:
    """
    Rolling-window JD MLE per doc/stage5_regime.md Phase E.
    Returns list of theta dicts (mu, sigma, lam, mu_J, sigma_J) per window.
    Used for W_JD_rolling and jump burstiness (Var(λ_t)/E[λ_t]).
    """
    n = len(r_train)
    if n < window_bars:
        return []
    thetas = []
    for start in range(0, n - window_bars + 1, stride_bars):
        r_slice = r_train[start : start + window_bars]
        if len(r_slice) < 500:  # need enough for stable MLE
            continue
        try:
            mu, sigma, lam, mu_J, sigma_J = fit_static_mle(
                r_slice, config_path, out_filename=None
            )
            thetas.append({
                "mu": mu, "sigma": sigma, "lam": lam, "lambda": lam,
                "mu_J": mu_J, "sigma_J": sigma_J,
            })
        except Exception:
            continue
    return thetas


def load_eth_train_returns(config_path: str = "config.yaml") -> np.ndarray:
    """Load ETH train segment y from npz (split=='train')."""
    return load_asset_train_returns("ETH", config_path)


def load_asset_train_returns(asset: str, config_path: str = "config.yaml") -> np.ndarray:
    """
    Load train segment y (log returns) for any asset from npz.
    Used by Stage 5 for multi-asset static JD MLE.
    """
    config = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    feat_path = root / config["paths"]["features"] / f"{asset}_tft_arrays.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    data = np.load(feat_path, allow_pickle=True)
    y = data["y"]
    split = data["split"]
    train_mask = np.asarray(split) == "train"
    r = np.asarray(y[train_mask], dtype=np.float64)
    r = r[np.isfinite(r)]
    return r


if __name__ == "__main__":
    r = load_eth_train_returns()
    fit_static_mle(r)
