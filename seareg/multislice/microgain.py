"""Detached four-cardinal evidence and same-state displacement allocation."""
import torch
import torch.nn.functional as F

EPS = 1e-8


@torch.no_grad()
def local_quality(features, coordinates, spatial, clean, cs_valid, slice_codes):
    device = features.device
    h = features.detach().float().cpu()
    xy = coordinates.detach().float().cpu()
    c = clean.detach().float().cpu()
    codes = slice_codes.detach().long().cpu()
    a = spatial.detach().cpu().coalesce()
    n, dim = h.shape
    if xy.shape != (n, 2) or c.shape != h.shape or a.shape != (n, n) or codes.shape != (n,):
        raise ValueError('local evidence axes mismatch')
    if not all(torch.isfinite(x).all() for x in (h, xy, c, a.values())) or (a.values() < 0).any():
        raise ValueError('invalid local evidence input')
    row, col = a.indices(); weights = a.values().float()
    if ((codes[row] != codes[col]) & (weights > 0)).any():
        raise ValueError('cross-slice spatial edge')
    delta = xy[col] - xy[row]
    cardinal = ((delta.abs().sum(1) - 1).abs() < 1e-6) & (delta.abs().min(1).values < 1e-6)
    feature_ok = torch.linalg.vector_norm(h, dim=1) > EPS
    active = cardinal & (row != col) & (weights > 0) & feature_ok[row] & feature_ok[col]
    row, col, delta, weights = row[active], col[active], delta[active], weights[active]
    sector = torch.where(delta[:, 0].abs() > delta[:, 1].abs(),
                         torch.where(delta[:, 0] > 0, 0, 2),
                         torch.where(delta[:, 1] > 0, 1, 3))
    bucket = row * 4 + sector
    mass = torch.zeros(n * 4).index_add_(0, bucket, weights).reshape(n, 4)
    z = F.normalize(h, dim=1, eps=EPS)
    means = torch.zeros(n * 4, dim).index_add_(0, bucket, z[col] * weights[:, None]).reshape(n, 4, dim) / mass.clamp_min(EPS)[:, :, None]
    raw_means = torch.zeros(n * 4, dim).index_add_(0, bucket, h[col] * weights[:, None]).reshape(n, 4, dim) / mass.clamp_min(EPS)[:, :, None]
    spread = torch.linalg.vector_norm(z[col] - means[row, sector], dim=1)
    sigma = torch.zeros(n * 4).index_add_(0, bucket, spread * weights).reshape(n, 4) / mass.clamp_min(EPS)
    mean = means.mean(1)
    d = torch.linalg.vector_norm(z - mean, dim=1)
    leave_one_out = (means.sum(1)[:, None, :] - means) / 3
    residual = (torch.linalg.vector_norm(means - leave_one_out, dim=2) + sigma).max(1).values
    outlier = torch.where(d + residual > EPS, (d - residual).clamp_min(0) / (d + residual).clamp_min(EPS), torch.zeros_like(d)).clamp(0, 1)
    v = c - h; u = raw_means.mean(1) - h
    vnorm = torch.linalg.vector_norm(v, dim=1); unorm = torch.linalg.vector_norm(u, dim=1)
    alignment = (F.normalize(v, dim=1, eps=EPS) * F.normalize(u, dim=1, eps=EPS)).sum(1).clamp(0, 1)
    alignment = torch.where((vnorm > EPS) & (unorm > EPS), alignment, torch.zeros_like(alignment))
    valid = (mass > EPS).all(1) & feature_ok & cs_valid.detach().bool().cpu() & (vnorm > EPS)
    quality = torch.where(valid, outlier * alignment, torch.zeros_like(d))
    return {k: t.to(device) for k, t in dict(quality=quality, outlier=outlier, alignment=alignment,
                                            valid=valid, center_distance=d, background_residual=residual).items()}


@torch.no_grad()
def allocate_matched03(capacity, quality, displacement_norm, valid, slice_codes):
    device = capacity.device
    cap, q, norm = (x.detach().double().cpu() for x in (capacity, quality, displacement_norm))
    codes = slice_codes.detach().long().cpu()
    mask = valid.detach().bool().cpu() & (norm > EPS)
    if not all(torch.isfinite(t).all() for t in (cap, q, norm)):
        raise ValueError('nonfinite microgain allocation')
    if (cap < 0).any() or (cap > 1 + 1e-7).any() or (q < 0).any() or (q > 1 + 1e-7).any() or (norm < 0).any():
        raise ValueError('invalid microgain allocation')
    local03 = .03 * cap * q * mask
    matched = torch.zeros_like(cap)
    unique = torch.unique(codes)
    if not torch.equal(unique, torch.arange(unique.numel())):
        raise ValueError('slice codes must be contiguous from zero')
    alpha = torch.zeros(len(unique), dtype=torch.float64)
    for code in unique:
        rows = codes == code
        denominator = (cap[rows] * norm[rows] * mask[rows]).sum()
        if denominator > 0:
            alpha[code] = ((local03[rows] * norm[rows]).sum() / denominator).clamp(0, .03)
            matched[rows] = alpha[code] * cap[rows] * mask[rows]
    return {k: t.to(device=device, dtype=capacity.dtype) for k, t in
            dict(local03=local03, matched03=matched, alpha=alpha).items()}
