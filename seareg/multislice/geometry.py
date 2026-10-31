"""Geometry calculations and a coverage-only report on the input spatial graph."""
import math
import torch
import torch.nn.functional as F

EPS = 1e-8


def spatial_coverage_report(coordinates, spatial, slice_codes):
    """Report graph coverage; this does not assess feature or donor validity."""
    xy = torch.as_tensor(coordinates).detach().float().cpu()
    codes = torch.as_tensor(slice_codes).detach().long().cpu()
    if hasattr(spatial, 'tocoo'):
        coo = spatial.tocoo()
        graph = torch.sparse_coo_tensor(torch.as_tensor([coo.row.tolist(), coo.col.tolist()]),
                                        torch.as_tensor(coo.data), coo.shape).coalesce()
    else:
        graph = spatial.detach().cpu().coalesce()
    n = len(xy)
    if xy.shape != (n, 2) or codes.shape != (n,) or graph.shape != (n, n):
        raise ValueError('coverage axes mismatch')
    if not torch.isfinite(xy).all() or not torch.isfinite(graph.values()).all() or (graph.values() < 0).any():
        raise ValueError('invalid coverage input')
    row, col = graph.indices()
    weights = graph.values().float()
    if ((codes[row] != codes[col]) & (weights > 0)).any():
        raise ValueError('cross-slice spatial edge')
    delta = xy[col] - xy[row]
    active = (row != col) & (weights > 0) & (delta.square().sum(1) > EPS)
    sector = torch.floor((torch.atan2(delta[:, 1], delta[:, 0]) + math.pi / 8) / (math.pi / 4)).long().remainder(8)
    mass = torch.zeros(n * 8).index_add_(0, row * 8 + sector, weights * active).reshape(n, 8)
    cardinal = ((delta.abs().sum(1) - 1).abs() < 1e-6) & (delta.abs().min(1).values < 1e-6) & active
    direction = torch.where(delta[:, 0].abs() > delta[:, 1].abs(),
                            torch.where(delta[:, 0] > 0, 0, 2), torch.where(delta[:, 1] > 0, 1, 3))
    cmass = torch.zeros(n * 4).index_add_(0, row * 4 + direction, weights * cardinal).reshape(n, 4)
    eight = (mass > EPS).all(1); four = (cmass > EPS).all(1)
    rows = []
    for code in torch.unique(codes, sorted=True).tolist():
        selected = codes == code
        count = int(selected.sum()); e = int(eight[selected].sum()); c = int(four[selected].sum())
        rows.append(dict(slice_code=int(code), spots=count, eight_sector_covered=e,
                         eight_sector_fraction=e / count, cardinal_covered=c, cardinal_fraction=c / count))
    return dict(total_spots=n, per_slice=rows, unit_cardinal_distance=1.0, cardinal_tolerance=1e-6)


def correction_cache_key(spatial, g0, geometry):
    def signature(t):
        return (t.data_ptr(), t._version, tuple(t.shape), str(t.device), str(t.dtype))
    return (tuple(spatial.shape), signature(spatial._indices()), signature(spatial._values()),
            signature(g0), tuple((name, signature(value)) for name, value in sorted(geometry.items())))


@torch.no_grad()
def compute_selective_geometry(features, coordinates, spatial, g0):
    device = features.device
    z = F.normalize(features.detach().float().cpu(), dim=1, eps=EPS)
    xy = coordinates.detach().float().cpu()
    a = spatial.detach().cpu().coalesce()
    g = g0.detach().float().cpu()
    n, dim = z.shape
    if xy.shape != (n, 2) or g.shape != (n,) or a.shape != (n, n):
        raise ValueError('geometry axes mismatch')
    if not all(torch.isfinite(t).all() for t in (z, xy, g, a.values())):
        raise ValueError('nonfinite geometry input')
    if (g < 0).any() or (g > 1).any() or (a.values() < 0).any():
        raise ValueError('invalid gate/spatial weights')
    row, col = a.indices()
    w = a.values().float()
    delta = xy[col]-xy[row]
    active = (row != col) & (w > 0) & (delta.square().sum(1) > EPS)
    sector = torch.floor((torch.atan2(delta[:, 1], delta[:, 0])+math.pi/8)/(math.pi/4)).long().remainder(8)
    bucket = row*8+sector
    weight = w*active
    mass = torch.zeros(n*8).index_add_(0, bucket, weight).reshape(n, 8)
    means = torch.zeros(n*8, dim).index_add_(0, bucket, z[col]*weight[:, None]).reshape(n, 8, dim)/mass.clamp_min(EPS)[:, :, None]
    distances = torch.linalg.vector_norm(z[row]-z[col], dim=1)
    distances = torch.zeros(n*8).index_add_(0, bucket, distances*weight).reshape(n, 8)/mass.clamp_min(EPS)
    dispersion = torch.linalg.vector_norm(z[col]-means[row, sector], dim=1)
    dispersion = torch.zeros(n*8).index_add_(0, bucket, dispersion*weight).reshape(n, 8)/mass.clamp_min(EPS)
    halves = (torch.arange(8)[:, None]+torch.arange(4)[None, :]).remainder(8)
    half_mean = means[:, halves].mean(2)
    half_d = distances[:, halves].mean(2)
    half_sigma = (dispersion[:, halves]+torch.linalg.vector_norm(means[:, halves]-half_mean[:, :, None, :], dim=3)).mean(2)
    opposite_d = half_d.roll(-4, 1)
    opposite_sigma = half_sigma.roll(-4, 1)
    boundary = ((opposite_d-half_d).clamp_min(0)/(opposite_d+half_d+half_sigma+opposite_sigma).clamp_min(EPS)).clamp(0, 1).max(1).values
    donor_mass = torch.zeros(n).index_add_(0, row, w*(1-g[col]))
    valid = (mass > EPS).all(1) & (donor_mass > EPS)
    boundary = torch.where(valid, boundary, torch.zeros_like(boundary))
    return {name: value.to(device) for name, value in dict(boundary=boundary, valid_coverage=valid,
        edge_keys=row*n+col, edge_sectors=sector, edge_active=active).items()}


@torch.no_grad()
def prepare_selective_correction(spatial, g0, geometry):
    a = spatial.coalesce()
    n = a.size(0)
    row, col = a.indices()
    values = a.values()
    g = g0.detach().to(values)
    keys = row*n+col
    full_keys = geometry['edge_keys'].to(keys.device)
    pos = torch.searchsorted(full_keys, keys)
    if full_keys.numel() == 0:
        if keys.numel(): raise ValueError('current spatial support has unknown/absent geometry edges')
    elif (pos >= full_keys.numel()).any() or not torch.equal(full_keys[pos], keys):
        raise ValueError('current spatial support has unknown/absent geometry edges')
    donor = 1-g
    denominator = torch.sparse.mm(a, donor[:, None])
    valid = geometry['valid_coverage'].to(keys.device) & (denominator[:, 0] > EPS)
    receiver = torch.where(valid, g*(1-geometry['boundary'].to(values)), g)
    changed = valid & (receiver != g)
    if not all(torch.isfinite(x).all() for x in (receiver, denominator)):
        raise FloatingPointError('nonfinite selective coefficients')
    return dict(receiver=receiver, denominator=denominator, changed=changed,
        cache_key=correction_cache_key(spatial, g0, geometry),
        source_references=(spatial, g0, tuple(geometry.values())))


def selective_correct_message(p1, spatial, g0, geometry, *, prepared=None):
    g = g0.detach().to(p1)
    prep = prepare_selective_correction(spatial, g0, geometry) if prepared is None else prepared
    if prep['cache_key'] != correction_cache_key(spatial, g0, geometry):
        raise ValueError('stale correction cache fingerprint: graph, gate or geometry changed')
    clean = torch.sparse.mm(spatial, p1*(1-g)[:, None])/prep['denominator'].to(p1).clamp_min(EPS)
    original = p1+g[:, None]*(clean-p1)
    candidate = p1+prep['receiver'].to(p1)[:, None]*(clean-p1)
    return torch.where(prep['changed'][:, None], candidate, original)
