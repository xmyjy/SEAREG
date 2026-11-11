"""Adopted Fig. 4 downstream analysis for saved P22 tutorial runs.

The regional contrast, distal selection and network gates follow the
original P22 analysis used for Fig. 4 in the v12 manuscript.
"""

import hashlib
import json
from math import ceil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter

GROUPS_ATAC = {"CC/oligo": ("D2",), "LV/VZ": ("D7",), "Cortex": ("D8", "D9", "D10", "D11")}
GROUPS_H3 = {"CC/oligo": ("D2",), "LV/VZ": ("D7",), "Cortex": ("D8", "D9")}
GROUPS = {group: (GROUPS_ATAC[group], GROUPS_H3[group]) for group in GROUPS_ATAC}
KEY = ["group", "TF", "peak_id", "target_gene"]



def compute_domain_feature_scores(embeddings, labels, targets):
    """Apply the v12 P22 float32 contrast against all remaining spots."""
    spot_embedding = embeddings["spot"]
    spot_domains = labels.set_index("spot_id").domain.reindex(spot_embedding.index)
    spots = l2_rows(spot_embedding.to_numpy())
    tables = []
    for kind in ("peak", "motif"):
        features = embeddings.get(kind)
        if features is None:
            continue
        normalized = l2_rows(features.loc[:, spot_embedding.columns].to_numpy())
        for region, domains in targets.items():
            inside = spot_domains.isin(domains).to_numpy()
            centroid_in = spots[inside].mean(axis=0)
            centroid_out = spots[~inside].mean(axis=0)
            mean_in = normalized @ centroid_in
            mean_out = normalized @ centroid_out
            # The original motif ranking projects the difference of centroids.
            delta = normalized @ (centroid_in - centroid_out) if kind == "motif" else mean_in - mean_out
            tables.append(pd.DataFrame({
                "domain": region,
                "entity_type": kind,
                "entity_id": features.index,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "delta_cosine": delta,
            }))
    return pd.concat(tables, ignore_index=True)


def take_distal(table, groups, rank_name):
    selected = []
    for group in groups:
        valid = table[
            table.group.eq(group) & table.valid_interval & table.is_distal
            & ~table.overlaps_blacklist & table.delta_cosine.gt(0)
        ]
        ranked = valid.sort_values("delta_cosine", ascending=False).drop_duplicates("peak_id")
        head = ranked.head(ceil(len(ranked) * .10)).copy()
        head[rank_name] = np.arange(1, len(head) + 1)
        selected.append(head)
    return pd.concat(selected, ignore_index=True)


def support_hits(query, subject):
    subjects = {
        name: (table.start.to_numpy(), table.end.to_numpy(), table.peak_id.to_numpy())
        for name, table in subject.sort_values("start").groupby("chrom", sort=False)
    }
    rows = []
    for peak in query.itertuples(index=False):
        starts, ends, ids = subjects.get(peak.chrom, (np.empty(0), np.empty(0), np.empty(0)))
        stop = np.searchsorted(starts, peak.end, side="left")
        overlaps = np.minimum(peak.end, ends[:stop]) - np.maximum(peak.start, starts[:stop])
        passed = (overlaps >= 50) & (10 * overlaps >= peak.end - peak.start)
        rows.append((
            peak.peak_id, bool(passed.any()), ";".join(ids[:stop][passed]), int(overlaps[passed].max(initial=0)),
        ))
    return pd.DataFrame(rows, columns=["peak_id", "supported", "overlap_h3_ids", "max_overlap_bp"])


def support_tables(atac, h3):
    result, details = [], []
    for ag in GROUPS_ATAC:
        query = atac[atac.group.eq(ag)]
        for hg in GROUPS_H3:
            subject = h3[h3.group.eq(hg)]
            hits = support_hits(query, subject)
            count = int(hits.supported.sum())
            result.append({
                "atac_group": ag, "h3_group": hg, "n_atac": len(query), "n_h3": len(subject),
                "n_supported": count, "support_fraction": count / len(query), "is_matched": ag == hg,
            })
            details.append(query.merge(hits, on="peak_id", validate="one_to_one").assign(
                atac_group=ag, h3_group=hg, is_matched_pair=ag == hg
            ))
    return pd.DataFrame(result), pd.concat(details, ignore_index=True)


def read_table(path, **kwargs):
    return pd.read_csv(path, sep='\t', **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_peak(value):
    return str(value).replace(':', '-').replace('_', '-')


def l2_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def h3_profiles(path, labels):
    """Calculate measured H3 values across the full raw matrix, including its extra features."""
    with h5py.File(path, 'r') as handle:
        obs = handle['obs/_index'].asstr()[:]
        features = handle['var/_index'].asstr()[:]
        x = handle['X']
        counts = sparse.csr_matrix((x['data'][:], x['indices'][:], x['indptr'][:]),
                                   shape=tuple(x.attrs['shape'])).astype(np.float32)

    raw_barcodes = spot_keys(obs)
    labels = labels.set_index("spot_id")["domain"]
    if len(raw_barcodes.symmetric_difference(labels.index)) or len(raw_barcodes) != 9370:
        raise ValueError("Raw H3K27ac matrix must cover the exact P22 H3K27ac spot cohort")
    labels = labels.reindex(raw_barcodes).to_numpy(str)
    library = np.asarray(counts.sum(axis=1)).ravel()
    assert counts.shape == (9370, 87043) and (library > 0).all()
    normalized = sparse.diags(1e4 / library) @ counts
    normalized.data = np.log1p(normalized.data)
    means = {
        group: np.asarray(normalized[np.isin(labels, domains)].mean(axis=0)).ravel()
        for group, (_, domains) in GROUPS.items()
    }
    feature_table = pd.Series([canonical_peak(x) for x in features]).str.extract(
        r'^(.+)-(\d+)-(\d+)$'
    ).dropna().rename(columns={0: 'chrom', 1: 'start', 2: 'end'})
    feature_table['feature_index'] = feature_table.index
    feature_table = feature_table.astype({'start': int, 'end': int})
    chrom_features = {chrom: group.sort_values('start') for chrom, group in feature_table.groupby('chrom')}
    return chrom_features, means


def raw_h3_effects(peaks, profiles):
    chrom_features, means = profiles
    rows = []
    for peak in peaks[['group', 'peak_id', 'chrom', 'start', 'end']].itertuples(index=False):
        row = peak._asdict()
        intervals = chrom_features.get(peak.chrom)
        indices = np.array([], dtype=int)
        weights = np.array([], dtype=float)
        if intervals is not None:
            cutoff = np.searchsorted(intervals.start.to_numpy(), peak.end, side='left')
            possible = intervals.iloc[:cutoff]
            overlap = np.minimum(peak.end, possible.end.to_numpy()) - np.maximum(peak.start, possible.start.to_numpy())
            eligible = (overlap >= 50) & (overlap / max(1, peak.end - peak.start) >= 0.10)
            indices = possible.loc[eligible, 'feature_index'].to_numpy(int)
            weights = overlap[eligible].astype(float)
        region_values = {
            group: float(np.average(values[indices], weights=weights)) if len(weights) else np.nan
            for group, values in means.items()
        }
        matched = region_values[peak.group]
        outside = np.mean([value for group, value in region_values.items() if group != peak.group])
        row.update({f'raw_h3_mean_{group}': value for group, value in region_values.items()})
        row.update(n_overlapping_raw_h3_features=len(indices),
                   max_overlap_bp_raw_h3=int(weights.max()) if len(weights) else 0,
                   raw_h3_matched_mean=matched, raw_h3_mismatched_mean=outside,
                   raw_h3_matched_minus_mismatched=matched - outside)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_tfs(motifs, tf_tests, motif_to_tf):
    assignment = motifs.merge(motif_to_tf, on='motif', how='inner').merge(
        tf_tests, left_on=['atac_group', 'candidate_tf_gene'], right_on=['group', 'gene'],
        how='inner', validate='many_to_one',
    )
    assignment = assignment[assignment.rna_delta_mean.gt(0) & assignment.rna_fdr.lt(0.05)]
    assignment = assignment.sort_values(
        ['atac_group', 'motif', 'rna_delta_mean', 'rna_fdr', 'candidate_tf_gene'],
        ascending=[True, True, False, True, True],
    ).drop_duplicates(['atac_group', 'motif'])
    return assignment[['atac_group', 'motif', 'candidate_tf_gene', 'rna_delta_mean', 'rna_fdr']].rename(
        columns={'candidate_tf_gene': 'TF', 'rna_delta_mean': 'TF_RNA_effect', 'rna_fdr': 'TF_RNA_FDR'},
    )


def sequence_pairs(links, motifs, peak, motif, pwm):
    """Use the frozen positive scan matrix. Cosine records the strongest motif per triplet."""
    peak_lookup = {canonical_peak(name): i for i, name in enumerate(peak.index)}
    motif_lookup = {name: i for i, name in enumerate(motif.index)}
    peak_vectors, motif_vectors = l2_rows(peak.to_numpy()), l2_rows(motif.to_numpy())
    with h5py.File(pwm, 'r') as handle:
        scanned_peaks = [canonical_peak(x) for x in handle['row_names'].asstr()[:]]
        scanned_motifs = handle['col_names'].asstr()[:]
        scan_index = {name: i for i, name in enumerate(scanned_peaks)}
        requested = sorted({scan_index[x] for x in links.peak_id if x in scan_index})
        scan_values = np.asarray(handle['mat'][requested, :])
    local_index = {scanned_peaks[i]: j for j, i in enumerate(requested)}
    rows = []
    for group, group_peaks in links[['atac_group', 'peak_id']].drop_duplicates().groupby('atac_group'):
        allowed = set(motifs.loc[motifs.atac_group.eq(group), 'motif'])
        for name in group_peaks.peak_id:
            if name not in local_index or name not in peak_lookup:
                continue
            values = scan_values[local_index[name]]
            hit = [i for i in np.flatnonzero(values > 0) if scanned_motifs[i] in allowed
                   and scanned_motifs[i] in motif_lookup]
            if not hit:
                continue
            hits = scanned_motifs[hit]
            cosines = motif_vectors[[motif_lookup[x] for x in hits]] @ peak_vectors[peak_lookup[name]]
            for rank, k in enumerate(np.argsort(-cosines, kind='stable'), start=1):
                rows.append((group, name, hits[k], rank, float(cosines[k]), float(values[hit[k]])))
    pairs = pd.DataFrame(rows, columns=['atac_group', 'peak_id', 'motif',
                                        'motif_rank_within_peak_by_aux_cosine',
                                        'motif_peak_embedding_cosine', 'sequence_motif_support'])
    return pairs.merge(motifs, on=['atac_group', 'motif'], validate='many_to_one')


def build_triplets(support, raw_h3, motifs, p2g, rna_effects, tf_tests, motif_to_tf, atac, pwm):
    raw = raw_h3.rename(columns={'group': 'atac_group'})
    raw_columns = ['atac_group', 'peak_id', 'raw_h3_matched_minus_mismatched', 'n_overlapping_raw_h3_features']
    peaks = support.merge(raw[raw_columns], on=['atac_group', 'peak_id'], validate='one_to_one')
    assert len(peaks) == len(support)
    peaks['raw_h3_positive'] = peaks.raw_h3_matched_minus_mismatched.gt(0)
    links = peaks.merge(p2g, on='peak_id').merge(rna_effects, on=['atac_group', 'target_gene'],
                                                how='left', validate='many_to_one')
    links['target_cross_rna_positive'] = links.target_ATAC_RNA_effect.gt(0) & links.target_H3_RNA_effect.gt(0)
    pairs = sequence_pairs(links, motifs, atac['peak'], atac['motif'], pwm)
    tfs = assign_tfs(motifs, tf_tests, motif_to_tf)
    triplets = links.merge(pairs, on=['atac_group', 'peak_id']).merge(
        tfs, on=['atac_group', 'motif'], validate='many_to_one',
    )
    triplets = triplets.sort_values(KEY + ['motif_peak_embedding_cosine'],
                                    ascending=[True, True, True, True, False]).drop_duplicates(KEY)
    triplets['candidate_triplet'] = (
        triplets.raw_h3_positive & triplets.target_cross_rna_positive
        & triplets.motif_spatial_rank.le(50) & triplets.motif_spatial_delta.gt(0)
        & triplets.sequence_motif_support.gt(0) & triplets.TF_RNA_effect.gt(0)
        & triplets.TF_RNA_FDR.lt(0.05) & triplets.Correlation.ge(0.2) & triplets.FDR.le(0.1)
    )
    triplets['triplet_id'] = triplets.TF + '|' + triplets.peak_id + '|' + triplets.target_gene
    triplets['triplet_rank_score'] = 0.0
    for _, index in triplets.groupby('group').groups.items():
        part = triplets.loc[index]
        target_mean = (part.target_ATAC_RNA_effect + part.target_H3_RNA_effect) / 2
        terms = [part.delta_cosine, part.raw_h3_matched_minus_mismatched, part.motif_spatial_delta,
                 part.Correlation, target_mean]
        score = sum(term.rank(method='average', pct=True).fillna(0) for term in terms)
        triplets.loc[index, 'triplet_rank_score'] = score
    qualified = triplets[triplets.candidate_triplet].sort_values(['group', 'triplet_rank_score'], ascending=[True, False])
    return qualified


def collapse_edges(triplets):
    return triplets.groupby(['group', 'TF', 'target_gene'], as_index=False).agg(
        n_candidate_enhancers=('peak_id', 'nunique'), n_triplets=('triplet_id', 'nunique'),
        max_P2G_r=('Correlation', 'max'), min_P2G_FDR=('FDR', 'min'),
        best_motif_rank=('motif_spatial_rank', 'min'), max_raw_H3_delta=('raw_h3_matched_minus_mismatched', 'max'),
        motifs=('motif', lambda x: ';'.join(sorted(set(x)))),
        candidate_peaks=('peak_id', lambda x: ';'.join(sorted(set(x)))),
    )


def spot_keys(values):
    keys = pd.Index([str(value).removesuffix("-1") for value in values])
    if not keys.is_unique:
        raise ValueError("Spot barcode normalization would merge distinct spots")
    return keys


def read_run(run, embedding_dir, adopted_coordinates, target_k, *, motif):
    """Load the run's exact partition and same propagated exports by SHA-256."""
    from seareg.analysis import load_result

    analysis = Path(run) / "analysis"
    domain_file = analysis / f"spatial_domains_K{target_k}.tsv"
    metadata_file = analysis / "spot_metadata.tsv"
    receipt_file = analysis / "analysis_manifest.json"
    receipt = load_result(analysis).metadata
    provenance = receipt["provenance"]
    export_path = Path(embedding_dir) / "export.json"
    if sha256(export_path) != provenance.get("export_sha256"):
        raise ValueError(f"{export_path} does not match the analysis receipt")
    export = json.loads(export_path.read_text())
    files_sha256 = {row["path"]: row["sha256"] for row in export["entries"]
                    if row["representation"] == "propagated"}
    tables = {}
    inputs = [domain_file, metadata_file, receipt_file, analysis / "manifest.json",
              adopted_coordinates, export_path]
    for kind in (("spot", "peak", "motif") if motif else ("spot", "peak")):
        name = f"{kind}_propagated.tsv.gz"
        path = Path(embedding_dir) / name
        if sha256(path) != files_sha256.get(name):
            raise ValueError(f"{path} does not match the analysis receipt")
        tables[kind] = read_table(path, index_col=0)
        inputs.append(path)
    labels = read_table(domain_file).set_index("spot_id")
    xy = read_table(metadata_file).set_index("spot_id")[["x", "y"]]
    ref = read_table(adopted_coordinates, index_col=0)
    for frame in (labels, xy, ref):
        frame.index = spot_keys(frame.index)
    model_spots = spot_keys(tables["spot"].index)
    if any(len(model_spots.symmetric_difference(frame.index)) for frame in (labels, xy, ref)):
        raise ValueError("T3/T4 run, embeddings and publication evidence must have identical spot cohorts")
    labels = labels.loc[model_spots, "domain"]
    xy = xy.loc[model_spots]
    ref = ref.loc[model_spots]
    if not np.array_equal(xy.to_numpy(), ref[["x", "y"]].to_numpy()):
        raise ValueError("The saved spatial coordinates do not match the publication evidence")
    membership = pd.crosstab(labels, ref.domain)
    if not ((membership > 0).sum(axis=0).eq(1).all() and (membership > 0).sum(axis=1).eq(1).all()):
        raise ValueError("Saved labels are not a one-to-one relabeling of the adopted P22 partition")
    relabel = membership.idxmax(axis=1)
    adopted_labels = pd.DataFrame({"spot_id": model_spots, "domain": labels.map(relabel).to_numpy()})
    tables["spot"].index = model_spots
    tables["peak"].index = tables["peak"].index.map(canonical_peak)
    return tables, adopted_labels, xy.to_numpy(float), relabel, inputs


def register_h3_to_atac(atac_xy, h3_xy):
    """Match full-section centroids and RMS radii."""
    center_a, center_h = atac_xy.mean(axis=0), h3_xy.mean(axis=0)
    radius_a = np.sqrt(np.square(atac_xy - center_a).sum(axis=1).mean())
    radius_h = np.sqrt(np.square(h3_xy - center_h).sum(axis=1).mean())
    return (h3_xy - center_h) * (radius_a / radius_h) + center_a


def domain_density_cosine(atac_labels, atac_xy, h3_labels, h3_xy):
    """Calculate the complete Fig. 4d matrix on the adopted domain labels."""
    h3_aligned = register_h3_to_atac(atac_xy, h3_xy)
    stacked = np.vstack([atac_xy, h3_aligned])
    minimum, maximum = stacked.min(axis=0), stacked.max(axis=0)
    padding = 0.025 * (maximum - minimum)
    x_edges, y_edges = np.linspace(minimum - padding, maximum + padding, 81).T

    def density(labels, xy, domain):
        coordinates = xy[labels.domain.eq(domain).to_numpy()]
        hist, _, _ = np.histogram2d(coordinates[:, 1], coordinates[:, 0], bins=[y_edges, x_edges])
        vector = gaussian_filter(hist, sigma=1.5).ravel()
        return vector / np.linalg.norm(vector)

    atac_domains = [f"D{i}" for i in range(13)]
    h3_domains = [f"D{i}" for i in range(11)]
    left = [density(atac_labels, atac_xy, domain) for domain in atac_domains]
    right = [density(h3_labels, h3_aligned, domain) for domain in h3_domains]
    return pd.DataFrame(
        np.asarray(left) @ np.asarray(right).T,
        index=pd.Index([f"A{i}" for i in range(13)], name="ATAC_domain"),
        columns=pd.Index([f"H{i}" for i in range(11)], name="H3K27ac_domain"),
    )


def distal_and_motifs(embeddings, labels, groups, annotation_path, rank_name):
    scores = compute_domain_feature_scores(embeddings, labels, groups)
    peaks = scores[scores.entity_type.eq("peak")].rename(
        columns={"domain": "group", "entity_id": "peak_id", "mean_in": "score_in", "mean_out": "score_out"}
    )
    fields = ["group", "peak_id", "chrom", "start", "end", "valid_interval", "is_distal", "overlaps_blacklist"]
    annotation = read_table(annotation_path, usecols=fields)
    peaks = annotation.merge(peaks, on=["group", "peak_id"], validate="one_to_one")
    if len(peaks) != len(annotation) or len(peaks) != sum(scores.entity_type.eq("peak")):
        raise ValueError("Peak annotations must cover the saved section's entire peak universe")
    motifs = scores[scores.entity_type.eq("motif")].rename(
        columns={"domain": "atac_group", "entity_id": "motif", "delta_cosine": "motif_spatial_delta"}
    )
    motifs = motifs.sort_values(["atac_group", "motif_spatial_delta", "motif"], ascending=[True, False, True])
    motifs["motif_spatial_rank"] = motifs.groupby("atac_group").cumcount() + 1
    motifs = motifs[motifs.motif_spatial_rank.le(50) & motifs.motif_spatial_delta.gt(0)]
    return take_distal(peaks, groups, rank_name), motifs[
        ["atac_group", "motif", "motif_spatial_rank", "motif_spatial_delta"]
    ]


def network_evidence(support_details, h3_labels, evidence):
    support = support_details[support_details.is_matched_pair & support_details.supported].copy()
    raw = raw_h3_effects(support, h3_profiles(evidence["raw_h3"], h3_labels))
    links = read_table(evidence["p2g"]).rename(columns={"peak_name": "peak_id", "symbol": "target_gene"})
    links = links[links.Correlation.ge(0.2) & links.FDR.le(0.1)].sort_values(
        ["peak_id", "target_gene", "FDR", "Correlation"], ascending=[True, True, True, False]
    ).drop_duplicates(["peak_id", "target_gene"])
    return support, raw, links


def read_evidence_manifest(path):
    """Resolve explicit evidence paths relative to a user-provided JSON."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    return {field: (path.parent / value).resolve() for field, value in raw.items()}


def summarize_network(triplets, edges):
    counts = triplets.groupby("group", sort=False).agg(
        triplets=("triplet_id", "size"), TFs=("TF", "nunique"),
        elements=("peak_id", "nunique"), targets=("target_gene", "nunique"),
    )
    return counts.join(edges.groupby("group").size().rename("edges")).reindex(list(GROUPS)).fillna(0).astype(int)
