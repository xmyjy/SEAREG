"""Reuse example domain names and colors only for an identical partition."""
import json
from pathlib import Path


def tutorial_domain_style(labels, dataset, *, styles_dir=None):
    """Return display labels and a palette without modifying analysis labels.

    The bundled examples retain their established presentation. A new cohort
    or partition keeps its native domain names and the normal plotting colors.
    This function never assigns spots to different clusters.
    """
    if labels is None:
        return None, None
    styles_dir = Path(styles_dir) if styles_dir is not None else Path(__file__).parent / "styles"
    styles = json.loads((styles_dir / "domain_styles.json").read_text())
    spec = styles.get(dataset)
    if spec is None or len(labels) != spec["spots"]:
        return labels, None
    import pandas as pd
    reference = pd.read_csv(styles_dir / spec["labels"], sep="\t", dtype=str)
    if (labels.spot_id.duplicated().any() or reference.spot_id.duplicated().any()
            or set(labels.spot_id) != set(reference.spot_id)):
        return labels, None
    aligned = reference.set_index("spot_id").loc[labels.spot_id]
    mapping = {}
    for domain in labels.domain.astype(str).unique():
        names = aligned.loc[labels.domain.astype(str).to_numpy() == domain, "domain"].unique()
        if len(names) != 1:
            return labels, None
        mapping[domain] = str(names[0])
    if len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(spec["palette"]):
        return labels, None
    display = labels.copy()
    display["domain"] = labels.domain.astype(str).map(mapping)
    return display, dict(spec["palette"])
