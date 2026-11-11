"""CPU inference and named embedding exports; never draw training RNG."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import write_json,identity,file_sha256


def embedding_directory(run, epoch):
    """Use current epoch names, while keeping historical sealed runs readable."""
    base=Path(run)/'embeddings'
    current=base/f'{epoch}_epoch'
    legacy=base/f'epoch_{epoch:04d}'
    return legacy if not current.is_dir() and legacy.is_dir() else current

def infer_cpu(model, full, dna, spatial, use_spatial, rs, rd):
    # Seed deepcopy with CPU parameters: never clone the full embedding on CUDA
    # before moving it, and never initialize a second model through the RNG.
    import torch
    with torch.no_grad():
        memo = {
            id(p): torch.nn.Parameter(
                p.detach().to(device="cpu", copy=True), requires_grad=p.requires_grad
            )
            for p in model.parameters()
        }
        cpu_model = copy.deepcopy(model, memo).eval()
        values = cpu_model.propagate(
            full.cpu(),
            None if dna is None else dna.cpu(),
            None if spatial is None else spatial.cpu(),
            use_spatial,
            rs,
            rd,
        )
        return tuple(None if x is None else x.detach() for x in values)


def export_embeddings(output, data, model, values, epoch, *, data_fingerprint=None, final=False):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    u, p, k = values
    entries = []
    entities = [
        ("spot", data.spot_ids, u, model.embedding_spot),
        ("peak", data.peak_ids, p, model.embedding_peak),
    ]
    if k is not None:
        entities.append((data.dna_kind, data.dna_ids, k, model.embedding_kmer))
    for kind, ids, propagated, ego in entities:
        representations = {"propagated": propagated, "ego": ego}
        for representation, array in representations.items():
            array = array.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("cannot export non-finite embedding")
            zero_rows=int(np.count_nonzero(np.linalg.norm(array.astype(np.float64),axis=1)==0))
            if final and representation == "propagated" and zero_rows:
                raise ValueError(f"zero final {kind} embedding row has no direction")
            path = output / f"{kind}_{representation}.tsv.gz"
            # compresslevel only changes the compressed bytes, never the TSV text.
            # Default level 9 spent ~21 s per export on this cohort.
            pd.DataFrame(
                array,
                index=pd.Index(ids, name="entity_id"),
                columns=[f"latent_{j}" for j in range(array.shape[1])],
            ).to_csv(path, sep="\t", compression={"method": "gzip", "compresslevel": 1})
            entries.append(
                {
                    "entity_type": kind,
                    "representation": representation,
                    "path": path.name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "row_ids_hash": identity(ids),
                    "sha256": file_sha256(path),
                    "epoch": epoch,
                    "analyzable": zero_rows == 0,
                    "zero_direction_rows": zero_rows,
                }
            )
    write_json(
        output / "export.json",
        {
            "schema_version": 1,
            "epoch": epoch,
            "data_fingerprint": data.fingerprint if data_fingerprint is None else data_fingerprint,
            "entries": entries,
            "analysis_ready": {rep:all(item['analyzable'] for item in entries if item['representation']==rep)
                               for rep in ('propagated','ego')},
        },
    )
    return entries

