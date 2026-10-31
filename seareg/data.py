"""Named input matrices, embeddings and verified artifact handles."""

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import numpy as np
import scipy.sparse as sp
from numbers import Integral
from pathlib import Path
import json
import pandas as pd
import re
import os
from .utils import artifact_writer, file_sha256, verify_artifact, write_json, protect_output, identity
from .artifacts import ModelNetwork as PreparedNetwork, StageArtifact, validated_operation, validation_scope


def names(values, label):
    values = tuple(x.decode("utf8") if isinstance(x, bytes) else str(x) for x in values)
    if any(not x or x in ("nan", "None", "<NA>") or "\t" in x or "\n" in x for x in values):
        raise ValueError(f"{label}: invalid ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: IDs must be unique")
    return values


def peak_keys(values):
    result = []
    for value in names(values, "peak intervals"):
        match = re.fullmatch(r"(chr.+?)[:_-](\d+)[_-](\d+)", value)
        if match is None:
            raise ValueError(f"invalid genomic peak interval: {value}")
        chrom, start, end = match.groups()
        start, end = int(start), int(end)
        if end <= start:
            raise ValueError(f"peak end must exceed start: {value}")
        result.append(f"{chrom}:{start}-{end}")
    return names(result, "canonical peak intervals")

def _revcomp(kmer):
    kmer = kmer.upper()
    if not set(kmer) <= set("ACGTN"):
        raise ValueError(f"invalid kmer alphabet: {kmer}")
    return kmer.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def canonical_csr(matrix, binary=False):
    x = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError("matrix requires finite nonnegative values")
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if not np.isfinite(x.data).all():
        raise ValueError("matrix duplicate summation introduced non-finite values or overflow")
    if binary:
        x.data[:] = 1
    return x


def matrix_hash(x):
    h = hashlib.sha256()
    for a in (
        np.asarray(x.shape, dtype=np.int64),
        x.indptr.astype(np.int64),
        x.indices.astype(np.int64),
        x.data,
    ):
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class PreparedData:
    X: sp.csr_matrix
    spot_ids: tuple[str, ...]
    peak_ids: tuple[str, ...]
    coordinates: np.ndarray | None = None
    slice_ids: tuple[str, ...] = ()
    dna: sp.csr_matrix | None = None
    dna_ids: tuple[str, ...] = ()
    dna_kind: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.provenance = deepcopy(self.provenance)
        self.X = canonical_csr(self.X, binary=True)
        self.spot_ids = names(self.spot_ids, "spots")
        self.peak_ids = names(self.peak_ids, "peaks")
        if self.X.shape != (len(self.spot_ids), len(self.peak_ids)):
            raise ValueError("ATAC names/shape mismatch")
        if min(self.X.shape) == 0 or not self.X.nnz:
            raise ValueError("empty ATAC matrix")
        self.slice_ids = (
            tuple(str(x) for x in self.slice_ids)
            if len(self.slice_ids)
            else ("sample",) * self.X.shape[0]
        )
        if len(self.slice_ids) != self.X.shape[0] or any(
            x in ("nan", "None", "<NA>", "") for x in self.slice_ids
        ):
            raise ValueError("slice ID shape/values invalid")
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates).copy()
            if (
                self.coordinates.ndim != 2
                or self.coordinates.shape[0] != self.X.shape[0]
                or self.coordinates.shape[1] < 2
                or not np.isfinite(self.coordinates).all()
            ):
                raise ValueError("spatial coordinates shape/values invalid")
        if self.dna is not None:
            self.dna = canonical_csr(self.dna)
            self.dna_ids = names(self.dna_ids, "DNA")
            if self.dna.shape != (self.X.shape[1], len(self.dna_ids)) or not self.dna.nnz:
                raise ValueError("DNA shape/empty graph invalid")
            if self.dna_kind not in ("motif", "kmer"):
                raise ValueError("DNA kind must be explicit")
        elif len(self.dna_ids) or self.dna_kind:
            raise ValueError("DNA metadata without a matrix")

    @property
    def fingerprint(self):
        return identity(
            {
                "X": matrix_hash(self.X),
                "spots": self.spot_ids,
                "peaks": self.peak_ids,
                "spatial": (
                    None
                    if self.coordinates is None
                    else {
                        "sha256": hashlib.sha256(self.coordinates.tobytes()).hexdigest(),
                        "shape": self.coordinates.shape,
                        "dtype": str(self.coordinates.dtype),
                    }
                ),
                "slices": self.slice_ids,
                "dna": None if self.dna is None else matrix_hash(self.dna),
                "dna_ids": self.dna_ids,
                "dna_kind": self.dna_kind,
                "provenance": self.provenance,
            }
        )


def validated_frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label}: nonempty named DataFrame required")
    result = frame.copy(deep=True)
    result.index = pd.Index(names(result.index, label), name="entity_id")
    try:
        values = result.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: numeric embedding required") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite embedding")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError(f"{label}: zero embedding row has no direction")
    result.iloc[:, :] = values
    return result.astype(np.float64)


def row_l2(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("row-L2 requires finite nonzero rows")
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm == 0):
        raise ValueError("row-L2 requires finite nonzero row norms without overflow")
    return x / np.maximum(norm, 1e-12)


def read_embeddings(path, expected_ids=None):
    # Parse IDs as text before selecting an index so that "001" != "1".
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if table.shape[1] < 2:
        raise ValueError("embedding TSV requires ID and numeric columns")
    index = table.iloc[:, 0].tolist()
    frame = table.iloc[:, 1:].copy()
    frame.index = index
    if "slice_id" in frame:
        frame = frame.drop(columns="slice_id")
    frame = validated_frame(frame, "embedding TSV")
    if expected_ids is not None:
        expected = names(expected_ids, "expected embeddings")
        missing = set(expected) - set(frame.index)
        extra = set(frame.index) - set(expected)
        if missing or extra:
            raise ValueError(
                f"embedding cohort mismatch: missing={list(missing)[:5]}, extra={list(extra)[:5]}"
            )
        frame = frame.loc[list(expected)]
    return frame


@dataclass
class EmbeddingSet:
    spot: pd.DataFrame
    peak: pd.DataFrame | None = None
    motif: pd.DataFrame | None = None
    dna_kind: str = "motif"
    coordinates: np.ndarray | None = None
    slice_ids: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.spot = validated_frame(self.spot, "spot")
        for label in ("peak", "motif"):
            frame = getattr(self, label)
            if frame is not None:
                frame = validated_frame(frame, label)
                if frame.shape[1] != self.spot.shape[1] or list(frame.columns) != list(
                    self.spot.columns
                ):
                    raise ValueError("shared latent dimensions/column order mismatch")
                setattr(self, label, frame)
        if self.dna_kind not in ("motif", "kmer"):
            raise ValueError("DNA entity kind must be motif or kmer")
        if self.coordinates is not None:
            self.coordinates = np.array(self.coordinates, copy=True)
            if (
                self.coordinates.ndim != 2
                or self.coordinates.shape[0] != len(self.spot)
                or self.coordinates.shape[1] < 2
                or not np.isfinite(self.coordinates).all()
            ):
                raise ValueError("spatial coordinates do not match spot identity/shape")
        self.slice_ids = (
            tuple(self.slice_ids) if len(self.slice_ids) else ("sample",) * len(self.spot)
        )
        if len(self.slice_ids) != len(self.spot) or any(
            str(x) in ("", "nan", "None", "<NA>") for x in self.slice_ids
        ):
            raise ValueError("invalid slice labels")
        self.provenance = dict(self.provenance)


@validated_operation
def from_run(path, epoch=None, representation="propagated"):
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1):
        raise ValueError("epoch must be a positive integer")
    path = Path(path)
    with validation_scope() as context:
        manifest = context.verify(path, "training_run")
    meta = json.loads((path / "data.json").read_text())
    epoch = manifest["metadata"]["final_adopted_output_epoch"] if epoch is None else epoch
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    epoch = int(epoch)
    if representation not in ("ego", "propagated"):
        raise ValueError("unknown embedding representation")
    from .exports import embedding_directory
    directory = embedding_directory(path,epoch)
    export = json.loads((directory / "export.json").read_text())
    if (
        export.get("schema_version") != 1
        or export.get("epoch") != epoch
        or export.get("data_fingerprint") != manifest["metadata"]["data_fingerprint"]
    ):
        raise ValueError("export artifact schema, epoch or input fingerprint mismatch")
    if export.get('analysis_ready',{}).get(representation) is False:
        raise ValueError(f'epoch {epoch} {representation} embedding is not analyzable: '
                         'zero-direction rows are recorded in export.json; '
                         'use an eligible later epoch or an eligible representation')

    def load(kind, ids, rep):
        entry = next(
            (
                x
                for x in export["entries"]
                if x["entity_type"] == kind and x["representation"] == rep
            ),
            None,
        )
        if entry is None:
            raise ValueError(f"{kind}/{rep} was not exported at epoch {epoch}")
        if entry["row_ids_hash"] != identity(ids):
            raise ValueError("export ID fingerprint mismatch")
        name = Path(entry["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("unsafe embedding path outside verified artifact")
        candidate = directory / name
        relative = str(candidate.relative_to(path))
        if relative not in manifest["files"] or manifest["files"][relative] != entry["sha256"]:
            raise ValueError("embedding path/hash is not registered in verified artifact")
        frame = read_embeddings(candidate, ids)
        if list(frame.shape) != entry["shape"] or entry["epoch"] != epoch:
            raise ValueError("embedding export shape/epoch mismatch")
        return frame

    spot = load("spot", meta["spot_ids"], representation)
    peak = load(
        "peak",
        meta["peak_ids"],
        representation,
    )
    motif = (
        load(
            meta["dna_kind"],
            meta["dna_ids"],
            representation,
        )
        if meta["dna_ids"]
        else None
    )
    coordinates = (
        np.load(path / "coordinates.npy", allow_pickle=False)
        if (path / "coordinates.npy").exists()
        else None
    )
    return EmbeddingSet(
        spot,
        peak,
        motif,
        meta["dna_kind"] or "motif",
        coordinates,
        tuple(meta["slice_ids"]),
        {
            "run_path": str(path.resolve()),
            "run_fingerprint": manifest["fingerprint"],
            "epoch": epoch,
            "representation": representation,
            "feature_representation": (
                representation
            ),
            "joint_space_compatible": True,
            "training_seed": manifest["metadata"]["config"]["seed"],
        },
    )


def import_legacy(
    spot_path,
    peak_path,
    metadata_h5ad,
    motif_path=None,
    dna_kind="motif",
    representation="UNKNOWN",
    epoch=None,
):
    import anndata as ad

    if representation not in ("UNKNOWN", "propagated", "ego"):
        raise ValueError("invalid declared legacy representation")
    if epoch is not None and (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1):
        raise ValueError("declared legacy epoch must be a positive integer")
    source = ad.read_h5ad(metadata_h5ad, backed="r")
    try:
        spot = read_embeddings(spot_path, tuple(source.obs_names))
        peak = read_embeddings(peak_path) if peak_path is not None else None
        if peak is not None:
            from .preprocessing import peak_keys

            known = set(peak_keys([p for p in source.var_names if str(p).startswith("chr")]))
            if not set(peak_keys(peak.index)) <= known:
                raise ValueError("legacy peak IDs are absent from metadata input")
        coords = np.asarray(source.obsm["spatial"]).copy() if "spatial" in source.obsm else None
        if all(k in source.obs for k in ("slice_id", "sample_id")):
            if not np.array_equal(
                pd.factorize(source.obs["slice_id"], sort=False)[0],
                pd.factorize(source.obs["sample_id"], sort=False)[0],
            ):
                raise ValueError("conflicting slice_id and sample_id partitions")
        key = next((k for k in ("slice_id", "sample_id") if k in source.obs), None)
        slices = tuple(source.obs[key].astype(str)) if key else ("sample",) * len(spot)
    finally:
        source.file.close()
    motif = read_embeddings(motif_path) if motif_path else None
    paths = {"spot": spot_path, "peak": peak_path, "dna": motif_path, "metadata": metadata_h5ad}
    provenance = {
        "import_protocol": "explicit_legacy_tsv_v1",
        "representation": representation,
        "joint_space_compatible": True,
        "epoch": epoch if epoch is not None else "UNKNOWN",
        "representation_and_epoch_evidence": (
            "caller_declared_unverified"
            if representation != "UNKNOWN" or epoch is not None
            else "UNKNOWN"
        ),
        "dna_identity_evidence": (
            "self_reported_TSV_IDs_without_annotation_verification" if motif_path else None
        ),
        "sources": {
            k: {"path": str(Path(p).resolve()), "sha256": file_sha256(p)}
            for k, p in paths.items()
            if p is not None
        },
    }
    return EmbeddingSet(spot, peak, motif, dna_kind, coords, slices, provenance)


@dataclass(frozen=True)
class TrainingRun(StageArtifact):
    KIND = 'training_run'

    @classmethod
    def load(cls, path):
        result=super().load(path)
        status = json.loads((result.path / "status.json").read_text())
        if status.get("state") != "complete":
            raise ValueError("training run is incomplete")
        return result

    @property
    def final_epoch(self):
        return self.manifest["metadata"]["final_adopted_output_epoch"]

    @validated_operation
    def load_embeddings(self, epoch=None, representation="propagated"):
        self.verify()
        if epoch is not None and (type(epoch) is not int or epoch < 1):
            raise ValueError("epoch must be a positive integer")
        return from_run(self.path, epoch, representation)


@dataclass(frozen=True)
class InputArtifact:
    path: Path
    metadata: dict

    @property
    def paths(self):
        return {key: (self.path / entry["path"]).resolve()
                for key, entry in self.metadata["files"].items()}

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        verify_artifact(path, "input_identity")
        metadata = json.loads((path / "inputs.json").read_text())
        result = cls(path, metadata)
        for key, candidate in result.paths.items():
            if not candidate.is_file() or file_sha256(candidate) != metadata["files"][key]["sha256"]:
                raise ValueError(f"input source changed or unavailable: {key}")
        return result


load_run = TrainingRun.load


load_inputs = InputArtifact.load
