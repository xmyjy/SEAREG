"""Explicit peak–motif generation, independent of a training configuration."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Explicit local peak–motif annotation; no training is launched."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--atac", type=Path, help="counts H5AD; filter before scanning")
    source.add_argument("--prepared", type=Path, help="saved network with the final peak axis")
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("legacy", "fimo"), default="legacy",
                        help="legacy preserves the historical scanner; fimo preserves complete candidate sites")
    parser.add_argument("--motifs", type=Path, help="explicit local DNA MEME file for FIMO")
    parser.add_argument("--fimo", default=None, help="local FIMO executable")
    parser.add_argument("--motif-pseudocount", type=float, default=None)
    parser.add_argument("--graph-selection", choices=("all_candidates", "site_p"))
    parser.add_argument("--graph-p-cutoff", type=float)
    parser.add_argument("--graph-output", type=Path, help="separate explicitly selected candidate graph")
    parser.add_argument("--species", nargs="+", default=["Homo sapiens", "Mus musculus"])
    parser.add_argument("--p-cutoff", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--slice-key", default=None)
    parser.add_argument("--no-spatial", action="store_true")
    args = parser.parse_args(argv)
    try:
        from .config import PreprocessConfig
        if args.prepared is not None and (args.no_spatial or args.slice_key is not None):
            raise ValueError("--prepared fixes its peak axis; omit preprocessing options")
        preprocessing = (
            PreprocessConfig(use_spatial=not args.no_spatial, slice_key=args.slice_key)
            if args.atac is not None else None
        )
        source = args.atac if args.atac is not None else args.prepared
        if args.method == "fimo":
            from .annotation import annotate_peak_motifs, export_motif_graph
            if args.motifs is None:
                raise ValueError("--method fimo requires --motifs")
            if (args.graph_selection is None) != (args.graph_output is None):
                raise ValueError("--graph-selection and --graph-output must be supplied together")
            if args.graph_selection == "site_p":
                if args.graph_p_cutoff is None or not 0 < args.graph_p_cutoff <= args.p_cutoff:
                    raise ValueError("site_p requires --graph-p-cutoff no larger than --p-cutoff")
            elif args.graph_p_cutoff is not None:
                raise ValueError("--graph-p-cutoff requires --graph-selection site_p")
            result = annotate_peak_motifs(source, args.fasta, args.assembly, args.motifs,
                args.output, preprocessing=preprocessing, fimo=args.fimo or "fimo",
                p_cutoff=args.p_cutoff, motif_pseudocount=args.motif_pseudocount or 0,
                timeout=args.timeout)
            graph = None if args.graph_selection is None else export_motif_graph(
                result, args.graph_output, selection=args.graph_selection, p_cutoff=args.graph_p_cutoff)
            response = {"state": "complete", "path": str(result.path),
                        "motif_file": None if graph is None else str(graph / "freq_motif.h5"),
                        "confidence": "sequence candidates, not a high-confidence relation set"}
        else:
            from .sequence import scan_peak_motifs_legacy
            if any(value is not None for value in (args.motifs, args.fimo, args.motif_pseudocount,
                       args.graph_selection, args.graph_output, args.graph_p_cutoff)):
                raise ValueError("FIMO options require --method fimo")
            path = scan_peak_motifs_legacy(source, args.fasta, args.assembly, args.output,
                preprocessing=preprocessing, species_names=tuple(args.species),
                p_cutoff=args.p_cutoff, threads=args.threads, rscript=args.rscript,
                timeout=args.timeout)
            response = {"state": "complete", "path": str(path),
                        "motif_file": str(path / "freq_motif.h5")}
        print(json.dumps(response, indent=2))
        return 0
    except Exception as exc:
        print(f"SEAREG annotation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
